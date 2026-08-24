# Accumulating Compute Lowering

This document describes how the tt-lang compiler lowers operations that
accumulate results across multiple invocations - reductions, matmul
K-accumulation, and user-written `+=` loops - onto the Tenstorrent
compute engines.

## Overview

An accumulation in tt-lang can be compiled in three ways with the same
program results and different thread-local data movement:

1. Keep the partial value in the destination register file (DST).
2. Add packed output tiles into L1 through the packer.
3. Carry the partial value through an explicit compiler-managed dataflow
   buffer (DFB).

The current compiler lowering recognizes these accumulation forms:

- `reduce_tile` and `matmul_tiles` accumulate per tile over a reduction
  dimension. The `dst-accumulation` pass option on `ttl-lower-to-loops`
  selects DST (loops reordered so DST spans the reduction) or L1 (loops in
  declaration order with per-iteration packer accumulation). `reduce_max` is
  L1-incompatible because packer accumulation only adds, and is always lowered
  to DST accumulation.

- User-written `out_blk += ...` loops lower to L1 accumulation. The
  `TTKernelInsertL1Accumulation` pass brackets each annotated loop
  group with `pack_reconfig_l1_acc` calls.

- The store-then-accumulate pattern (`out_blk.store(v); for K-1: out_blk
  += ...`) is lowered via L1 accumulation with a modified guard sequence. The
  configuration before the loop group enables packer accumulation, so iteration
  0 adds to the value packed before the loop instead of overwriting it.
  `precededByNonAccumulatingPack` detects that preceding non-accumulating pack.

- A loop-carried additive tensor recurrence (`acc = acc + contribution`)
  can be accumulated in DST when the accumulator is initialized from a
  DFB-backed tensor and each recurrence update has one DFB-backed
  contribution. The contribution may be acquired once per iteration or held
  resident across the loop. Loop-carried tensor iter_args that do not match
  this additive recurrence remain ordinary tensor state;
  `ttl-materialize-loop-state` stores the initial value, updates a state DFB
  each iteration, and reloads the final value after the loop.

The accumulation-scope IR declares which destination tensor views participate
in an accumulation region, plus the initial-state policy for each output. Later
lowering can select DST, L1 packer accumulation, or explicit DFB state without
reconstructing that policy from neighboring stores or DFB operations.

The rest of this document details each piece: accumulation scopes,
`DstSectionOp` as the IR primitive that keeps DST live, the choice between DST
and L1 accumulation, the emitted loop structure, per-op init insertion, and the
L1 accumulation guard placement.

## Accumulation Scope IR

`ttl.accumulation_scope` declares the accumulation contract for one or more
destination tensor views. It records which outputs share a region, how each
output is initialized, and which value returned by the region updates each
output. The op does not select the storage mechanism used for partial values.
It has:

- `outputs`: destination tensor views governed by the accumulation policy;
- `inits`: init operands for outputs whose initial mode is `init`;
- `initial_modes`: one accumulation initial-mode per output (`overwrite`,
  `accumulate_existing`, or `init`);
- `body`: a single-block region with one block argument and one yielded value
  per output.

The op has `RecursiveMemoryEffects`; its effects are the effects of the body.
It produces no tensor results. Tensor result support is deferred until the
compiler needs accumulation scopes that return SSA values instead of only
governing stores.

The verifier is structural:

- initial-mode count equals output count;
- init modes have matching init operands;
- init operand types match their corresponding outputs;
- the body has one block argument and one yielded value per output;
- body arguments and yielded values match their output types;
- nested `ttl.accumulation_scope` is rejected until nested accumulation
  semantics are defined.

The verifier does not prove that stores target the declared outputs or that
control flow reaches an update. Those checks require surrounding IR and are
performed by the passes that form and lower accumulation scopes.

Initial modes have these meanings:

- `overwrite`: the first executed contribution defines the accumulator value.
- `accumulate_existing`: an existing value in the output location
  participates in the result.
- `init`: an init operand initializes the accumulator, independent of the final
  output location.

Example:

```mlir
ttl.accumulation_scope
    outs(%out_view : tensor<...>)
    inits(%init : tensor<...>)
{
^bb0(%acc: tensor<...>):
  %next = ttl.add %acc, %contribution : tensor<...>, tensor<...> -> tensor<...>
  ttl.yield %next : tensor<...>
} initial_modes([init])
```

Accumulation scopes expose accumulator state as block arguments and return the
updated state through `ttl.yield`. Cross-output dependence is represented by
ordinary SSA use-def edges between yielded values.

```mlir
ttl.accumulation_scope
    outs(%out0, %out1 : tensor<...>, tensor<...>)
    inits(%init0, %init1 : tensor<...>, tensor<...>)
{
^bb0(%acc0: tensor<...>, %acc1: tensor<...>):
  %next0 = ttl.add %acc0, %acc1 : tensor<...>, tensor<...> -> tensor<...>
  %next1 = ttl.add %acc1, %next0 : tensor<...>, tensor<...> -> tensor<...>
  ttl.yield %next0, %next1 : tensor<...>, tensor<...>
} initial_modes([init, init])
```

`AccumulationScopeOpInterface` gives consumers a common contract for ops that
declare accumulation outputs and policies. The initial implementation is
`ttl.accumulation_scope`; later PRs extend the same contract to structured
reductions where the reduction body already represents accumulation.

## Loop-Carried Tensor State

A Python `for` loop that reassigns a tensor variable read on a later
iteration (`acc = acc + x`, `state = ttl.math.relu(state)`) compiles to an
`scf.for` with a ranked-tensor `iter_arg`. `ttl-materialize-loop-state`
eliminates those tensor iter_args before compute lowering by creating
compiler-managed DFB state:

```
store init -> state DFB
for ...:
    wait/attach state DFB
    compute next state
    reserve/store next state -> state DFB
wait/attach final state DFB
```

The pass preserves non-tensor loop iter_args. It also preserves zero-trip
loop semantics because the initial value is stored before the rewritten loop
and the final value is read after the loop.

## Tensor Recurrence Accumulation

A tensor recurrence is an `scf.for` iter_arg whose value is computed in one
iteration and read in a later iteration. General tensor recurrences are lowered
through compiler-managed DFB state, as described above. The compiler has a
specialized DST lowering for the additive recurrence:

```
acc = acc + contribution
```

The DST form keeps the accumulator in the destination register file for the
full source loop and packs the final value once. This avoids the per-iteration
pack, wait, and reload required by general DFB-state materialization.

The pipeline recognizes eligible recurrences before `ttl-materialize-loop-state`.
It represents the recurrence with `ttl.accumulation_scope` using `init`
initial mode, then lowers that scope directly to one `ttl.dst_section` whose
accumulator stays resident in DST across the source loop. A recurrence that is
not represented as an accumulation scope remains an ordinary loop-carried
tensor value and uses the general DFB-state lowering.

### Eligibility

The DST recurrence form requires all of the following:

- one tensor loop-carried accumulator, updated only by
  `ttl.add(acc, contribution)` or the commuted form;
- one final non-accumulating `ttl.store` of the loop result;
- one dataflow-buffer-backed contribution matching the accumulator type;
- the contribution is either streamed through a loop-local `ttl.cb_wait` and
  matching `ttl.cb_pop`, or acquired before the loop and reused as a resident
  loop-invariant value;
- no loop-local stores or other side effects;
- a DFB-backed init value;
- contribution `ttl.cb_wait` and `ttl.cb_pop` operations use the block size
  encoded in the DFB type, without a `num_tiles` attribute;
- a static nonzero output tile count that fits the logical DST capacity.

The source-loop trip count may be static, dynamic, or zero. Streamed
contributions keep capacity independent of the trip count by preserving the
per-iteration `ttl.cb_wait` / `ttl.cb_pop` pair. Resident contributions hold
one contribution block across the loop and release it with `ttl.cb_pop` after
the final use.

A recurrence that fails any condition is not diagnosed: formation leaves it as an
ordinary loop-carried value for general DFB-state materialization, which is
correct but incurs the per-iteration pack and reload. In particular, a resident
contribution released before the loop's final use (an early pop) falls back this
way instead of forming a DST section, because the block must stay live for every
iteration.

### Lowering

The lowered section keeps every output tile's accumulator slot resident in DST
for the source loop. Streamed contributions preserve the source loop's
per-iteration acquire/release protocol:

```mlir
ttl.dst_section {
  %init_tile = tensor.extract %init[%i, %j]
  %token, %acc = ttl.copy_tile %init_tile[%i, %j] into dst[%idx]

  scf.for ... {
    %contribution = ttl.cb_wait %contribution_dfb
    %contribution_tile = tensor.extract %contribution[%i, %j]
    ttl.tile_accumulate %acc, %contribution_tile add into dst[%idx]
    ttl.cb_pop %contribution_dfb
  }

  ttl.tile_store %placeholder, %out_view[%i, %j] from dst[%idx]
}
```

For a resident contribution, the contribution is acquired before the section and
read directly inside the loop. If the input IR does not already contain the
matching release (`ttl.cb_pop`) after the final use, lowering emits one after
the section:

```mlir
%contribution = ttl.cb_wait %contribution_dfb

ttl.dst_section {
  %init_tile = tensor.extract %init[%i, %j]
  %token, %acc = ttl.copy_tile %init_tile[%i, %j] into dst[%idx]

  scf.for ... {
    %contribution_tile = tensor.extract %contribution[%i, %j]
    ttl.tile_accumulate %acc, %contribution_tile add into dst[%idx]
  }

  ttl.tile_store %placeholder, %out_view[%i, %j] from dst[%idx]
}

ttl.cb_pop %contribution_dfb
```

For multi-tile output blocks, the section contains one stable DST index per
output tile. Each init copy, per-iteration `ttl.tile_accumulate`, and final
store for a tile use that same index. This requires the output tile count to fit
the logical DST capacity: f32 uses four slots and bf16 uses eight slots in the
default double-buffered mode.

`ttl.tile_accumulate` denotes in-place accumulation in DST. The accumulator
operand and result share one DST slot, and the contribution remains
dataflow-buffer backed. For `combiner = add`, TTKernel lowering emits
`binary_dest_reuse_tiles(..., Add, DestToSrcA)`; other combiners require their
own legality and TTKernel lowering rules.

## DstSectionOp

`ttl.dst_section` demarcates a DST register acquisition scope. All
tile compute ops and stores in the body share one acquire/release
cycle. When lowered to TTKernel (`expandDstSections` in
`ConvertTTLToTTKernel`), the body is split at the first `TileStoreOp`
into math and pack phases:

    acquire -> [math ops] -> commit -> wait -> [pack ops] -> release

`ttl.dst_section` appears in four lowering forms:

- **Non-subblocked**: one `dst_section` per tile loop iteration
- **Subblocked**: one `dst_section` wrapping the unrolled tile sequence
- **Accumulating compute**: one `dst_section` per parallel iteration, with
  the reduction loop inside
- **Tensor recurrence**: one `dst_section` for the output block whose
  accumulator tiles stay resident in DST, with the source recurrence loop
  inside

Tile compute lowering uses `DstSectionOp`, including matmul lowering
(`LowerMatmulBlock`).

## DST vs L1 accumulation

Two mechanisms for multi-tile reduction:

**DST accumulation** (`dst-accumulation=true`): Reorders loops so
parallel dimensions are outer and reduction dimensions are inner. `DstSectionOp`
wraps the reduction loop, so DST persists across iterations. One
pack after the entire reduction. More efficient (no L1 round-trip)
but holds the output DFB reserve longer.

**L1 accumulation** (`dst-accumulation=false`): Loops in declaration
order with per-tile `DstSectionOp`. Each iteration acquires DST,
computes, packs. `pack_reconfig_l1_acc(1)` makes the packer add to
the existing L1 value from the second iteration onward. See the
"Guard placement around L1 accumulation loops" section below for the
full enable/disable sequence and how it changes when a non-accumulating
pack precedes the loop.

Selection: the `dst-accumulation` pass option on `ttl-lower-to-loops`
controls the mode. The pipeline maps `maximize_dst` to this option.
`reduce_max` always uses DST accumulation because L1 accumulation
(`pack_reconfig_l1_acc`) accumulates via addition, which is only
correct for sum. Computes containing `ttl.tile_accumulate` (tensor
recurrence accumulation, see Tensor Recurrence Accumulation) also
always use DST accumulation, regardless of the option.

## Loop structure

### DST accumulation (parallel-outer, reduction-inner)

`generateAccumulatingLoops` separates parallel and reduction dimensions
from `iterator_types`:

```
for each parallel dimension:     // output tile iteration
    dst_section {
        for each reduction dimension: // accumulate into DST
            <tile ops>
        <stores with placeholder tile + explicit dst_index>
    }
```

Stores use a placeholder tile value (via `UnrealizedConversionCastOp`)
with an explicit `dst_index` operand, since the SSA tile value from
`reduce_tile` is loop-local.

### L1 accumulation (declaration-order loops)

```
for each dimension (declaration order):
    dst_section {
        <tile ops>
        <stores>
    }
```

Reduction loops are annotated with `ttl.reduction_loop`.
`TTKernelInsertL1Accumulation` inserts the guard after
`tile_regs_acquire` inside reduction loops.

### Guard placement around L1 accumulation loops

`TTKernelInsertL1Accumulation` brackets each loop group (consecutive
sibling loops sharing a pack CB, collected by `collectLoopGroups`) with
`pack_reconfig_l1_acc` calls. The standard sequence disables L1 accumulation
before the group, conditionally enables it inside the first iteration's
last pack so subsequent iterations accumulate, and disables it again
after the group:

```
pack_reconfig_l1_acc(0)
for iv = lb..ub:
    ...pack...
    if iv == lb: pack_reconfig_l1_acc(1)
pack_reconfig_l1_acc(0)
```

When a non-accumulating pack into the loop's pack CB precedes the loop in
the same parent block, L1 already holds a value the loop must accumulate
onto. The call before the group becomes `pack_reconfig_l1_acc(1)`, and the
per-iteration conditional enable on the loop group is omitted because every
iteration must accumulate from iteration 0 onward:

```
pack_tile(...)                  // prior pack runs with L1 accumulation disabled
pack_reconfig_l1_acc(1)
for iv = lb..ub:
    ...pack...
pack_reconfig_l1_acc(0)
```

`precededByNonAccumulatingPack` selects between the two sequences by
walking backward over the L1 accumulation loop's parent block and classifying
each predecessor op as a contributor (a pack that leaves a prior value
in L1) or a boundary (an op that resets or shadows the L1 slot, or one
whose ordering or side effects the walk cannot prove). See the helper's
implementation for the exact classification rules.

The pass is idempotent: a prior run leaves a `pack_reconfig_l1_acc`
either inside the L1 accumulation loop body or immediately preceding the loop,
and the second run detects either signal and returns.

## Per-op init insertion

`TTKernelInsertInits` uses two targeted walks instead of a block walk:

1. `walk(TileRegsAcquireOp)`: iterates top-level ops between acquire and
   release. Each top-level op may contain compute ops in nested regions
   (e.g., `reduce_tile` inside a reduction `scf.for`); these are
   discovered via `op.walk()`. Init is inserted before the top-level
   operation that contains the compute op. Consecutive ops with the same
   initialization parameters share one init (`prevKey` tracks the previous
   parameter key in forward order).

2. `walk(func::FuncOp)`: handles compute ops outside sync regions
   (unit tests). Skips ops already processed by walk 1.

Broadcast, reduce, and transpose inits resolve their output DFB from a
`ttl.*_output_cb_index` attribute propagated during TTL-to-TTKernel
conversion.

## IR trace: 2x2 reduce_sum along dimension 0

Input: `tensor<2x2xtile>`, scaler: `tensor<1x1xtile>`,
output: `tensor<1x2xtile>`.

### DST accumulation (dst-accumulation=true)

After LowerToLoops:
```mlir
scf.for %j = %c0 to %c2 step %c1 {       // parallel
    ttl.dst_section {
        scf.for %i = %c0 to %c2 step %c1 { // reduction
            %in = tensor.extract %inp[%i, %j]
            %sc = tensor.extract %scaler[%c0, %c0]
            %out = tensor.extract %init[%c0, %j]
            ttl.tile_reduce %in, %sc, %out sum reduce_dim_col into dst[%c0]
        } {ttl.reduction_loop, ttl.tile_loop_stride = 2}
        ttl.tile_store %placeholder, %view[%c0, %j] from dst[%c0]
    }
} {ttl.tile_loop_stride = 1}
```

After TTKernel conversion + insert-inits:
```
init_sfpu(cb0, cb2)
for j = 0..2:                              // parallel
    tile_regs_acquire()
    reduce_init(cb0, cb1, cb2, SUM, REDUCE_COL)
    for i = 0..2:                          // reduction (DST persists)
        reduce_tile(cb0, cb1, i*2+j, 0, 0, SUM, REDUCE_COL)
    reduce_uninit()
    tile_regs_commit() / tile_regs_wait()
    pack_tile(0, cb2, j)
    tile_regs_release()
cb_push_back(cb2, 2)
```

### L1 accumulation (dst-accumulation=false)

After LowerToLoops:
```mlir
scf.for %i = %c0 to %c2 step %c1 {       // reduction (declaration order)
    scf.for %j = %c0 to %c2 step %c1 {   // parallel
        ttl.dst_section {
            ttl.tile_reduce ... into dst[%c0]
            ttl.tile_store ...
        }
    } {ttl.tile_loop_stride = 1}
} {ttl.reduction_loop, ttl.tile_loop_stride = 2}
```

After TTKernel conversion + insert-inits + L1 accumulation:
```
init_sfpu(cb0, cb2)
for i = 0..2:                              // reduction
    for j = 0..2:                          // parallel
        tile_regs_acquire()
        if (i != 0) pack_reconfig_l1_acc(1)
        reduce_init(...)
        reduce_tile(cb0, cb1, i*2+j, 0, 0, SUM, REDUCE_COL)
        reduce_uninit()
        tile_regs_commit() / tile_regs_wait()
        pack_tile(0, cb2, j)               // overwrites or adds to L1
        tile_regs_release()
cb_push_back(cb2, 2)
```
