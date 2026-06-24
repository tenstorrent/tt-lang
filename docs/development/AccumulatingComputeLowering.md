# Accumulating Compute Lowering

This document describes how the tt-lang compiler lowers operations that
accumulate results across multiple invocations - reductions, matmul
K-accumulation, and user-written `+=` loops - onto the Tenstorrent
compute engines.

## Overview

An accumulation in tt-lang can be compiled in three ways with the same
program semantics and different thread-local data movement:

1. Keep the partial value in the destination register file (DST).
2. Add packed output tiles into L1 through the packer.
3. Carry the partial value through an explicit compiler-managed dataflow
   buffer (DFB).

The current compiler lowering recognizes these accumulation forms:

- `reduce_tile` and `matmul_tiles` accumulate per-tile over a reduction
  dim. The `dst-accumulation` pass option on `ttl-lower-to-loops`
  selects DST (loops reordered so DST spans the reduction) or L1 (loops
  in declaration order with per-iteration pack acc). `reduce_max` is
  L1-incompatible (L1 acc only adds) and is always lowered to DST acc.

- User-written `out_blk += ...` loops lower to L1 accumulation. The
  `TTKernelInsertL1Accumulation` pass brackets each annotated loop
  group with `pack_reconfig_l1_acc` calls.

- The store-then-accumulate pattern (`out_blk.store(v); for K-1: out_blk
  += ...`) is lowered via L1 acc with a modified guard sequence: the
  pre-group reconfig enables L1 acc so iteration 0 accumulates onto the
  prior-pack value rather than overwriting it. `precededByNonAccumulatingPack`
  detects the preceding non-accumulating pack.

The accumulation-scope IR declares which destination tensor views participate
in an accumulation region, plus the initial-state policy for each output. Later
lowering can select DST, L1 packer accumulation, or explicit DFB state without
reconstructing that policy from neighboring stores or DFB operations.

The rest of this document details each piece: accumulation scopes,
`DstSectionOp` as the IR primitive that keeps DST live, the choice between DST
and L1 accumulation, the emitted loop structure, per-op init insertion, and the
L1-acc guard placement.

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
compiler needs value-style accumulation scopes.

The verifier is structural:

- initial-mode count equals output count;
- init modes have matching init operands;
- init operand types match their corresponding outputs;
- the body has one block argument and one yielded value per output;
- body arguments and yielded values match their output types;
- nested `ttl.accumulation_scope` is rejected until nested accumulation
  semantics are defined.

The verifier does not prove that stores target the declared outputs or that
control flow reaches an update. Those are nonlocal formation and strategy
lowering responsibilities.

Initial modes have these meanings:

- `overwrite`: the first executed contribution defines the accumulator value.
- `accumulate_existing`: an existing value in the output location
  participates in the result.
- `init`: an init operand seeds the accumulator, independent of the final
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

## DstSectionOp

`ttl.dst_section` demarcates a DST register acquisition scope. All
tile compute ops and stores in the body share one acquire/release
cycle. When lowered to TTKernel (`expandDstSections` in
`ConvertTTLToTTKernel`), the body is split at the first `TileStoreOp`
into math and pack phases:

    acquire -> [math ops] -> commit -> wait -> [pack ops] -> release

Three placement modes:

- **Non-subblocked**: one `dst_section` per tile loop iteration
- **Subblocked**: one `dst_section` wrapping the unrolled tile sequence
- **Accumulating**: one `dst_section` per parallel iteration, with
  the reduction loop inside

All computes use `DstSectionOp`, including matmul (`LowerMatmulBlock`).

## DST vs L1 accumulation

Two mechanisms for multi-tile reduction:

**DST accumulation** (`dst-accumulation=true`): Reorders loops so
parallel dims are outer and reduction dims are inner. `DstSectionOp`
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
correct for sum.

## Loop structure

### DST accumulation (parallel-outer, reduction-inner)

`generateAccumulatingLoops` separates parallel and reduction dims
from `iterator_types`:

```
for each parallel dim:           // output tile iteration
    dst_section {
        for each reduction dim:  // accumulate into DST
            <tile ops>
        <stores with placeholder tile + explicit dst_index>
    }
```

Stores use a placeholder tile value (via `UnrealizedConversionCastOp`)
with an explicit `dst_index` operand, since the SSA tile value from
`reduce_tile` is loop-local.

### L1 accumulation (declaration-order loops)

```
for each dim (declaration order):
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
`pack_reconfig_l1_acc` calls. The standard sequence disables L1 acc
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
onto. The reconfig before the group becomes enable, and the
per-iteration conditional enable on the root loop is omitted because
every iteration must accumulate from iteration 0 onward:

```
pack_tile(...)                  // prior pack runs with L1 acc disabled
pack_reconfig_l1_acc(1)
for iv = lb..ub:
    ...pack...
pack_reconfig_l1_acc(0)
```

`precededByNonAccumulatingPack` selects between the two sequences by
walking backward over the L1-acc loop's parent block and classifying
each predecessor op as a contributor (a pack that leaves a prior value
in L1) or a boundary (an op that resets or shadows the L1 slot, or one
whose execution semantics the walk cannot model). See the helper's
implementation for the exact classification rules.

The pass is idempotent: a prior run leaves a `pack_reconfig_l1_acc`
either inside the L1-acc loop body or immediately preceding the loop,
and the second run detects either signal and returns.

## Per-op init insertion

`TTKernelInsertInits` uses two targeted walks instead of a block walk:

1. `walk(TileRegsAcquireOp)`: iterates top-level ops between acquire and
   release. Each top-level op may contain compute ops in nested regions
   (e.g., `reduce_tile` inside a reduction `scf.for`); these are
   discovered via `op.walk()`. Init is inserted before the flat
   container op. Consecutive ops with the same init key share one
   init (forward-order dedup via `prevKey`).

2. `walk(func::FuncOp)`: handles compute ops outside sync regions
   (unit tests). Skips ops already processed by walk 1.

Bcast, reduce, and transpose inits resolve their output DFB from a
`ttl.*_output_cb_index` attribute propagated during TTL-to-TTKernel
conversion.

## IR trace: 2x2 reduce_sum along dim 0

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

After TTKernel conversion + insert-inits + L1 acc:
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
