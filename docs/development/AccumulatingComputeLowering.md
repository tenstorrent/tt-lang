# Accumulating Compute Lowering

This document describes how the tt-lang compiler lowers operations that
accumulate results across multiple invocations - reductions, matmul
K-accumulation, and user-written `+=` loops - onto the Tenstorrent
compute engines.

## Overview

Tenstorrent hardware supports two accumulation mechanisms, and the
compiler maps each accumulation source to one of them.

**DST register accumulation.** The compute engines hold partial results
in a destination register file (DST) that persists across tile ops as
long as it stays acquired (not released). Per-tile accumulation (matmul
K-reduction, reduce across a reduction dim) happens inside one
acquire/release cycle.

**L1 packer accumulation.** The pack unit can add each packed tile to
the existing L1 value instead of overwriting, controlled by
`pack_reconfig_l1_acc(1)` (enable) and `pack_reconfig_l1_acc(0)`
(disable). Accumulation across separate acquire/release cycles uses
this mechanism because DST is released between iterations.

The compiler surface covers these accumulation sources and initial-state
forms:

- `reduce_tile` and `matmul_tiles` accumulate per-tile over a reduction
  dim. The `dst-accumulation` pass option on `ttl-lower-to-loops`
  selects DST (loops reordered so DST spans the reduction) or L1 (loops
  in declaration order with per-iteration pack acc). `reduce_max` is
  L1-incompatible (L1 acc only adds) and is always lowered to DST acc.

- User-written `out_blk += ...` loops lower to L1 accumulation. The
  compiler forms a `ttl.accumulation_scope`, records whether iteration 0
  overwrites or accumulates onto a prior value, then lowers the scope to
  loop metadata consumed by `TTKernelInsertL1Accumulation`. `+=` on a
  plain tensor variable is rewritten by `visit_AugAssign` to
  `acc = acc + x` and follows the loop-carried recurrence flow below.

- Loop-carried additive recurrences inside an `scf.for` (`acc = acc +
  x` or `acc = x + acc`, plain tensor target) are formed as
  `ttl.accumulation_scope` before general tensor state materialization.
  `ttl-lower-accumulation-scopes` selects DST residency when legal and
  otherwise lowers to a pre-loop initial store plus in-loop
  accumulating `ttl.store` operations annotated for L1 packer
  accumulation.

- The store-then-accumulate pattern (`out_blk.store(v); for K-1:
  out_blk += ...`) is lowered via L1 acc with a modified
  reconfiguration sequence: the reconfiguration before the group enables
  L1 acc so iteration 0 accumulates onto the previously packed value
  rather than overwriting it. This is represented before TTKernel
  lowering by `ttl.l1_acc_initial = accumulate_existing` on the
  accumulation loop.

The rest of this document details each piece: semantic accumulation
scopes, loop-carried tensor state elimination
(`ttl-materialize-loop-state`), `DstSectionOp` as the IR primitive that
keeps DST live, the choice between DST and L1 accumulation, the emitted
loop structure, per-op init insertion, and the L1-acc reconfiguration
placement.

## Loop-carried tensor state and accumulation scopes

A Python `for` loop that reassigns a tensor variable read on the next
iteration (`acc = acc + x`, `acc = relu(acc)`) compiles to an `scf.for`
with a ranked-tensor `iter_arg`. The accumulation pipeline first forms
`ttl.accumulation_scope` around recognized additive recurrences. General
tensor recurrences that remain after strategy lowering are eliminated by
`ttl-materialize-loop-state` before compute lowering.

### Why tensor iter_args, not DFBs directly

The frontend could emit DFB state directly from the AST and skip the
tensor `iter_arg` form. It does not, for the following reasons.

**Layering.** A rebound Python loop variable is a value carried to the
next iteration; a tensor `scf.for` iter_arg is its direct translation.
Emitting DFBs would force the AST walker to choose CB indices, block
counts, and slot flow control, which are backend concerns.

**Strategy decided in MLIR.** Additive-vs-general classification depends
on use-def structure - the single add, its single use, the consumer
store, and the reserve feeding it - which
`matchAdditiveTensorAccumulation` matches reliably and the AST cannot.
The frontend stays a correctness-only component that identifies
loop-carried variables; MLIR scope formation and strategy lowering
choose DST, L1 packer accumulation, or general DFB state.

**Shared state materialization.** Additive, elementwise, and tuple
recurrences are all tensor iter_args at the frontend. Additive
recurrences are consumed by accumulation strategy lowering, while
remaining tensor state uses the same reserve/store/wait/attach helper
code as `ttl-insert-intermediate-dfbs` through `DFBMaterialization`.

Tensor-level loops also remain subject to standard canonicalization, CSE,
and dead-code elimination, which do not apply to side-effecting DFB ops.

### Why not one-shot bufferization

Upstream MLIR eliminates tensor `scf.for` iter_args with one-shot
bufferization: `scf::ForOp`'s `BufferizableOpInterface` implementation
threads each tensor iter_arg through the loop as a memref and drops the
tensor result. tt-lang does not bufferize tensors to memref. On-chip a
tensor value lives in the DST register file or in a dataflow buffer (DFB)
accessed through `cb_reserve`/`store`/`cb_wait`/`attach_cb`; neither is a
memref. This pass eliminates the tensor iter_arg by realizing the carried
state as DFB state. Additive recurrences are handled before this pass by
`ttl.accumulation_scope` lowering because strategy selection must choose
between DST residency and L1 packer accumulation. Generic bufferization
would emit memref load/store and would not produce either hardware
accumulation mechanism. The pass therefore implements iter_arg
elimination directly against DFB ops. It does not reuse bufferization's
conflict/aliasing analysis; the
double-buffer assumption below stands in for it.

### Lowering strategies

Additive tensor recurrences use `ttl-form-accumulation-scopes` followed
by `ttl-lower-accumulation-scopes`.

**DST strategy:** when the recurrence satisfies the DST legality rules,
the lowering creates a reduction-style `ttl.compute` with
`ttl.tile_accumulate_add`. The generated `ttl.dst_section` spans the
reduction loop, so the accumulator stays in DST until the final store.

**L1 packer strategy:** when DST is not selected, the lowering creates a
pre-loop non-accumulating store of the initial value and per-iteration
accumulating stores into the same output reservation. The loop is
annotated with `ttl.l1_acc_loop`, `ttl.l1_acc_initial =
accumulate_existing`, and `ttl.l1_acc_scope_id`.

**General recurrence:** any tensor iter_arg that remains after
accumulation scope lowering uses compiler-allocated double-buffered DFB
state. The init is stored before the loop; each iteration consumes the
current state (`cb_wait`/`attach_cb`), computes, and produces the next
state (`cb_reserve`/`store`); a post-loop `cb_wait`/`attach_cb` yields
the final state value that replaces the loop result.

### Invariants

Preconditions:

- Runs on `func.func` nested in a `ModuleOp`, once per `scf.for`.
- Additive scope formation matches only when all of the following hold;
  any tensor iter_arg failing them remains for general DFB state
  materialization:
  - the loop result has exactly one use, a non-accumulate `ttl.store`;
  - the yielded value is a single-use `ttl.add` in the loop body;
  - the iter_arg is one add operand and has no other use;
  - the other operand (the contribution) is not the iter_arg;
  - the store's destination is a `cb_reserve` whose only other uses are
    result-unused `attach_cb`s;
  - that `cb_reserve` and the store sit in the loop's parent block.

Postconditions:

- The rewritten `scf.for` carries no tensor iter_args or results;
  non-tensor iter_args keep their relative order.
- Additive scope lowering consumes every formed `ttl.accumulation_scope`.
  `ttl-materialize-loop-state` then removes any remaining tensor iter_arg,
  so no tensor iter_arg reaches compute lowering.

Structural invariants:

- Each compiler-allocated state DFB is created with block count 2
  (`DFBMaterialization.cpp`), a fixed double buffer. The pass does not
  size it from the loop; it assumes one carried value in flight per
  iteration, so two slots suffice and larger counts would only waste L1.
  Its `bind_cb` is emitted at function entry, where `finalize-dfb-indices`
  requires compiler-allocated binds to live.
- The general strategy emits exactly one consume and one produce of the
  state DFB per iteration, keeping `cb_reserve`/`cb_wait` accounting
  balanced.
- Correctness assumes the loop-carried state is consumed before it is
  reproduced within an iteration, so two slots suffice. The pass does not
  verify this; it holds for the recurrences the frontend emits.

### Performance

DST-resident accumulation is the preferred mechanism when the compiler
can keep the recurrence inside one acquire/release cycle. The partial
value never leaves the register file, and the final result is packed
once.

L1 packer accumulation is used when the selected strategy materializes
the recurrence as stores to an output DFB across loop iterations. The
packer adds in place in L1 and avoids the DFB-to-DST load that an
explicit add would perform every iteration. Enabling L1 accumulation by
default in the d2m backend produced a significant measured speedup:
https://github.com/tenstorrent/tt-mlir/pull/8387.

General DFB state materialization does not use a hardware accumulation
mechanism. It round-trips the state through L1 each iteration because a
non-additive recurrence cannot be expressed as DST or packer
accumulation.

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

Reduction loops are annotated with `ttl.reduction_loop`,
`ttl.l1_acc_initial`, and `ttl.l1_acc_scope_id`.
`TTKernelInsertL1Accumulation` consumes that metadata after conversion to
place packer L1 accumulation reconfiguration.

### Reconfiguration placement around L1 accumulation loops

`TTKernelInsertL1Accumulation` brackets each semantic scope group with
`pack_reconfig_l1_acc` calls. Scope groups are formed from
`ttl.l1_acc_scope_id`; the pass no longer infers semantic grouping from
shared pack dataflow buffers. The standard overwrite sequence disables
L1 accumulation before the group, conditionally enables it after the
first iteration's last pack so subsequent iterations accumulate, and
disables it again after the group:

```
pack_reconfig_l1_acc(0)
for iv = lb..ub:
    ...pack...
    if iv == lb: pack_reconfig_l1_acc(1)
pack_reconfig_l1_acc(0)
```

When `ttl.l1_acc_initial = accumulate_existing`, lowering has already
proved that L1 holds the initial value for the scope. The reconfiguration
before the group enables L1 accumulation, and the per-iteration
conditional enable on the root loop is omitted because every iteration
must accumulate from iteration 0 onward:

```
pack_tile(...)                  // prior pack runs with L1 acc disabled
pack_reconfig_l1_acc(1)
for iv = lb..ub:
    ...pack...
pack_reconfig_l1_acc(0)
```

The loop producer selects between the two sequences with
`ttl.l1_acc_initial`. `overwrite` disables L1 acc before the loop so
iteration 0 writes the baseline tile. `accumulate_existing` enables L1
acc before the loop so iteration 0 adds onto a value materialized by an
earlier store.

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
