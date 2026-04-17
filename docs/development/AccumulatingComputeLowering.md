# Accumulating Compute Lowering

## Problem

Several tile operations accumulate results in DST registers across
multiple invocations: `reduce_tile` sums/maxes across a reduction
dimension, `matmul_tiles` accumulates C += A * B across the K
dimension. The hardware requirement: DST must remain live (not
re-acquired) across the full accumulation scope.

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
the existing L1 value from the second iteration onward.

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

`precededByNonAccumulatingPack` detects the prior pack via a backward
walk over the L1-acc loop's parent block. A `pack_tile` into one of the
loop's pack CBs contributes a prior value, as does any pack inside a
non-annotated `scf.for` (the compiler-generated tile-loop wrappers
carrying `ttl.tile_loop_stride` produce this shape around a user
`.store(...)`). The walk stops at any op that resets or shadows the L1
slot: `cb_reserve_back` or `cb_push_back` on one of the pack CBs, an
annotated `scf.for` (`ttl.l1_acc_loop` or `ttl.reduction_loop`) that
packs to one of them (it has its own enable scope), and any other
region-bearing op (`scf.if`, `scf.while`, custom region ops) whose body
packs to one of them (the walk does not reason about their execution
semantics).

The walk requires the prior pack to sit in the L1-acc loop's parent
block because L1 acc enablement depends on deterministic execution
ordering immediately before the loop. A pack in an outer region
executes only once relative to multiple iterations of an enclosing
wrapper, so its value is not the most recent on iterations after the
first.

For multi-output loops, the walk returns true only when every CB in the
loop's pack-CB set is covered by some preceding non-accumulating pack.
L1 acc is a single switch for the whole sync region, so partial coverage
must fall back to the standard pattern; enabling before the group with
some CBs uncovered would corrupt their iteration 0 (acc onto stale L1).

Sibling loops in a group always emit an unconditional enable before the
loop and a per-iteration enable inside it, regardless of whether the
root has a prior pack. The per-iteration enable on a sibling is a
redundant no-op when the root's reconfig already enabled L1 acc.

The pass is idempotent: a prior run leaves a `pack_reconfig_l1_acc`
either inside the L1-acc loop body (standard pattern) or immediately
preceding the loop (prior-value pattern), and the second run detects
either signal and returns without re-emitting.

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
