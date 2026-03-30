# Accumulating Compute Lowering

## Problem

Several tile operations accumulate results in DST registers across
multiple invocations: `reduce_tile` sums/maxes across a reduction
dimension, `matmul_tiles` accumulates C += A * B across the K
dimension. The hardware requirement: DST must remain live (not
re-acquired) across the full accumulation scope.

## Approach

`TTLLowerToLoops` detects accumulating ComputeOps (body contains
`AccumulatingOpTrait`) and generates parallel-outer / reduction-inner
loops with `DstSectionOp` wrapping the reduction loop and stores.
DST registers persist across reduction iterations.

### DstSectionOp

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

### Loop structure for accumulating computes

`generateAccumulatingLoops` separates parallel and reduction dims
from `iterator_types`, generates parallel loops outer and reduction
loops inner:

```
for each parallel dim:           // output tile iteration
    dst_section {
        for each reduction dim:  // accumulate into DST
            <tile ops>
        <stores with placeholder tile + explicit dst_idx>
    }
```

Stores use a placeholder tile value (via `UnrealizedConversionCastOp`)
with an explicit `dst_idx` attribute, since the SSA tile value from
`reduce_tile` is loop-local. `TileStoreLowering` reads `dst_idx`
from the store attribute.

### Per-op init insertion

`TTKernelInsertInits` Phase 2 walks `TileRegsAcquireOp` to process
each sync region. For each acquire, it iterates the flat ops between
acquire and release in the same block. Each flat op may contain
compute ops in nested regions (e.g., `reduce_tile` inside a
reduction `scf.for`); these are discovered via `op.walk()`. The
init is inserted before the flat container op (e.g., before the
`scf.for`, not inside it), and hoisted above any compiler-generated
loops. Consecutive compute ops with the same init key share a single
init (forward-order dedup via `prevKey`).

Bcast, reduce, and transpose inits resolve their output CB from a
`ttl.*_output_cb_index` attribute propagated during TTL-to-TTKernel
conversion, rather than scanning for pack ops in the sync region.

### Detecting accumulating ComputeOps

A `ComputeOp` requires accumulation lowering when its body contains
any operation with `TTLAccumulatingOpTrait`. Detection is structural.
`SubblockComputeForDST` skips accumulating computes (asserted in
`LowerToLoops`).

### L1 accumulation fallback

`TTKernelInsertL1Accumulation` inserts `pack_reconfig_l1_acc(1)` from
the second iteration of reduction loops for `reduce_sum`. This is
skipped for `reduce_max` (L1 accumulation uses additive packing,
incorrect for max). With DST accumulation, L1 accumulation is
redundant for reduce but remains as a fallback for cases where DST
accumulation is not applicable.

## IR trace: 2x2 reduce_sum along dim 0 (DST accumulation)

Input: `tensor<2x2xtile>`, scaler: `tensor<1x1xtile>`, output: `tensor<1x2xtile>`.

### After LowerToLoops

```mlir
scf.for %j = %c0 to %c2 step %c1 {       // parallel (output cols)
    ttl.dst_section {
        scf.for %i = %c0 to %c2 step %c1 { // reduction (input rows)
            %in = tensor.extract %inp[%i, %j]
            %sc = tensor.extract %scaler[%c0, %c0]
            %out = tensor.extract %init[%c0, %j]
            ttl.tile_reduce %in, %sc, %out sum reduce_dim_col {dst_idx = 0}
        } {ttl.reduction_loop, ttl.tile_loop_stride = 2}
        ttl.tile_store %placeholder, %view[%c0, %j] {dst_idx = 0}
    }
} {ttl.tile_loop_stride = 1}
```

### After TTKernel conversion + insert-inits

```
init_sfpu(cb0, cb2)
reduce_init(cb0, cb1, cb2, SUM, REDUCE_COL)
for j = 0..2:                              // parallel
    tile_regs_acquire()
    for i = 0..2:                          // reduction (DST persists)
        reduce_tile(cb0, cb1, i*2+j, 0, 0, SUM, REDUCE_COL)
    reduce_uninit()
    tile_regs_commit() / tile_regs_wait()
    pack_tile(0, cb2, j)
    tile_regs_release()
cb_push_back(cb2, 2)
```
