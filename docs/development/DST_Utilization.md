# Maximizing DST Register Utilization

## Goal

Process the maximum number of tiles per DST synchronization cycle. A
synchronization cycle is the region between `tile_regs_acquire` and
`tile_regs_release`. Per-tile synchronization (one acquire/release per
tile) is correct but wastes DST capacity. Maximizing utilization means
filling all available DST registers before committing, then packing all
results before releasing. This reduces synchronization overhead
proportionally to the number of tiles processed per cycle.

The target output for N tiles in one cycle is:

```
acquire
for i in 0..N: unpack + compute → DST[i]   // fill DST
commit; wait
pack_tile_block(DST[0..N-1] → CB)           // drain DST
release
```

N is the **subblock size**, determined by DST capacity and per-iteration
register pressure.

## Components

The following table lists the components required to achieve maximized
DST utilization, their current implementation status, and the pass or
file responsible.

| # | Component | Status | Location |
|---|-----------|--------|----------|
| 1 | DST register allocation | Done | `TTLAssignDST.cpp` |
| 2 | Subblock size computation (`unroll_factor`) | Done | `TTLAssignDST.cpp` (lines 866-904) |
| 3 | TilingInterface on ComputeOp | Done | `TTLOps.cpp`, `TTLOps.td` |
| 4 | Subblock partitioning pass | Done | `TTLSubblockComputeForDST.cpp` |
| 5 | `extract_slice` tracing in `getAttachedCB()` | Done | `TTLOpsUtils.h` |
| 6 | `extract_slice` cleanup in final lowering | Done | `ConvertTTLToTTKernel.cpp` |
| 7 | Operation category traits | Done | `TTLBase.td`, `TTL.h`, `TTLOps.td`, `TTLOpsUtils.h` |
| 8 | FPU-aware DST pressure in `unroll_factor` | Done | `TTLAssignDST.cpp` |
| 9 | Pipeline option to gate DST maximization | Done | `TTLPipelines.h`, `TTLPipelines.cpp`, `ttl_api.py` |
| 10 | Integrated unrolling in lower-to-loops | Done | `ConvertTTLComputeToSCF.cpp` |
| 11 | Subblock-level synchronization insertion | Done | `TTLInsertTileRegsSync.cpp` |
| 12 | Operation grouping (by-kind scheduling) | Done | `TTLScheduleOperations.cpp` |
| 13 | Init insertion | Done | `TTKernelInsertInits.cpp` |
| 14 | DST spilling (CB-based) | Not started | — |

Components 1-13 are implemented on the `bnorris/max-dst` branch.
Component 14 (DST spilling) is not yet implemented. The remainder of
this document describes each component and the pipeline that connects
them.

After implementing each component, all tests must pass.

## Pipeline

Pipeline (`bnorris/max-dst`):

```
convert-ttl-to-compute
set-compute-kernel-config
assign-dst                  ← DST allocation + unroll_factor [1, 2, 7, 8]
subblock-compute-for-dst    ← outer loop over subblocks [3, 4]
insert-tile-regs-sync       ← one sync region per subblock [11]
lower-to-loops              ← unrolled emit for inner subblock [10]
schedule-operations         ← group by kind, place commit/wait [12]
annotate-cb-associations
convert-ttl-to-ttkernel     ← [5, 6]
ttkernel-insert-inits       ← one init per consecutive group [13]
canonicalize, cse
```

`lower-to-loops` directly emits N unrolled copies for subblocked
computes rather than creating an scf.for loop. Sync insertion stays
before lowering — after subblocking, each inner `ttl.compute` fits in
DST by construction, so one acquire/release per compute is correct.
Operation grouping (12) then reorders the lowered tile ops within the
sync region (loop fission into compute-phase + pack-phase) and places
commit/wait at the boundary. Init consolidation (13) is a separate pass
after conversion to TTKernel. These concerns are orthogonal to sync
placement.

## Component Details

### 1-2. DST Allocation and Subblock Size

`TTLAssignDST` performs interval-based linear scan allocation
(documented in `DST_Allocation.md`) and computes:

```
dstPerIteration = maxDstUsed + 1
unrollFactor = min(floor(capacity / dstPerIteration), totalTiles)
```

The `unroll_factor` is attached as `ttl.unroll_factor` on the ComputeOp.

FPU-aware execution (component 8) reduces `dstPerIteration` for
FPU-eligible binary ops. An FPU binary uses 0 DST input slots (operands
come from CBs), so `tile_add %a, %b` where both are block args costs 1
DST slot (output only) instead of 3 (2 copies + output). This is
detected in Phase 0 of `TTLAssignDST` and marked with the
`ttl.fpu_binary` attribute. Phase 0 is gated by the
`enable-fpu-binary-ops` option (default true); when disabled, all binary
ops use the SFPU path with `copy_tile`.

**FPU binary DST register reuse prevention**: When multiple FPU binary
ops appear in the same compute body, their output DST registers must be
distinct. FPU binary ops (`add_tiles`, `sub_tiles`, `mul_tiles`)
accumulate into their output DST register — the result includes the
previous DST contents. If two FPU binary ops share a DST index, the
second reads the first's residual and produces a corrupted result.
tt-mlir's D2M dialect prevents this via `getDstRegInPlace() = false` for
binary tile-tile ops (see `D2MOpsInterfaces.td` and
`InsertDstRegisterAccess.cpp`). `TTLAssignDST` achieves the same effect
by extending FPU binary result intervals in Phase 2 so the linear scan
allocator assigns distinct registers. The interval end is set to
`lastFPUStart + 1` (strictly greater than the last FPU binary op's start
index) because the expiry condition uses `<=`.

This is tested in `dst_fpu_binary.mlir` Test 6
(`@fpu_binary_no_dst_reuse`), which verifies that the pattern
`abs(a) + (a+b) + (a*b)` assigns different DST indices to the FPU
`tile_add` and `tile_mul`.

> **TODO**: The proper long-term fix is to pass `acc_to_dest=false`
> explicitly to FPU init functions in tt-mlir's TTKernel dialect
> (`AddTilesInitOp`, `SubTilesInitOp`, `MulTilesInitOp` — currently
> have a `FIXME` in `TTKernelOps.td` where the parameter is commented
> out). With explicit overwrite mode, DST reuse between FPU binary ops
> would be safe and the interval extension could be removed, improving
> DST utilization.

### 3-4. TilingInterface and Subblocking

`ComputeOp` implements MLIR's TilingInterface with four methods:
`getLoopIteratorTypes`, `getIterationDomain`, `getTiledImplementation`,
`getResultTilePosition`.

`TTLSubblockComputeForDST` uses these methods to partition the iteration
space into DST-sized subblocks via **multi-dimensional rectangular
tiling**. For each dimension, the subblock size is chosen as a divisor of
that dimension's size, maximizing the total subblock size (product of
per-dim subblock sizes) while staying within `unroll_factor`.

#### Subblock size algorithm

Given dimension sizes `[d0, d1, ...]` and `unrollFactor`, find subblock
sizes `[s0, s1, ...]` where each `si` divides `di`, `product(si) <=
unrollFactor`, and the product is maximized. Ties are broken by
preferring larger inner (higher-index) dimensions, which minimizes the
number of outer loop iterations.

The search enumerates divisor combinations per dimension with pruning
(`currentProduct > unrollFactor` cuts the branch). This is fast in
practice: dimension sizes are small (typically <= 32) and rank is low
(2-3), giving at most ~100 divisor combinations.

#### DST utilization examples

Block shape is the per-core tile grid, not the full tensor. A tensor
of `T_r x T_c` tiles distributed over a `G_r x G_c` core grid gives
each core a block of `(T_r/G_r) x (T_c/G_c)` tiles. In most realistic
cases tensors contain many blocks; larger tensors on more cores produce
smaller per-core blocks. Example mappings for a 32x32 tile tensor:

```
Tensor (tiles)   Core Grid   Block Shape   Blocks
--------------   ---------   -----------   ------
32x32            32x32       1x1            1024
32x32            16x16       2x2             256
32x32            8x8         4x4              64
32x32            4x4         8x8              16
```

The following table shows subblock decomposition for these block shapes.
DST capacity depends on data type and double-buffering: **8 tiles for
bf16** (16 physical / 2), **4 tiles for f32** (16 / 2 / 2).
`dstPerIter` is the number of DST registers consumed per tile iteration
(1 for unary/FPU-binary chains, 2+ for SFPU binary). `unrollFactor` is
the maximum number of tiles that fit in DST per sync region:
`min(floor(capacity / dstPerIter), totalTiles)`. The subblocking pass
finds the largest subblock with tile count ≤ `unrollFactor`, subject to
each dimension dividing evenly, so `tiles/subblock` may be less than
`unrollFactor` when divisor constraints prevent an exact fit.

```
Block   Type  Cap  dst/   unroll  Subblock                Tiles/    Sub-    DST
Shape         acity iter  Factor                          subblk   blocks   util
------  ----  ---  ----  ------  ----------------------  ------  ------  ------
1x1     bf16    8     1       1  [1,1] — no loop              1       1   12.5%
1x1     bf16    8     4       1  [1,1] — no loop              1       1     50%
1x1     bf16    8     8       1  [1,1] — no loop              1       1    100%
2x1     bf16    8     1       2  [2,1] — no loop              2       1     25%
2x1     bf16    8     2       2  [2,1] — no loop              2       1     50%
2x1     bf16    8     4       2  [2,1] — no loop              2       1    100%
4x1     bf16    8     1       4  [4,1] — no loop              4       1     50%
4x1     bf16    8     2       4  [4,1] — no loop              4       1    100%
1x4     bf16    8     1       4  [1,4] — no loop              4       1     50%
1x4     bf16    8     2       4  [1,4] — no loop              4       1    100%
1x8     bf16    8     1       8  [1,8] — no loop              8       1    100%
2x4     bf16    8     1       8  [2,4] — no loop              8       1    100%
2x8     bf16    8     1       8  [1,8] — loop d0              8       2    100%
4x4     bf16    8     1       8  [2,4] — loop d0 step 2       8       2    100%
3x4     bf16    8     1       8  [3,2] — loop d1 step 2       6       2     75%
3x4     bf16    8     2       4  [1,4] — loop d0,d1 step 2    4       3    100%
4x3     bf16    8     1       8  [2,3] — loop d0 step 2       6       2     75%
4x3     bf16    8     2       4  [4,1] — loop d1              4       3    100%
3x3     bf16    8     1       8  [1,3] — loop d0              3       3   37.5%
3x3     bf16    8     2       4  [1,3] — loop d0              3       3     75%
8x8     bf16    8     1       8  [1,8] — loop d0              8       8    100%
4x4     f32     4     1       4  [1,4] — loop d0              4       4    100%
2x8     f32     4     1       4  [1,4] — loop d0,d1 step 4    4       4    100%
```

Notes:
- DST utilization = `tiles/subblock * dstPerIter / capacity` — measures
  what fraction of DST registers are occupied per subblock cycle.
- For small blocks (1x1 through 1x4), higher `dstPerIter` from complex
  operation chains fills the remaining DST registers, reaching 100%
  utilization even with few tiles. E.g., 1x1 with dstPerIter=8 (a chain
  of 8 live intermediates) uses all 8 DST registers.
- "no loop" means `unrollFactor >= totalTiles`, so the entire compute
  fits in one DST cycle.
- 3x3 bf16: 9 tiles, no divisor combo of 9 reaches 8, best is [1,3]=3.
  Max achievable util is 75% (at dstPerIter=2).
- 3x4 and 4x3: both dimensions tiled (multi-dim subblocking). These
  shapes exercise the stride annotation logic where the tile loop stride
  differs from the loop upper bound.
- Hardware tests (`test_elementwise_shapes.py`) cover all shapes from
  1x1 through 4x4 (bf16).

#### Generated IR

For a 4x4 bf16 unary op (subblock=[2,4], loop on dim 0 step 2):

```mlir
scf.for %iv = 0 to 4 step 2 {
  %a_sub = tensor.extract_slice %a[%iv, 0] [2, 4] [1, 1]
  %out_sub = tensor.extract_slice %out[%iv, 0] [2, 4] [1, 1]
  ttl.compute ins(%a_sub) outs(%out_sub) { ... }
} {ttl.subblock_stride = 4 : index}
```

For a 2x8 f32 op (subblock=[1,4], loops on d0 step 1 and d1 step 4):

```mlir
scf.for %i = 0 to 2 step 1 {
  scf.for %j = 0 to 8 step 4 {
    %a_sub = tensor.extract_slice %a[%i, %j] [1, 4] [1, 1]
    %out_sub = tensor.extract_slice %out[%i, %j] [1, 4] [1, 1]
    ttl.compute ins(%a_sub) outs(%out_sub) { ... }
  } {ttl.subblock_stride = 1 : index}
} {ttl.subblock_stride = 8 : index}
```

The subblock loops do not carry loop-carried values (`iter_args`).
Each subblock writes its output tiles directly to the output CB at the
correct absolute position (computed from the subblock offset + the local
tile index within the subblock). Because the output CB is reserved once
for the entire block before the loop begins, each subblock iteration can
write to its portion independently.

When `unroll_factor >= totalTiles`, no outer loop is generated (the
compute already fits in one subblock).

#### Loop annotations

Three discardable attributes annotate compiler-generated loops and ops:

- **`ttl.subblock_stride`** (IndexAttr on `scf.for`): marks subblock
  loops. Value is the linearized stride for that dimension (product of
  all dimension sizes after it).
- **`ttl.tile_loop`** (IndexAttr on `scf.for`): marks tile iteration
  loops created by `lower-to-loops`. Value is the linearization stride
  from the full tensor shape (which may differ from the loop's upper
  bound when the compute has been subblocked).
- **`ttl.full_linearization_strides`** (DenseI64ArrayAttr on inner
  `ComputeOp`): set by the subblock pass on inner compute ops. Contains
  the row-major strides of the original (full) tensor shape.
  `lower-to-loops` reads this to annotate tile loops with the correct
  stride values.

`computeCBTileIndexFromLoops` (in `ConversionUtils.h`) uses these
attributes to compute correct absolute CB tile indices during
TTL-to-TTKernel conversion:
- Tile loops contribute `IV * stride` from the `ttl.tile_loop` attribute.
- Subblock loops contribute `IV * stride` from `ttl.subblock_stride`.
- Unmarked loops (user loops, external loops) are ignored.

#### Non-identity indexing maps

Multi-dim tiling supports non-identity indexing maps (broadcast,
reduction) because `getTiledImplementation` maps iteration domain
offsets/sizes to per-operand slices via the indexing maps. For a
broadcast map `(d0,d1)->(d0,0)`, an operand `tensor<Mx1>` with offsets
`[o0, o1]` and sizes `[s0, s1]` gets sliced at `[o0, 0]` with size
`[s0, 1]`. No flattening or map transformation is needed.

### 5-6. Extract Slice Support

Subblocking introduces `tensor.extract_slice` ops between `attach_cb`
and the inner `ttl.compute`. Two downstream utilities needed extension:

- `getAttachedCB()` in `TTLOpsUtils.h`: traces through
  `tensor::ExtractSliceOp` to find the source tensor's CB.
- `removeTensorDataflowOps()` in `ConvertTTLToTTKernel.cpp`: erases
  dead `tensor::ExtractSliceOp` during final cleanup.

### 7-8. Operation Category Traits and FPU-Aware DST Pressure

Each operation's execution category is determined by orthogonal traits
defined in TableGen (see `DST_Allocation.md`, Operation Category Traits).
The following table summarizes DST register usage of different operation
types.

| Category | Traits | Input source | DST inputs | DST outputs |
|----------|--------|-------------|-----------|------------|
| FPU binary | `CBInput` | CB (both) | 0 | 1 |
| `dest_reuse` | `DSTInputs` + `CBInput` + `InPlace` | 1 CB + 1 DST | 0 (reused) | 1 (overwrites) |
| Unary | `DSTInputs` + `InPlace` | DST | 0 (in-place) | 0 (overwrites) |
| SFPU binary | `DSTInputs` | DST (both) | 2 | 1 |
| Broadcast | `CBInput` | CB | 0 | 1 |
| Reduce | `CBInput` + `Accumulating` | CB (input + scaler) | 0 | 1 |
| Matmul | `CBInput` + `Accumulating` | CB (A + B) | 0 | 1 |
| Transpose (CB) | `CBInput` | CB | 0 | 1 |
| Transpose (DST) | `DSTInputs` + `InPlace` | DST | 0 (in-place) | 0 (overwrites) |

FPU binary ops consume 0 DST input slots. For `exp(a + b)`, the FPU add
reads from CBs and writes to DST; the SFPU exp operates in-place. The
per-iteration footprint is 1, not 3 (which the SFPU-only path would
require for `copy_tile(a)`, `copy_tile(b)`, `add_result`).

Five orthogonal traits classify operations. All are defined in
`TTLBase.td` with C++ implementations in `TTL.h`:

- **`TTLCBInputTileOpTrait`**: Input(s) read from CB, not DST.
- **`TTLDSTInputsTrait`**: At least one operand is consumed from DST.
- **`TTLInPlaceOpTrait`**: Result overwrites the DST input (shared slot).
- **`TTLAccumulatingOpTrait`**: Result accumulates across invocations.
- **`TTLCBOutputTileOpTrait`**: Op carries an explicit output CB operand;
  init configures the PACK thread. Affects init consolidation ordering.

No separate annotation pass is required. The allocator in
`TTLAssignDST` queries these traits compositionally:

1. `hasTrait<TTLCBInputTileOpTrait>()` identifies CB-only block
   arguments (those consumed only by CB-reading operations), which are
   excluded from DST allocation entirely.
2. `hasTrait<TTLInPlaceOpTrait>()` triggers interval merging (Phase 2
   in `DST_Allocation.md`), so in-place chains share a single DST slot.
3. The combination of trait queries determines `dstPerIteration`, which
   feeds the `unroll_factor` computation.

### 9. Pipeline Option

DST maximization is an optimization, not required for correctness. The
compiler must be able to produce valid code without it (per-tile
synchronization, no subblocking). The `maximize-dst` option in
`TTLToTTKernelPipelineOptions` controls whether `subblock-compute-for-dst`
runs (default is true). `assign-dst` always runs (DST index attributes are needed
regardless). See the [Pipeline Options](#pipeline-options) section for
details.

### 10. Integrated Unrolling in Lower-to-Loops

Each inner subblock `ttl.compute` (after subblocking) has exactly
`unroll_factor` tiles. Creating an scf.for loop only to immediately
unroll it is unnecessary. Instead, `lower-to-loops` directly emits N
copies of the loop body with incrementing DST indices when the compute
is marked for full unrolling.

The subblocking pass (component 4) attaches a `ttl.fully_unroll`
attribute to the inner `ttl.compute`. When `lower-to-loops` encounters
this attribute, it emits:

```
// For unroll_factor = 4:
body[0] with DST[0]
body[1] with DST[1]
body[2] with DST[2]
body[3] with DST[3]
```

Each copy is identical except for the tile index and DST register index.
The DST index for copy k is `k * dstPerIteration` (or simply `k` when
`dstPerIteration == 1`). No scf.for is created for the inner subblock;
only the outer subblock loop (from component 4) remains as a loop.

This approach was chosen over a separate unrolling pass (prototyped on
`bnorris/unroll-for-dst` using `loopUnrollByFactor`) because:
- The inner compute is always exactly `unroll_factor` tiles; creating a
  loop to immediately unroll it is roundabout.
- DST index assignment is straightforward during emission (incrementing
  counter) rather than requiring a post-hoc callback to patch indices.
- One fewer pass in the pipeline.

### 11. Subblock-Level Synchronization

Current `TTLInsertTileRegsSync` wraps each `ttl.compute` body with
acquire/commit/wait/release (per-tile when placed before
lower-to-loops). The target is one sync region per subblock.

After subblocking, each inner `ttl.compute` fits in DST by construction.
Sync insertion wraps the entire inner compute with one acquire/release
region — the same placement as today, but now one region covers all tiles
in the subblock rather than just one tile:

```mlir
scf.for %iv = ... {   // outer loop over subblocks
  acquire
  ttl.compute ins(%sub) outs(%sub_out) { ... }
  commit; wait
  release
}
```

After `lower-to-loops` expands the compute (component 10), the tile
ops appear inside the sync region. Operation grouping (component 12)
then reorders into compute-phase + pack-phase and places commit/wait at
the boundary:

```mlir
scf.for %iv = ... {   // outer loop over subblocks
  acquire
  // N copies: copy/compute → DST[0..N-1]
  commit; wait
  // N packs: DST[0..N-1] → CB
  release
}
```

Init ops (common inits like `binary_op_init_common` / `init_sfpu`, and
per-op inits like `exp_tile_init`) are not emitted by the sync pass.
They are inserted later by the consolidation pass (component 13) at the
TTKernel level, after conversion. This keeps sync insertion focused on
DST lifecycle management.

Sync insertion does not need to run after lowering. It operates on
`ttl.compute` ops (before lowering), and the commit/wait placement
within the sync region is handled by the grouping pass (component 12).

### 12. Operation Grouping

Hand-written tt-metal kernels group operations by kind within the
unrolled body:

```
[init copy] [all copies] [init compute] [all computes] [commit/wait] [all packs] [release]
```

After lowering, the compiler produces by-iteration ordering:

```
copy[0] compute[0] store[0] copy[1] compute[1] store[1] ...
```

`TTLScheduleOperations` reorders to by-kind ordering while respecting
data dependencies. The algorithm assigns each op a depth (longest path
from any root in the dependency graph), then sorts by (depth, category).
Within the same depth and category, ops are independent (e.g.,
`exp_tile(0)` and `exp_tile(1)` have no data dependency), so reordering
within a kind is free.

Grouping provides:
- Init consolidation: one `*_init` per operation kind per subblock.
- Pipeline overlap: PACK thread can operate concurrently via DST
  double-buffering while MATH proceeds to the next subblock.
- Unpacker efficiency: avoids repeated CB switching within a group.

### 13. Init Consolidation

Each tt-metal operation kind requires an init call before first use
(`exp_tile_init`, `add_tiles_init`, `copy_tile_to_dst_init_short`).
Without grouping, init is called before every operation. With grouping,
one init per kind per subblock suffices.

When switching between kinds within a subblock, `*_init_short`
reconfigures UNPACK + MATH only (not PACK), and
`*_init_short_with_dt` additionally reconfigures data formats.

Full-init operations (broadcast `unary_bcast_init`, reduce
`reduce_init`) configure UNPACK + MATH + PACK and must appear before
any short-init operations to avoid clobbering PACK configuration.

`TTKernelInsertInits` is the single source of truth for all init
ops. The conversion pass does not emit init ops; this pass inserts
them in two phases:

- **Phase 1 (common inits)**: For each sync region
  (`tile_regs_acquire` ... `tile_regs_release`), inserts
  `binary_op_init_common(in0_cb, in1_cb, out_cb)` if FPU binary ops are
  present, or `init_sfpu(in_cb, out_cb)` otherwise. CB operands are
  derived from the ops in the region (CopyTile/FPU binary for inputs,
  PackTile for output). Common inits are hoisted above enclosing
  compiler-generated loops (`ttl.tile_loop`, `ttl.subblock_stride`) but
  not past unmarked loops.
- **Phase 2 (per-op inits)**: Maintains an init key `(TypeID, operands,
  discriminator)` for each compute op and inserts an init only when the
  key changes between consecutive ops. Tracking resets at sync
  boundaries.

The grouping pass (component 12) must provide the ordering guarantees
that make per-op init consolidation maximally effective.

### 14. DST Spilling

When per-iteration DST pressure exceeds capacity (long operation chains
with many live intermediates), the compiler must insert spill points
that pack intermediate values to L1 via temporary circular buffers and
reload them later.

Spilling interacts with subblock size: spilling reduces per-iteration
pressure, potentially increasing the achievable `unroll_factor`. The
spill/reload overhead must be weighed against the synchronization
savings from larger subblocks.

## Current State

**Implemented** (components 1-13, `bnorris/max-dst`):

The full DST maximization pipeline is operational. The pipeline computes
the correct subblock size (with FPU-aware DST pressure), partitions the
iteration space via TilingInterface, emits unrolled subblock bodies with
one sync region per subblock, groups operations by kind, and consolidates
init ops. FPU binary detection (component 8) marks
`tile_add`/`tile_sub`/`tile_mul` with both block-arg operands as
`ttl.fpu_binary`, reducing per-iteration DST pressure (0 input slots
instead of 2). The allocator prevents DST register reuse between FPU
binary ops (see component 1-2 details) to avoid accumulation-related
numerical corruption. The allocator uses trait queries (`isInPlaceOp`,
etc.) instead of type-specific checks, so new operations only need the
correct trait annotations.

**Not yet implemented** (component 14):

DST spilling is needed only when per-iteration pressure exceeds
capacity. Most elementwise operations have low pressure (1-2 DST
registers per iteration). Spilling becomes relevant for long fused
chains or reduction trees with many live intermediates.

## Pipeline Options

The DST maximization passes are an optimization, not required for
correctness. The compiler must be able to produce valid code without
them (per-tile synchronization, no subblocking). This requires a
pipeline option to gate the optimization passes.

### Options

`TTLToTTKernelPipelineOptions` in `TTLPipelines.h`:

| Option | Default | Description |
|--------|---------|-------------|
| `maximize-dst` | true | Enable subblock partitioning and operation scheduling |
| ~~`consolidate-inits`~~ | ~~true~~ | ~~Removed: init insertion is now unconditional~~ |
| `enable-fpu-binary-ops` | true | Use FPU execution for binary add/sub/mul when both operands are CB-backed |
| `lower-to-emitc` | false | Lower TTKernel to EmitC (for C++ translation) |

Python API equivalents (`CompilerOptions` in `ttl_api.py`):

| Python option | CLI flag | Pipeline option |
|---------------|----------|-----------------|
| `maximize_dst` | `--no-maximize-dst` | `maximize-dst=0` |
| ~~`consolidate_inits`~~ | ~~removed~~ | ~~removed~~ |
| `enable_fpu_binary_ops` | `--no-fpu-binary-ops` | `enable-fpu-binary-ops=0` |

Environment variable: `TTLANG_COMPILER_OPTIONS` (space-separated flags).

Example:
```bash
TTLANG_COMPILER_OPTIONS="--no-fpu-binary-ops --no-maximize-dst" python examples/script.py
```

### Pipeline Behavior

**With `--maximize-dst`** (default — optimized compilation):

The full pipeline as shown in the [Pipeline](#pipeline) section.

**Without `--maximize-dst`** (baseline compilation):

```
convert-ttl-to-compute
set-compute-kernel-config
assign-dst                  ← always runs (assigns dst_idx attributes)
insert-tile-regs-sync       ← per-tile sync (baseline behavior)
lower-to-loops
annotate-cb-associations
convert-ttl-to-ttkernel
ttkernel-insert-inits
```

No `subblock-compute-for-dst` or `schedule-operations`. Each tile gets
its own acquire/release cycle. `assign-dst` still runs because `dst_idx`
attributes are needed by all compute lowering.

**With `--no-fpu-binary-ops`**:

Phase 0 of `TTLAssignDST` is skipped. Binary add/sub/mul ops are not
marked with `ttl.fpu_binary` and use the SFPU path (copy_tile for both
operands, `add_binary_tile`/`sub_binary_tile`/`mul_binary_tile` instead
of `add_tiles`/`sub_tiles`/`mul_tiles`). The consolidation pass emits
`init_sfpu` instead of `binary_op_init_common`.

### Why This Matters

1. **Incremental development**: Each optimization component can be
   implemented and tested independently. The baseline path always works.
2. **Debugging**: When investigating miscompiles, disabling DST
   maximization isolates whether the bug is in the optimization or
   elsewhere.
3. **Correctness first**: The baseline path establishes a correct
   reference. The optimized path must produce equivalent results.
4. **Testing**: Lit tests can run both paths
   (`--ttl-to-ttkernel-pipeline{maximize-dst=false}` vs default) to
   verify the optimization preserves semantics.

The individual pass tests (e.g., `ttl-assign-dst`) are unaffected —
they invoke passes directly, not through the pipeline.

## Pipeline Ordering Constraints

1. `assign-dst` must run before `subblock-compute-for-dst` because
   the subblocking pass reads `ttl.unroll_factor`.
2. `subblock-compute-for-dst` must run before `lower-to-loops` because
   it operates on `ttl.compute` ops (not scf.for loops).
3. `insert-tile-regs-sync` must run before `lower-to-loops` because
   sync ops are inserted around `ttl.compute` bodies before they are
   expanded into loops.
4. `lower-to-loops` performs both lowering and unrolling (component 10):
   it reads `ttl.fully_unroll` and emits unrolled bodies directly.
5. `schedule-operations` runs after `lower-to-loops` because it
   reorders individual tile ops (not `ttl.compute` ops). It operates
   within the sync region established by `insert-tile-regs-sync`.
6. `ttkernel-insert-inits` runs after `convert-ttl-to-ttkernel`
   because it operates on TTKernel ops, not TTL ops. Scheduling
   (component 12) must have already grouped same-kind ops for
   consolidation to be maximally effective.
7. Sync insertion must run before `convert-ttl-to-ttkernel` because
   the conversion pass expects sync ops to be present.

## Related Documents

- `docs/development/DST_Allocation.md`: DST register allocation algorithm (phases 0-4),
  worked examples, operation category traits.
