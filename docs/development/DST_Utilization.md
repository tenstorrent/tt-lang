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
for i in 0..N: pack DST[i] → CB            // drain DST
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
| 8 | FPU-aware DST pressure in `unroll_factor` | Not started | `TTLAssignDST.cpp` |
| 9 | Integrated unrolling in lower-to-loops | Not started | `ConvertTTLComputeToSCF.cpp` |
| 10 | Subblock-level synchronization insertion | Not started | — |
| 11 | Operation grouping (by-kind scheduling) | Not started | Design: `DST_Allocation.md` Phase 0 |
| 12 | Init consolidation | Not started | — |
| 13 | DST spilling (CB-based) | Not started | Design: `CB_Spilling.md` |

Components 1-7 are implemented on the `bnorris/max-dst` branch.
Components 8-13 are required for the full optimization but have design
documents only. The remainder of this document describes each component
and the pipeline that connects them.

## Pipeline

Current pipeline (`bnorris/max-dst`):

```
convert-ttl-to-compute
set-compute-kernel-config
assign-dst                  ← computes unroll_factor [2]
subblock-compute-for-dst    ← partitions iteration space [4]
insert-tile-regs-sync       ← per-tile sync (unchanged)
lower-to-loops
annotate-cb-associations
convert-ttl-to-ttkernel
```

Target pipeline (full optimization):

```
convert-ttl-to-compute
set-compute-kernel-config
assign-dst                  ← trait-aware unroll_factor [2, 7, 8]
subblock-compute-for-dst    ← outer loop over subblocks [4]
lower-to-loops              ← unrolled emit for inner subblock [9]
schedule-operations         ← group by kind within unrolled body [11]
insert-tile-regs-sync       ← one sync cycle per subblock [10]
annotate-cb-associations
convert-ttl-to-ttkernel
```

The key differences from the current pipeline: (1) `lower-to-loops`
directly emits N unrolled copies for subblocked computes rather than
creating an scf.for loop, and (2) sync insertion runs after lowering so
that one acquire/release wraps the entire unrolled subblock body rather
than each individual tile.

## Component Details

### 1-2. DST Allocation and Subblock Size

`TTLAssignDST` performs interval-based linear scan allocation
(documented in `DST_Allocation.md`) and computes:

```
dstPerIteration = maxDstUsed + 1
unrollFactor = min(floor(capacity / dstPerIteration), totalTiles)
```

The `unroll_factor` is attached as `ttl.unroll_factor` on the ComputeOp.

Current limitation: `dstPerIteration` does not account for FPU-aware
execution. An FPU binary uses 0 DST input slots (operands come from
CBs), reducing per-iteration pressure. Without FPU awareness, binary ops
are treated as needing DST for both inputs, which underestimates the
achievable subblock size. See component 8.

### 3-4. TilingInterface and Subblocking

`ComputeOp` implements MLIR's TilingInterface with four methods:
`getLoopIteratorTypes`, `getIterationDomain`, `getTiledImplementation`,
`getResultTilePosition`.

`TTLSubblockComputeForDST` uses these methods to partition the iteration
space. For a ComputeOp with `unroll_factor < totalTiles`, it generates:

```mlir
scf.for %iv = 0 to %innerDim step %unrollFactor {
  %a_sub = tensor.extract_slice %a[0, %iv] [1, %unrollFactor] [1, 1]
  %init_sub = tensor.extract_slice %init[0, %iv] [1, %unrollFactor] [1, 1]
  ttl.compute ins(%a_sub) outs(%init_sub) { ... }
}
```

The outer loop is side-effect-only (no `iter_args`). Stores inside the
compute body (`ttl.tile_store`) reference an external reserve view that
covers the full output CB, so the tile_store writes remain valid
regardless of which subblock is executing.

When `unroll_factor >= totalTiles`, no outer loop is generated (the
compute already fits in one subblock).

Multi-dimensional iteration spaces are flattened to 1D before
partitioning. When outer dimensions contribute tiles (i.e., totalTiles >
innerDimSize), the pass inserts `tensor.collapse_shape` on all operands
to linearize the iteration space, creates a 1D `ttl.compute`, then
partitions normally. This requires all indexing maps to be identity
(broadcast maps are not yet supported for flattening).

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
The key insight is that the execution engine determines DST register
usage per operation:

| Category | Traits | Input source | DST inputs | DST outputs |
|----------|--------|-------------|-----------|------------|
| FPU binary | `CBInput` | CB (both) | 0 | 1 |
| `dest_reuse` | `DSTInputs` + `CBInput` + `InPlace` | 1 CB + 1 DST | 0 (reused) | 1 (overwrites) |
| Unary | `DSTInputs` + `InPlace` | DST | 0 (in-place) | 0 (overwrites) |
| SFPU binary | `DSTInputs` | DST (both) | 2 | 1 |
| Broadcast | `CBInput` | CB | 0 | 1 |
| Reduce | `CBInput` + `Accumulating` | CB (input + scaler) | 0 | 1 |
| Matmul | `CBInput` + `Accumulating` | CB (A + B) | 0 | 1 |

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

### 9. Integrated Unrolling in Lower-to-Loops

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

### 10. Subblock-Level Synchronization

Current `TTLInsertTileRegsSync` wraps each `ttl.compute` body with
acquire/commit/wait/release (per-tile when placed before
lower-to-loops). The target is one sync cycle per subblock.

After `lower-to-loops` emits the unrolled body (component 9), the outer
loop body contains N tiles' worth of operations in sequence. Sync
insertion after lowering wraps the entire unrolled body:

```mlir
scf.for %iv = ... {   // outer loop over subblocks
  acquire
  // N unrolled copies: copy/compute → DST[0..N-1]
  commit; wait
  // N stores: pack DST[0..N-1]
  release
}
```

This requires sync insertion to run after `lower-to-loops` in the
pipeline. The sync pass must distinguish between the compute phase
(before commit) and the pack phase (after wait). This separation can be
achieved by:

- Detecting `tile_store` ops and placing them after the wait. The commit
  point is the boundary between the last compute op and the first store.
- Or, emitting compute ops and store ops in separate groups during
  lowering (component 9 emits all N compute copies, then all N stores),
  so sync insertion can place commit/wait at the group boundary.

### 11. Operation Grouping

Hand-written tt-metal kernels group operations by kind within the
unrolled body:

```
[init copy] [all copies] [init compute] [all computes] [commit/wait] [all packs] [release]
```

After lowering, the compiler produces by-iteration ordering:

```
copy[0] compute[0] store[0] copy[1] compute[1] store[1] ...
```

A scheduling pass must reorder to by-kind ordering while respecting
data dependencies. The algorithm is described in `DST_Allocation.md`
Phase 0, adapted from LARS (Rawat et al., SC'18). Operations within the
same group are independent (e.g., `exp_tile(0)` and `exp_tile(1)` have
no data dependency), so reordering within a kind is free.

Grouping provides:
- Init consolidation: one `*_init` per operation kind per subblock.
- Pipeline overlap: PACK thread can operate concurrently via DST
  double-buffering while MATH proceeds to the next subblock.
- Unpacker efficiency: avoids repeated CB switching within a group.

### 12. Init Consolidation

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

Init lowering is a conversion concern (`convert-ttl-to-ttkernel`), but
the grouping pass must provide the ordering guarantees that make init
consolidation valid.

### 13. DST Spilling

When per-iteration DST pressure exceeds capacity (long operation chains
with many live intermediates), the compiler must insert spill points
that pack intermediate values to L1 via temporary circular buffers and
reload them later. Design documented in `CB_Spilling.md`.

Spilling interacts with subblock size: spilling reduces per-iteration
pressure, potentially increasing the achievable `unroll_factor`. The
spill/reload overhead must be weighed against the synchronization
savings from larger subblocks.

## Current State vs Target

**What works today** (components 1-7, `bnorris/max-dst`):

The pipeline computes the correct subblock size, partitions the
iteration space via TilingInterface, and produces structurally correct
IR. The inner `ttl.compute` operates on a sub-tensor of
`unroll_factor` tiles. However, synchronization is still per-tile
(inserted before lower-to-loops), so DST utilization is not yet
improved over the baseline. The subblocking pass establishes the
structural foundation that components 9-10 build on. The allocator
uses trait queries (`isInPlaceOp`, etc.) instead of type-specific
checks, so new operations only need the correct trait annotations.

**What is needed for actual DST maximization** (components 9-10):

`lower-to-loops` must emit unrolled bodies with incrementing DST indices
(9), and sync must wrap the entire unrolled body (10). These two
components are the critical path. With just these two additions, the
compiler produces one sync cycle per subblock, proportionally reducing
synchronization overhead.

**What improves code quality further** (components 8, 11-12):

FPU-aware execution (8) increases the subblock size for
FPU-eligible binary operations (0 DST input slots instead of 2).
Operation grouping (11) and init consolidation (12) match the patterns
found in hand-written tt-metal kernels, reducing init overhead and
enabling pipeline overlap. These are important for performance but not
on the critical path for correctness of the subblock sync pattern.

**What handles edge cases** (component 13):

DST spilling is needed only when per-iteration pressure exceeds
capacity. Most elementwise operations have low pressure (1-2 DST
registers per iteration). Spilling becomes relevant for long fused
chains or reduction trees with many live intermediates.

## Pipeline Ordering Constraints

1. `assign-dst` must run before `subblock-compute-for-dst` because
   the subblocking pass reads `ttl.unroll_factor`.
2. `subblock-compute-for-dst` must run before `lower-to-loops` because
   it operates on `ttl.compute` ops (not scf.for loops).
3. `lower-to-loops` performs both lowering and unrolling (component 9):
   it reads `ttl.fully_unroll` and emits unrolled bodies directly.
4. `lower-to-loops` must run before sync insertion so that sync wraps
   the full unrolled body.
5. Sync insertion must run before `convert-ttl-to-ttkernel` because
   the conversion pass expects sync ops to be present.
6. Operation grouping (if implemented) runs after `lower-to-loops` and
   before sync insertion, so that groups are established before sync
   boundaries are placed.

## Related Documents

- `DST_Allocation.md`: DST register allocation algorithm (phases 0-4),
  worked examples, operation category traits.
- `MaximizingDSTUtilization.md`: FPU-aware unrolling design, execution
  engine summary, per-case analysis (SFPU unary, SFPU binary, FPU+SFPU
  chains, dest_reuse), target C++ code generation.
- `LoopOptimizations.md`: Full loop optimization pipeline design,
  hardware background, synchronization model, spilling, compilation
  modes.
- `CB_Spilling.md`: DST spilling via temporary circular buffers.
