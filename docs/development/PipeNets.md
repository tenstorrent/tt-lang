# PipeNets

This document describes how PipeNets are owned, validated, lowered, and
scheduled in tt-lang. Both the compiler and the simulator consume the
same operation-level PipeNet collection; this doc covers the data flow, the
active-set guard pass that decouples launch extent from work extent,
how the simulator reproduces the compiler's active-set behavior at
scheduling time without running the MLIR pass, and the test coverage.

The launch grid (the device grid that `@ttl.operation(grid=...)`
schedules onto) is decoupled from the work extent described by the
user's PipeNets. Nodes launched outside the work extent are guarded out
at the IR level so they do not execute kernel function bodies that were
never meant to run on them.

## Overview

`ttl.PipeNet` describes a logical communication pattern between nodes. A
pipe carries data from a source coordinate (`src`) to either a single
destination (unicast) or a contiguous coordinate range (multicast). When
the launch grid is larger than the union of all pipe sources and
destinations, the extra nodes have no role in the communication, but
without explicit guards they still execute every kernel function body
the operation defines, potentially reading out-of-bounds tensor regions and
corrupting the multicast handshake (issue #541).

The compiler therefore computes the *active set* of an operation as the
union of every pipe's source cell and destination range across every
PipeNet, and wraps the body of every kernel function (any `func.func`
bearing the `ttl.kernel_thread` attribute, which marks compute and
data-movement kernels) in an `scf.if` predicate over the node
coordinates emitted by `ttl.core_x` / `ttl.core_y`. Inactive nodes fall
through directly to the function terminator.

## Operation pipenets

`OperationPipeNets` (defined in `python/_pipenets/__init__.py`)
is the per-operation data structure the compiler and the simulator
both consume. It holds:

- A list of `PipeNetUse` entries, each with an operation-local id
  (`0..N-1`, reset per invocation) and a tuple of `PipeUse` records
  (source `NodeCoord`, destination `NodeCoord` for unicast or
  `NodeRange` for multicast).
- `validate()`: empty PipeNet, overlapping multicast destinations,
  mixed unicast/multicast within one PipeNet.
- `active_node_set(grid)`: linearized union of every pipe's source
  and destination coordinates.

The compiler and the simulator both discover PipeNets by walking the
closure cells of the operation function and each registered thread's
wrapped function: body-local PipeNets show up through thread closures,
captured ones through the operation function's closure (spec L647).

Operation-local ids keep `ttl.create_pipe` ids stable across
invocations and keep TTKernel semaphore allocation
(`pipeNetId * 2` / `pipeNetId * 2 + 1`) deterministic. The
`OperationPipeNets` instance is built and validated before MLIR
emission on the compiler side and before `Program(...)` runs on the
simulator side. `PipeNet.__init__` also builds a one-PipeNet
`OperationPipeNets` and runs the same `validate()` synchronously, so
malformed PipeNets error at the construction source line.

## Pass placement

```
... -> ttl-finalize-dfb-indices
    -> ttl-annotate-cb-associations
    -> ttl-insert-pipenet-active-guards          (module-scoped)
    -> convert-ttl-to-ttkernel
    -> ttkernel-insert-inits
    ...
```

`ttl-insert-pipenet-active-guards` runs after DFB-index metadata
annotation (`ttl-annotate-cb-associations`) because the pipeline records
that metadata before the module-level guard pass moves the original body
under an `scf.if`. It runs before `convert-ttl-to-ttkernel` because
that pass:

* Consumes `ttl.create_pipe`, `ttl.if_src`, and `ttl.if_dst` (so the
  active-set guard must already exist around them).
* Lowers `ttl.core_x` / `ttl.core_y` to TTKernel coordinate ops, which
  the guard predicate uses.

Three independent pipeline definitions need to stay in sync: the C++
`createTTLToTTKernelPipeline` in `lib/Dialect/TTL/Pipelines/TTLPipelines.cpp`,
the Python frontend pipeline string in `python/ttl/ttl_api.py`, and the
me2e builder in `test/me2e/builder/pipeline.py`. All three insert the
new pass at the same anchor.

## Active-set computation

The pass walks every `ttl.create_pipe` op in the module. Each pipe
contributes two axis-aligned half-open boxes to a
`SmallVector<ActiveRect>`. `ActiveRect` is `{SmallVector<int64_t> lo,
SmallVector<int64_t> hi}`, one entry per dimension, so the
representation is rank-agnostic (in today's 2D dialect lo and hi each
have two entries; in 3D they would each have three). For 2D, each pipe
contributes:

1. A unit cell `[srcX, srcX+1) x [srcY, srcY+1)` for the source.
2. A range `[dstStartX, dstEndX+1) x [dstStartY, dstEndY+1)` for the
   destination.

`ttl.create_pipe` source and destination coordinates are declared as
`I64Attr` in `TTLOps.td`; the verifier checks that the attributes match
the result `PipeType`, are non-negative, and have `dstStart <= dstEnd`
on each axis. The active set is therefore static: the pass does not
trace SSA values or fold constants.

### Module scoping

The pass walks the entire `ModuleOp`. This is correct because tt-lang's
Python frontend creates a fresh MLIR module per `@ttl.operation`
invocation (`_compile_kernel` in `ttl_api.py` calls `Module.create(loc)`
per kernel and builds a fresh `OperationPipeNets` whose ids reset to
0..N-1). If a future change ever co-compiles multiple operations into
one module, the pass would need to scope the active set to the
enclosing operation's kernel functions; the pass description in
`Passes.td` documents this as an invariant.

## Predicate construction

For each kernel function (any `func.func` carrying the
`ttl.kernel_thread` attribute), the pass inserts:

```mlir
%x = ttl.core_x : index
%y = ttl.core_y : index
// For each rectangle r:
%xLo = arith.constant ... : index
%xHi = arith.constant ... : index
%yLo = arith.constant ... : index
%yHi = arith.constant ... : index
%xGe = arith.cmpi sge, %x, %xLo : index
%xLt = arith.cmpi slt, %x, %xHi : index
%yGe = arith.cmpi sge, %y, %yLo : index
%yLt = arith.cmpi slt, %y, %yHi : index
%xIn = arith.andi %xGe, %xLt : i1
%yIn = arith.andi %yGe, %yLt : i1
%inRect = arith.andi %xIn, %yIn : i1
// Combine across rectangles via arith.ori.
%active = arith.ori %prev, %inRect : i1
scf.if %active {
  ... original body ...
} {ttl.pipenet_active_guard}
return
```

The marker attribute `ttl.pipenet_active_guard` makes the pass
idempotent: a second run finds the existing guard via a function-scope
walk and skips the function. Subsequent canonicalization and CSE can
fold duplicated index constants from the rectangle predicates.

## Body movement

The pass moves the original body operations into the `then` region in
their original order rather than cloning them, preserving SSA values
without rewriting any uses:

1. Insert `ttl.core_x`, `ttl.core_y`, predicate ops, and `scf.if`
   (with auto-generated `scf.yield`) immediately before the function
   terminator.
2. Identify the first newly inserted op (the `core_x` definer).
3. Splice every operation from the start of the block up to (but not
   including) that anchor into the `then` block before the inserted
   `scf.yield`.

The function's `func.return` terminator stays at the end of the function
block, after the inserted `scf.if`.

## Invariants

The pass relies on these input properties. Structural violations are
reported at pass time; frontend and module-scope assumptions require
pipeline review if they change.

| Invariant | Rationale |
| --- | --- |
| Single-block kernel function | Body movement is via block splice; multiple blocks would need region-level rewriting. The Python frontend never emits multi-block kernel functions because user control flow lowers to `scf.if`/`scf.for`. |
| `func.return` terminator | The pass anchors the `scf.if` immediately before the terminator. Any other terminator type indicates a structural change that warrants pass-level review. |
| Static `I64Attr` pipe coordinates | The active set is computed at pass time. `TTLOps.td` declares the coordinates as attributes, and `CreatePipeOp` verifies consistency with the result pipe type. |
| One operation per module | The pass walks all pipes in the module to compute one active set. Multiple operations in one module would require per-operation scoping. |

## Skipping behavior

* No pipes in the module: the pass returns early without modifying any
  function. Operations that don't use PipeNet pay no cost.
* Empty function body: a function whose only operation is the terminator
  is left untouched, since there is nothing to guard.
* Functions without `ttl.kernel_thread` (host helpers, utility functions
  emitted alongside the kernel) are skipped; only kernel functions need
  node-coordinate predicates.

## Simulator parity

The simulator does not run the MLIR pass, so it mirrors the same
behavior at scheduling time. `operation()` builds the
`OperationPipeNets` from the operation's closure plus each thread's
wrapped function, validates it, and passes it to `Program(...)`.
`Program._run_cooperative` calls `pipenets.active_node_set(grid)`,
expands each pipe's source coordinate and destination range into a set
of linear node indices, and skips both kernel registration with the
greenlet scheduler and `operation_start`/`operation_end` trace events
for nodes outside the set. `active_node_set` returns `None` when the
collection has no PipeNets, which makes every launched node active.

The compiler and simulator both treat coordinates the same way:
row-major linearization (`coord[0] * grid[1] + coord[1]` in 2D),
consistent with `flatten_core_index`. A 1D coord on a 2D grid is
treated as an already-linear node index, mirroring the existing tt-lang
convention for kernels that schedule along a single dimension.

## Worked example

A small mcast matmul with work shape M_BLOCKS=4, N_BLOCKS=3 launched
under `grid="auto"` on a Wormhole device (8x7 grid):

```py
@ttl.operation(grid="auto")
def small_mcast_matmul(a, w, out):
    a_pipes = [
        ttl.Pipe(src=(0, row), dst=(slice(0, 3), row))   # broadcast A row
        for row in range(4)
    ]
    ttl.PipeNet(a_pipes)
    b_pipes = [
        ttl.Pipe(src=(col, 0), dst=(col, slice(0, 4)))   # broadcast B col
        for col in range(3)
    ]
    ttl.PipeNet(b_pipes)
    ...
```

Pipe sources contribute `{(0, 0), (0, 1), (0, 2), (0, 3), (0, 0), (1, 0),
(2, 0)}` and destinations contribute the rectangles `[0,3) x {row}` for
each row plus `{col} x [0,4)` for each col. The union covers exactly
`[0, 3) x [0, 4)`, twelve nodes. The remaining 8x7 - 12 = 44 launched
nodes have predicate `false` and skip every kernel function body.

`canonicalizer` and `cse` after `convert-ttl-to-ttkernel` can collapse
redundant constant ops in the predicate; the emitted C++ has one
active-set condition wrapping each kernel function.

## Test coverage

The same pytest file runs on hardware and on the simulator via
`test/scripts/ttlang-sim-pytest`, which patches `sys.modules` with the
simulator's `ttl` and `ttnn` before pytest collects, so hardware and
simulator coverage is the default for any test under `test/python/`.
Sim-only
tests under `test/sim/` are reserved for sim-internal helpers that have
no hardware analogue. Lit tests cover compile-time properties not
runtime-observable.

| #  | Behavior under test                                       | Dev | Sim | Lit |
|----|-----------------------------------------------------------|:---:|:---:|:---:|
|  1 | Empty PipeNet rejected at construction                    |  X  |  X  |     |
|  2 | Within-PipeNet mcast dst overlap rejected (full)          |  X  |  X  |     |
|  3 | Within-PipeNet mcast dst overlap rejected (partial)       |  X  |  X  |     |
|  4 | Unicast gather to same dst allowed                        |  X  |  X  |     |
|  5 | Nonoverlapping mcast pipes in one PipeNet allowed         |  X  |  X  |     |
|  6 | Pipe rejects open-bounded slices                          |  X  |  X  |     |
|  7 | Pipe rejects empty / inverted slices                      |  X  |  X  |     |
|  8 | Scatter on subgrid (work < launch, single mcast)          |  X  |  X  |     |
|  9 | Per-row scatter (multi-pipe disjoint dst, 2D active set)  |  X  |  X  |     |
| 10 | Cross-PipeNet destination overlap permitted               |  X  |  X  |     |
| 11 | Mixed unicast + multicast in one PipeNet                  |  X  |  X  |     |
| 12 | Loopback mcast (src in dst range)                         |  X  |  X  |     |
| 13 | Nested `if_src` / `if_dst` across two PipeNets (relay)    |  X  |  X  |     |
| 14 | 1D scatter (existing pattern)                             |  X  |  X  |     |
| 15 | 1D gather (existing pattern)                              |  X  |  X  |     |
| 16 | 1D gather, multiple tiles per source (existing)           |  X  |  X  |     |
| 17 | Ring forward (1D unicast +1, existing)                    |  X  |  X  |     |
| 18 | 2D broadcast (existing)                                   |  X  |  X  |     |
| 19 | Pipe chain / conv multi-stage (existing)                  |  X  |  X  |     |
| 20 | 1D mcast matmul auto-grid baseline (existing)             |  X  |  X  |     |
| 21 | Issue #541 regression: 4x3 work extent in auto launch     |  X  | (1) |     |
| 22 | Issue #541 regression: 2x2 work extent in auto launch     |  X  | (1) |     |
| 23 | 2D mcast matmul (work < launch via `_even_split`) [fixed] |  X  | (1) |     |
| 24 | Balanced 2D matmul (A on dm_read, B on dm_write) [fixed]  |  X  | (1) |     |
| 25 | Balanced 2D matmul + fused relu [fixed]                   |  X  | (1) |     |
| 26 | OperationPipeNets: src cell + dst range (mcast unit test)|     |  X  |     |
| 27 | OperationPipeNets: union across PipeNets                 |     |  X  |     |
| 28 | OperationPipeNets: unicast pipe single dst               |     |  X  |     |
| 29 | OperationPipeNets: None when empty                       |     |  X  |     |
| 30 | OperationPipeNets: validate empty PipeNet                |     |  X  |     |
| 31 | OperationPipeNets: validate overlapping mcast            |     |  X  |     |
| 32 | OperationPipeNets: operation-local id allocation         |     |  X  |     |
| 33 | sim pipe deadlock detection (existing)                    |     |  X  |     |
| 33a| Captured PipeNet works on hardware and sim (scatter)      |  X  |  X  |     |
| 34 | Frontend pipeline emits the active-set guard              |     |     |  X  |
| 35 | Pass collects rectangles + idempotent + skips no-pipe     |     |     |  X  |
| 36 | Pass coalesces source box contained in dst (loopback)     |     |     |  X  |
| 37 | Pass emits exact predicate constants for known src/dst    |     |     |  X  |
| 38 | Pass rejects multi-block kernel function (negative)       |     |     |  X  |
| 39 | Pass normalizes inverted destination ranges               |     |     |  X  |
| 40 | Guard survives `convert-ttl-to-ttkernel` (pipeline lit)   |     |     |  X  |

(1) Device-only due to a pre-existing simulator divergence: the
simulator's block-state machine accepts in-place `+=` only on a
*temporary* block (the result of a `fill` or a block expression), not
on a CB block that has already been written via `store(...)`. Hardware
accepts both. The matmul kernels in these tests use
`out_blk += a @ b` after an initial `out_blk.store(fill(...))`, which
the simulator rejects.

## Limitations

* Work larger than launch: the pass only disables launched nodes that are
  absent from all PipeNets. It does not add nodes or split work. Existing
  kernels that distribute more work than launched nodes via per-node
  block tiling (e.g. `_even_split` in `test_mcast_matmul.py`) are
  unaffected when every launched node appears in the active set.
* Typos in pipe coordinates change the active set. A kernel that writes
  `dst=(slice(0, 5), 0)` instead of `dst=(slice(0, 4), 0)` has a
  one-node larger active set, and that extra node will execute the body
  even if the user did not intend it. The active set is exactly what
  the PipeNet says, no more.
* Non-pipe work outside the PipeNet active set is also skipped. For an
  operation that uses any PipeNet, every node that must run a kernel
  function body must appear as a pipe source or destination.
* Inverted destination ranges are normalized, not rejected. The pass
  takes `min` / `max` of `dstStart` and `dstEnd`. Adding ordering to the
  `CreatePipeOp` verifier would let the pass assume `start <= end` and
  drop the defensive normalization.
* Three pipeline definitions: the new pass is wired into three separate
  strings (C++ pipeline, Python frontend, me2e builder). A future
  refactor consolidating these would prevent future passes from
  drifting between them.

## Future work

* Issue #505: lift the within-PipeNet multicast destination overlap
  restriction. Today a single PipeNet shares one semaphore pair across
  all its pipes, so a node receiving from two multicast sources cannot
  disambiguate the handshake. Per-source semaphore increments via
  `noc_semaphore_inc_multicast` in TTKernel would let one PipeNet
  describe true scatter-gather and all-to-all patterns. This is a
  TTKernel dialect + tt-metal change; it is unrelated to the active-set
  guard, but unblocking it would let `test_scatter_gather` and a
  single-PipeNet all-to-all version of `test_overlapping_pipenets` come
  off `@pytest.mark.skip`.
* Cross-chip (Galaxy / QuietBox / N300) PipeNets. tt-lang's
  `@ttl.operation` is a per-chip program by contract today; PipeNet
  coordinates are interpreted by the NoC, so they always refer to
  cores on a single chip. Users running on Galaxy already do so by
  composing per-chip operations and handling cross-chip data movement
  outside tt-lang (typically via ttnn CCL ops over the `tt_fabric`
  layer). There is no language construct for "this pipe crosses to
  chip (i, j)"; adding one is a language extension, not a free
  behavior change in the lowering. A future cross-chip PipeNet would
  introduce an explicit inter-chip pipe variant (e.g. carrying a
  `MeshCoordinate` for source and destination) that lowers to fabric
  ops alongside the existing intra-chip lowering. The
  `OperationPipeNets` data structure is small enough to grow that
  variant without affecting today's intra-chip path. Verifier
  bound-checking against the operation's grid extent (still future
  work) would also reject out-of-chip coordinates that today silently
  miscompile.
* If multiple operations are ever co-compiled into one module, scope the
  active-set walk to the enclosing operation by a marker attribute or by
  using a per-operation pass driver.
* Consider a `CreatePipeOp` verifier addition for `dstStart <= dstEnd`
  and for coordinates within a known device extent, eliminating the
  defensive normalization in this pass.
* Future spec extensions (e.g. n-D grids beyond 2D) need a dialect
  refactor, not a pass refactor. The pass already represents rectangles
  rank-agnostically (`ActiveRect.lo` / `ActiveRect.hi` are
  `SmallVector<int64_t>`) and `buildActivePredicate` loops over
  dimensions, so the rank is a property of how coordinates and pipe
  bounds are read, not of the predicate logic. Two helpers isolate the
  remaining 2D coupling: `readPipeSourceRect` /
  `readPipeDstRect` (pulling six named `I64Attr`s off `CreatePipeOp`)
  and `emitNodeCoords` (creating one `ttl.core_x` and one `ttl.core_y`
  op). When the dialect grows beyond 2D — e.g., `CreatePipeOp` adopts
  `DenseI64ArrayAttr` bounds and a single n-D node-coordinate op
  replaces `core_x` / `core_y` — only those helpers change. Affine
  maps are not the right tool here: the active set is a static union
  of axis-aligned rectangles, so per-dimension bounds are simpler than
  parameterized iteration domains and equally extensible.
