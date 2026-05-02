# PipeNet Active-Set Lowering

This document describes how the tt-lang compiler decouples the launch
extent of an operation (the device grid that `@ttl.operation(grid=...)`
schedules onto) from the *work extent* described by the user's PipeNets.
Cores launched outside the work extent are guarded out at the IR level so
they do not execute kernel-thread bodies that were never meant to run on
them.

## Overview

`ttl.PipeNet` describes a logical communication pattern between cores. A
pipe carries data from a source coordinate (`src`) to either a single
destination (unicast) or a contiguous coordinate range (multicast). When
the launch grid is larger than the union of all pipe sources and
destinations, the extra cores have no role in the communication — but
without explicit guards they still execute every kernel-thread body the
operation defines, reading out-of-bounds tensor regions and corrupting
the multicast handshake (issue #541).

The compiler therefore computes the *active set* of an operation as the
union of every pipe's source cell and destination range across every
PipeNet, and wraps each `ttl.kernel_thread` function body in an
`scf.if (core_x, core_y) ∈ active_set` predicate. Inactive cores fall
through directly to the function terminator.

## Pass placement

```
... -> ttl-finalize-dfb-indices
    -> ttl-annotate-cb-associations
    -> ttl-insert-pipenet-active-guards          (NEW; module-scoped)
    -> convert-ttl-to-ttkernel
    -> ttkernel-insert-inits
    ...
```

`ttl-insert-pipenet-active-guards` runs after CB-association annotation
because CB metadata only flows through ops in the original (unwrapped)
function body. It runs before `convert-ttl-to-ttkernel` because that
pass:

* Consumes `ttl.create_pipe`, `ttl.if_src`, and `ttl.if_dst` (so the
  guard must already exist around them).
* Lowers `ttl.core_x` / `ttl.core_y` to TTKernel coordinate ops, which
  the guard predicate uses.

Three independent pipeline definitions need to stay in sync: the C++
`createTTLToTTKernelPipeline` in `lib/Dialect/TTL/Pipelines/TTLPipelines.cpp`,
the Python frontend pipeline string in `python/ttl/ttl_api.py`, and the
me2e builder in `test/me2e/builder/pipeline.py`. All three insert the
new pass at the same anchor.

## Active-set computation

The pass walks every `ttl.create_pipe` op in the module. Each pipe
contributes two half-open rectangles to a `SmallVector<ActiveRect>`:

1. A unit cell `[srcX, srcX+1) x [srcY, srcY+1)` for the source.
2. A range `[min(dstStartX, dstEndX), max(...)+1) x [min(dstStartY,
   dstEndY), max(...)+1)` for the destination.

The pass tolerates inverted destination bounds via `min`/`max` rather
than rejecting them — `CreatePipeOp`'s verifier currently does not
enforce ordering, so the pass normalizes defensively.

`ttl.create_pipe` source and destination coordinates are stored as
`I64Attr` on the op (verified in `TTLOps.td`), so the active set is a
purely static analysis — no SSA tracing or constant folding required.

### Module scoping

The pass walks the entire `ModuleOp`. This is correct because tt-lang's
Python frontend creates a fresh MLIR module per `@ttl.operation`
invocation (`_compile_kernel` in `ttl_api.py` calls `Module.create(loc)`
per kernel and resets `PipeNet._next_id`). If a future change ever
co-compiles multiple operations into one module, the pass would need to
scope the active set to the enclosing operation's thread group; the pass
description in `Passes.td` documents this as an invariant.

## Predicate construction

For each function with the `ttl.kernel_thread` attribute, the pass
inserts:

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
walk and skips the function. Subsequent canonicalization and CSE fold
duplicated index constants (one per rectangle) and constant predicates
when the active set covers the entire launch grid.

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

## Invariants the pass relies on

The pass asserts these on input IR; violations are caught at pass time
rather than miscompiling:

| Invariant | Rationale |
| --- | --- |
| Single-block kernel-thread function | Body movement is via block splice; multiple blocks would need region-level rewriting. The Python frontend never emits multi-block kernel-thread functions because user control flow lowers to `scf.if`/`scf.for`. |
| `func.return` terminator | The pass anchors the `scf.if` immediately before the terminator. Any other terminator type indicates a structural change that warrants pass-level review. |
| Static `I64Attr` pipe coordinates | The active set is computed at pass time. `CreatePipeOp` already enforces this in its verifier. |
| One operation per module | The pass walks all pipes in the module to compute one active set. Multiple operations in one module would require per-operation scoping (currently unused; see "Future work" below). |

## Skipping behavior

* No pipes in the module: the pass returns early without modifying any
  function. Operations that don't use PipeNet pay no cost.
* Empty function body: a function whose only operation is the terminator
  is left untouched, since there is nothing to guard.
* Functions without `ttl.kernel_thread` (host helpers, utility functions
  emitted alongside the kernel) are skipped — only thread functions need
  core-coordinate predicates.

## Simulator parity

The simulator does not run the MLIR pass, so it mirrors the same
behavior at scheduling time. PipeNet construction registers the
PipeNet on `SimulatorContext.kernel_pipe_nets` (per-greenlet, cleared
per operation). `Program._run_cooperative` reads the registry, expands
each pipe's source coordinate and destination range into a set of
linear core indices, and skips both thread registration and
`operation_start`/`operation_end` trace events for cores outside the
set.

The sim active-set computation lives in
`python/sim/pipe._compute_active_linear_cores`, with grid passed
explicitly so the function is independent of the caller's frame
locals. The compiler and sim both treat coordinates the same way:
row-major linearization (`coord[0] * grid[1] + coord[1]` in 2D),
consistent with `flatten_core_index`.

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
`[0, 3) x [0, 4)` — twelve cores. The remaining 8x7 - 12 = 44 launched
cores have predicate `false` and skip every thread body.

`canonicalizer` and `cse` after `convert-ttl-to-ttkernel` collapse the
redundant constant ops in the predicate; the final emitted C++ is a
single multi-rectangle test wrapping each kernel function.

## Limitations

* Work larger than launch: the pass guards the launch grid against
  under-coverage, not over-coverage. Existing kernels that distribute
  more work than launched cores via per-node block tiling (e.g.
  `_even_split` in `test_mcast_matmul.py`) are unaffected, because every
  launched core is in the active set and the predicate folds to true.
* Typos in pipe coordinates change the active set. A kernel that writes
  `dst=(slice(0, 5), 0)` instead of `dst=(slice(0, 4), 0)` has a
  one-core larger active set, and that extra core will execute the body
  even if the user did not intend it. The active set is exactly what
  the PipeNet says, no more.
* Inverted destination ranges are normalized, not rejected. The pass
  takes `min` / `max` of `dstStart` and `dstEnd`. Adding ordering to the
  `CreatePipeOp` verifier would let the pass assume `start <= end` and
  drop the defensive normalization.
* Three pipeline definitions: the new pass is wired into three separate
  strings (C++ pipeline, Python frontend, me2e builder). A future
  refactor consolidating these would prevent future passes from
  drifting between them.

## Future work

* If multiple operations are ever co-compiled into one module, scope the
  active-set walk to the enclosing operation by a marker attribute or by
  using a per-operation pass driver.
* Consider a `CreatePipeOp` verifier addition for `dstStart <= dstEnd`
  and for coordinates within a known device extent, eliminating the
  defensive normalization in this pass.
* Future spec extensions (e.g. n-D grids beyond 2D) will require
  generalizing the predicate construction; the current implementation
  hard-codes `core_x` / `core_y`.
