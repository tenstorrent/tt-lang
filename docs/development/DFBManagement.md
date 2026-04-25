# Dataflow Buffer Management

This document describes how the tt-lang compiler manages dataflow buffers (DFBs) -- the L1-resident circular buffers that transfer data between compute and data movement threads on Tenstorrent hardware.

## Overview

DFBs originate from two sources. User-declared DFBs are created explicitly in the DSL via `make_dataflow_buffer_like` and correspond to the programmer's data movement plan. Compiler-allocated DFBs are inserted automatically at fusion split points where a tensor-level operation requires a CB-attached operand but receives the result of a fused expression chain.

The hardware supports at most 32 DFBs per node (indices 0--31). User and compiler-allocated DFBs share this index space. The compiler assigns indices sequentially during insertion (starting *after* the last user-declared DFB), then applies lifetime-based index reuse to reduce the physical DFB count.

## Pipeline

The DFB-related passes in `ttl-to-ttkernel-pipeline` execute in this order:

```
ttl-insert-intermediate-dfbs   (FuncOp)   Create compiler-allocated DFBs
ttl-insert-cb-sync             (FuncOp)   Insert cb_push / cb_pop
  ... compute lowering, DST assignment, loop lowering ...
ttl-finalize-dfb-indices       (Module)   Index reuse + limit check
ttl-annotate-cb-associations   (FuncOp)   Copy CB indices to tile ops
convert-ttl-to-ttkernel        (Module)   Lower to TTKernel dialect
ttkernel-insert-inits          (Module)   Insert hardware init calls
```

`ttl-finalize-dfb-indices` must precede `ttl-annotate-cb-associations` because annotation copies the `cb_index` attribute from `BindCBOp` onto tile operations (`bcast`, `reduce`, `transpose`). If annotation runs before finalization, the copied indices become stale after reuse rewrites them.

## DFB Lifecycle

A DFB has two lifecycle halves: the producer (write) side driven by `cb_reserve`/`cb_push`, and the consumer (read) side driven by `cb_wait`/`cb_pop`. For user-declared DFBs these halves span different threads: data movement writes to the CB, compute reads from it, and both threads reference the same CB index. For compiler-allocated intermediate DFBs both halves are in the same compute function.

```
|
v time
          Producer (write)          Consumer (read)
          ----------------          ---------------
bind_cb   cb_reserve                cb_wait              L1 buffer held
          store                     attach_cb              |
          cb_push ------ slot ----> ... consumer ops       |
          (slot returned) <-------- cb_pop               L1 buffer free
```

`cb_reserve` claims a buffer slot for the packer; `cb_push` releases that slot to the unpacker. `cb_wait` blocks until the slot is available; `cb_pop` releases it back to the packer. `bind_cb` allocates the L1 backing storage and is shared by both sides.

For compiler-allocated DFBs, `InsertIntermediateDFBs` creates the full sequence from `bind_cb` through `attach_cb`. `InsertCBSync` adds `cb_push` (after the producer's last use) and `cb_pop` (after the consumer's last use).

A DFB's L1 memory is reclaimable after its last `cb_pop`. This defines the interval used for index reuse.

## Intermediate DFB Insertion

`TTLInsertIntermediateDFBs` walks all operations implementing `DFBInputOpInterface` (reduce, bcast, matmul, transpose). For each operand that the interface marks as requiring a CB-attached value, the pass checks whether the operand traces to an existing CB via `getAttachedCB`. If not, the pass materializes the value through a fresh DFB: `bind_cb`, `cb_reserve`, `store`, `cb_wait`, `attach_cb`. The new DFB receives the `ttl.compiler_allocated` marker attribute.

When multiple `DFBInputOpInterface` operations consume the same non-CB-attached value, the materialization is shared -- only one DFB is created and the second consumer's operand is rewritten to the existing attached value.

Each DFB is created with `blockCount=2` (double-buffering) so the packer and unpacker can operate on different halves simultaneously within the same thread.

## Index Reuse

`TTLFinalizeDFBIndices` reduces the physical DFB count by assigning the same index to compiler-allocated DFBs whose lifetimes do not overlap. The algorithm runs per function. Compiler-allocated DFBs are intra-thread (both producer and consumer are in the same compute function), so their lifetimes are independent across functions.

Two DFBs may share an index only if they have identical `CircularBufferType` (shape, element type, block count). Since `CircularBufferType` is an MLIR uniqued type, this is a pointer comparison. The algorithm partitions DFBs by type and runs a linear scan within each partition.

### Algorithm

```
reuseDFBIndices(funcOp, compilerAllocatedBindCBOps):
  // Assign sequential indices to function-level operations.
  for op in funcOp.entryBlock:
    opIndex[op] = nextIdx++

  // Build intervals: [bind_cb position, last cb_pop position].
  // CBPopOps inside nested regions (loops, compute) are projected
  // to their function-level ancestor.
  // If no cb_pop exists, end = last operation (conservative).
  for bindOp in compilerAllocatedBindCBOps:
    start = opIndex[bindOp]
    end = max(getBodyIndex(pop) for pop in cbPopUsers(bindOp))
    if end == start:
      end = lastOpIdx
    intervals[bindOp.type].append({start, end, bindOp.result})

  // Linear scan per type partition. Each partition gets a contiguous
  // block of indices starting at baseIndex + offset.
  offset = 0
  for (type, typeIntervals) in intervals:
    sort typeIntervals by start
    maxSlot = 0
    for interval in typeIntervals:
      // Expire: free slots where active.end <= interval.start
      for active in activeList:
        if active.end <= interval.start:
          free(slot[active])
      slot[interval] = allocateFirstFreeSlot()
      maxSlot = max(maxSlot, slot[interval])

    // Rewrite BindCBOp cb_index attributes for this partition.
    for (value, s) in partitionAssignments:
      bindOp[value].cb_index = baseIndex + offset + s
    offset += maxSlot + 1
```

The expiration condition `active.end <= interval.start` matches the DST register allocator's convention. Because operation indices are integers assigned to distinct operations, strict inequality and non-strict inequality produce the same result.

### Correctness with control flow

The algorithm assigns sequential indices to function-level operations only. Structured operations (`scf.for`, `ttl.compute`) occupy a single index in this sequence; their contents are not individually numbered. This is sufficient because `InsertIntermediateDFBs` and `InsertCBSync` both run before `LowerToLoops`, placing `bind_cb` and `cb_pop` at function-level while the IR is flat. These operations remain at function-level after loop creation because they bracket compute regions.

If a later pass restructures a `CBPopOp` into a nested region, it is projected to its enclosing operation at function-level via `Block::findAncestorOpInBlock`. This overestimates liveness -- the interval extends to the structured op rather than to the specific point where the pop occurs -- but never produces incorrect reuse.

Two DFBs consumed simultaneously by the same operation (e.g., both operands of a matmul) necessarily have overlapping intervals because both `bind_cb` must precede the consumer and both `cb_pop` must follow it. The linear scan assigns them different slots.

### Module attribute and runtime integration

After rewriting indices, the pass calls `getNextAvailableDFBIndex`, which returns `max(cb_index) + 1` across all `BindCBOp`s in the module. This is the index space usage, not a count of distinct DFBs (sparse indices inflate it). The pass verifies this does not exceed `kMaxCircularBuffers` (32).

The pass then sets `ttl.base_cta_index` on every function. Compile-time arguments (CTAs) to each kernel are laid out as `[CB indices..., other args...]`. `base_cta_index` is the starting index of the non-CB arguments -- equivalently, one past the last CB index. CB indices occupy `[0, base_cta_index)`.

Finally, the pass builds the `ttl.compiler_allocated_dfbs` module attribute with one entry per unique physical index, deduplicated from the potentially many `BindCBOp`s that now share an index. The Python runtime reads this attribute to allocate L1 buffers at dispatch time.

## Limitations and Future Work

The linear scan operates on a flat sequence of function-level operations. It cannot distinguish between branches of an `scf.if`, so DFBs used in mutually exclusive branches are treated as overlapping. The current pipeline does not produce conditional control flow around DFB lifecycle operations. The DSL evaluates Python `if` during tracing, so only the taken branch appears in the generated IR; runtime-conditional control flow (`scf.if` with conditions dependent on tensor values) is not supported by the frontend at this time.

Index reuse is restricted to compiler-allocated DFBs. User-declared DFBs retain their original indices because the same CB index is referenced by multiple threads (reader, compute, writer) to implement cross-thread data flow. Reusing a user index in one function would invalidate references in the others.

Liveness is computed at function-level granularity. If a `CBPopOp` is inside a structured op (loop, compute region), it is projected to its enclosing operation at function-level. The infrastructure for this exists (`Block::findAncestorOpInBlock`) but is not currently exercised: all compiler-allocated DFB lifecycle ops remain at function-level because `InsertIntermediateDFBs` and `InsertCBSync` run before loop creation, and Python control flow unrolls at trace time.

The type compatibility constraint prevents reuse across DFBs with different shapes or element types, even when L1 footprints happen to match. A size-based rather than type-based compatibility check could recover some reuse opportunities.
