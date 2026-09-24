# Persistent SRAM Storage

## Overview

Some operation sequences repeatedly update the same state: an accumulator, a cache, or an intermediate consumed by a separately compiled operation. The state must retain its SRAM allocation and contents between launches. Reinitializing it on each launch loses the previous result; releasing it when host dispatch returns can invalidate accesses still executing on the device.

Caller-owned TTNN tensors already provide backing that can outlive one operation. `SRAMStorage` makes ownership, initialization, and completion-aware release explicit in TT-Lang. It also provides the ownership boundary needed to plan several persistent allocations together later.

## Design

### Storage-Owned Payloads

An `SRAMStorage` object declares tensors, allocates their backing, initializes it once, and owns it until close. Operations borrow those tensors. Their addresses remain stable, including when different compiled operations access them. Aliases retain the same owner, so creating another reference does not create another allocation or another authority to release it.

Declaration and allocation are separate because placement should consider the complete set of requirements before reserving storage. The initial implementation uses ordinary owned TTNN allocations. It does not yet jointly pack them with compiler scratch; [SRAM Allocation](SRAMAllocation.md) describes the existing placement machinery.

`addressing="uniform"` requests one local address on every participating worker node. `addressing="per-node"` lets TTNN allocate each node independently and uses the Metal hybrid-allocation prerequisite defined in [SRAM Allocation](SRAMAllocation.md#runtime-allocation-and-binding). The tensor's required sharding mode defines how its logical dimensions map to those nodes. Per-node storage supports direct local access only; general tensor access and multicast require one common base address and are rejected.

### Example: Sharing State Between Operations

`op1` and `op2` are different application-defined `@ttl.operation` functions. In this example, `op1` updates persistent state from an input tensor, and `op2` reads that state to produce an output tensor. Each function defines its own computation and DFB accesses. `batches` contains input/output tensor pairs.

```python
storage = ttl.SRAMStorage(device=mesh)
try:
    state = storage.tensor(
        shape=(64, 32),
        shard_shape=(32, 32),
        nodes=((0, 0), (1, 0)),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        addressing="per-node",
        initialize="zeros",
    )
    storage.allocate()

    for input_tensor, output_tensor in batches:
        op1(input_tensor, state)
        op2(state, output_tensor)
finally:
    storage.close()
```

The allocation is initialized once before the loop. `op2` observes the state written by `op1`, and the next iteration retains that state. Passing `state` supplies the same tensor backing to both operations; their definitions determine how it is accessed.

`close()` waits for outstanding uses before releasing storage. The owner can instead be retained on a model or session object and closed later. A `with ttl.SRAMStorage(...)` block is optional convenience for calling `close()` on exit; lexical scoping is not required.

### Relationship to DFB Protocols

Persistent storage preserves the tensor's bytes. Each operation establishes its own DFB producer/consumer protocol over those bytes.

For a tensor-backed DFB, a launch binds the retained tensor, explicitly publishes its existing contents, and consumes them through the normal `wait`/`pop` protocol. An update writes back to the retained tensor before its access completes. The next launch starts with fresh DFB control state and publishes the updated contents again. A `pop` releases the acquired DFB block; it neither erases the tensor nor releases the persistent allocation.

Two synchronization requirements therefore remain distinct: the DFB protocol orders accesses within an operation, while storage dependencies order accesses between operations. Persistent ownership does not replace DFB synchronization. Binding, publication, and update semantics are defined in [DFB Management](DFBManagement.md#tensor-backed-storage).

### Completion Determines Release

Host dispatch return does not establish device completion. `SRAMStorage` records completion after submission and retains the allocation until that completion is observed. Its uses are conservatively serialized; proving read-only access could permit additional concurrency later.

Submission and close must agree on which operations own an outstanding use. Closing rejects new uses and waits for previously submitted ones. When an operation borrows several storage owners, the runtime acquires their ownership protections in a common order to avoid deadlock.

A launch can fail after enqueueing work. Such failure must not release its arguments immediately. If a completion record cannot be established, storage remains retained until recovery proves that accesses have finished. Failed cleanup remains retryable and never releases another owner's allocation.

## Algorithms

These algorithms state the ownership decisions; backend-specific address queries and event operations implement them.

```text
allocate(storage):
    validate all declarations and the device configuration
    reserve owned backing for every declaration
    initialize requested payloads and wait for initialization completion
    publish backing for all references together
    on failure: complete submitted initialization, then release new reservations

submit(operation, storage owners):
    validate and acquire permission to use every owner
    order execution after preceding uses of those owners
    bind their current tensor backing and submit the operation
    record completion, including when submission throws
    retain storage until completion is established

close(storage):
    reject new uses
    wait for outstanding accesses, recovering completion if necessary
    release only this owner's reservations
```

## Runtime Integration

### TTNN and External Launchers

The internal storage-backend interface defines validation, allocation identity, dependency insertion, completion recording, recovery, waiting, and release. `SRAMStorage` uses a TTNN implementation; ownership algorithms do not contain target-specific event or address handling. Two tensor objects have the same allocation identity only when their device addresses, logical and padded dimensions, dtype, layout, tile geometry, and sharding configuration match. This preserves ownership when an external launcher returns a new wrapper for unchanged backing without discarding a changed interpretation of those bytes.

The runtime reuses TTNN-owned tensors and command-queue events. Completion must cover every queue, sub-device, and remote user accessing the allocation. TTNN's default event selection follows the current sub-device stall group, which may exclude a service. The adapter therefore records an explicit selection instead of relying on that mutable default.

When mesh lifecycle queries are unavailable, the adapter restricts completion to queue 0 and uses `SubDeviceId(0)` unless the caller supplies another explicit sub-device list. The caller must keep the mesh and active sub-device manager unchanged until close. When the queries are available, the adapter also validates queue bounds, active sub-device membership, mesh liveness, and manager identity before allocation and submission.

An external launcher uses `storage.submit(launcher, ..., state, ...)` to borrow the same backing under the declared queue and sub-device contract. It must finish submitting its accesses before returning and must not retain or deallocate raw borrowed tensors. External sockets and services keep ownership of their own protocol state; a continuously running service needs its own access-completion mechanism before it can safely borrow persistent storage.

Device close, reset, mesh reshape, and manager changes must not race storage operations. Closing a device does not make a persistent tensor usable on a replacement device.

### Compilation and Caching

Persistent references become ordinary tensor arguments before compilation and cache lookup. Existing tensor-backed DFB lowering supplies their device addresses. A cached operation receives the current backing on every invocation; it does not own the payload or repeat initialization. Closing one `SRAMStorage` object therefore does not transfer its state to another object that reuses the same compiled operation.

## Follow-On Work

Joint placement combines declared persistent tensors, fixed existing allocations, and known temporary-storage requirements. Persistent contents remain live between accesses, so idle time alone cannot justify reusing their bytes. Reusing temporary storage across launches additionally requires a declared and enforced execution order.

Read/write effect information can permit concurrent read-only borrowing. These optimizations preserve the ownership and completion rules above.
