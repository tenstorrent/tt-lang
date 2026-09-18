# Persistent SRAM Storage

## Overview

Some applications update the same accumulator, cache, or intermediate tensor across separately compiled operation launches. The tensor must retain both its SRAM address and its contents until the application releases it. Host dispatch return is insufficient because device access may still be in progress.

`SRAMStorage` owns these tensors, orders their uses by device completion, and releases them explicitly. Joint placement also reserves compiler-planned operation arenas within the same owned pools. Persistent payloads remain live for the storage lifetime; arenas from different prepared operations can reuse bytes because the owner serializes their device execution.

This design preserves the Python operation and DFB APIs. It changes host allocation and runtime binding. Existing tensor arguments keep their addresses, physical pool bases are runtime arguments, and the LLKs are unchanged.

## API

Declarations and operation preparation precede allocation so placement can consider the complete requirement set.

```python
storage = ttl.SRAMStorage(device=mesh)
state = storage.tensor(
    shape=(64, 32),
    shard_shape=(32, 32),
    cores=((0, 0), (1, 0)),
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    addressing="uniform",
    initialize="zeros",
)

storage.prepare_operation(op1, input_tensor, state)
storage.prepare_operation(op2, state, output_tensor)
storage.allocate()

for input_tensor, output_tensor in batches:
    op1(input_tensor, state)
    op2(state, output_tensor)

storage.close()
```

`op1` and `op2` are different `@ttl.operation` functions. `prepare_operation` compiles one specialization without execution and records its immutable SRAM requirements. `allocate` performs joint placement, validates device programs, initializes persistent payloads, and publishes all bindings atomically. `close` waits for outstanding device accesses before releasing reservations.

A context manager is optional. Storage lifetime is controlled by the `SRAMStorage` object rather than Python lexical scope.

## Ownership and Completion

An `SRAMStorage` object has exclusive release authority for its pools and retained views. A `StorageReference` identifies one declared persistent tensor without exposing its provisional backing. Aliases preserve that identity and do not create another allocation owner.

Every operation prepared for joint placement must borrow at least one reference from the storage owner. Invocation through that reference acquires the owner, inserts dependencies on the preceding completion records, submits the operation, and records completion on every declared queue and sub-device. This enforced ordering is the proof that arenas belonging to different prepared operations cannot execute concurrently.

Host exceptions do not prove that a submission failed before enqueue. When completion recording fails, the owner retains all reservations until backend recovery establishes completion. Cleanup releases only resources acquired by the transaction and remains retryable.

An external launcher can borrow persistent tensors through `storage.submit(launcher, ..., state, ...)`. The launcher must finish submitting its accesses before returning and must use the owner's declared queues and sub-devices. The joint operation arena is available only to its prepared TT-Lang operation.

## Joint Placement

### Requirements

Each persistent declaration contributes one movable requirement with persistent lifetime. Each compiler-planned operation arena contributes one movable requirement with invocation lifetime. Tensor arguments supplied by the application contribute fixed external requirements; the allocator neither moves nor owns them. The live TT-Metal allocator state accounts for those existing allocations when the data budget is computed.

Each requirement defines a per-location extent, alignment, addressing mode, and address-equality domains. Uniform requirements use one equal base across their participating cores. Per-core requirements allow independent physical bases. A pool never mixes the two addressing modes.

### Conflict Construction

Persistent regions conflict with every region on an overlapping location because their contents remain live until close. Arenas within one operation conflict because they can be used concurrently during that launch. Arenas from different operations do not conflict because the storage owner establishes a device-completion dependency between launches.

Requirements with overlapping locations form one pool component for an addressing mode. Disjoint components use independent reservations. A component uses the maximum required alignment and delegates byte placement to the existing C++ `SRAMAllocator` strategy interface.

```text
plan_joint_storage(persistent requirements, prepared operations):
    validate ownership, lifetime, addressing, and fixed tensor contracts
    collect persistent regions and compiler arena regions
    partition regions by addressing mode and overlapping locations
    for each partition:
        add conflicts for overlapping persistent regions
        add conflicts for arenas from the same operation
        place aligned regions with the selected SRAMAllocator strategy
    reject any location whose combined reservations exceed its data budget
    return pool reservations, view offsets, and efficiency metrics
```

This partition can reserve more bytes than separate allocations. A pool spanning the union of uneven core sets reserves its full per-location extent on every member core. The runtime therefore reports the measured result and does not claim that pooling reduces SRAM use.

### Reservation and Publication

The placement offsets are relative to an owned pool. TTNN allocates each pool through the normal Metal allocator, and owner-retaining tensor views bind persistent declarations and operation arenas to their assigned offsets. A view can cover a subset of the pool's cores while retaining the complete pool allocation.

Program preparation revalidates the requirement contract against the final views. Ownership, extent, alignment, addressing, domains, uses, and arena layout must match the provisional contract. A tensor argument may use another physical base on a later launch when every other requirement remains identical. This permits repeated use of one prepared specialization with compatible input and output allocations.

Reservations remain provisional while every prepared operation is compiled and its Metal program layout is finalized. Persistent payload initialization occurs only after all program checks succeed. The owner then records and waits for initialization completion before publishing tensor references and operation bindings together. Any allocation, view construction, program preparation, initialization, or completion failure releases the provisional views and pools in reverse order.

```text
allocate_joint_storage(plan):
    reserve every pool and construct owner-retaining views
    bind physical pool bases as runtime arguments
    compile and finalize every prepared device program without dispatch
    initialize persistent payloads once
    record and wait for initialization completion
    publish all references and operation bindings atomically
    on failure: establish completion and release new views and pools
```

## Program Capacity

Data capacity and program capacity are independent. Metal's allocatable SRAM interval already excludes firmware and reserved program memory. Joint placement compares data reservations with that interval and does not subtract compiled code a second time.

`tt::tt_metal::experimental::program_preparation::prepare` compiles kernels and finalizes program offsets and runtime-argument configuration without dispatching the workload. Finalization rejects a program configuration that exceeds the architecture's kernel-configuration buffer. The result reports the maximum finalized configuration size and kernel-binary size. Kernel binaries larger than the prefetcher cache remain valid because Metal dispatches them without that cache. TTNN exposes preparation through `ttnn.experimental.prepare_generic_op`. Target memory maps and processor limits remain inside TT-Metal.

Pool bases remain runtime arguments. Changing a physical reservation address therefore updates invocation arguments without creating a new kernel specialization.

Data overflow is rejected before reservation. Program-configuration overflow after provisional reservation triggers transactional rollback before persistent initialization or publication. Tests distinguish exact-fit data and program-configuration cases, data overflow before program preparation, and program-configuration overflow after reservation.

## Launch Behavior

Persistent payloads are initialized once during allocation. Before each prepared operation launch, the runtime resets only that operation's DFB control-record views. The preceding owner completion dependency ensures that the reset cannot race an earlier operation using the shared arena. Scratch payload bytes are not cleared unless operation semantics require initialization.

A tensor-backed DFB publishes the persistent tensor contents through its ordinary `wait` and `pop` protocol on every launch. `pop` releases a logical DFB block; it does not erase the tensor or release the persistent allocation. Cross-launch storage ordering and within-launch DFB synchronization remain separate requirements.

## Metrics

The plan and runtime report physical shard bytes, not only compiler offsets.

- `required_peak_bytes`: for each location, persistent extents plus the largest prepared operation's simultaneous arena extent, summed across locations.
- `planned_reservation_bytes`: pool extents after allocator alignment and location replication.
- `actual_reservation_bytes`: allocator-reported pool shard bytes after TTNN allocation.
- `fragmentation_bytes`: actual reservation minus required peak.
- `efficiency`: required peak divided by actual reservation.
- `separate_planned_peak_bytes`: the aligned fixed-layout estimate if each requirement uses its own reservation and operation scratch is released between launches.
- `preparation_seconds` and `allocation_seconds`: host time spent preparing specializations and completing the allocation transaction.
- Per-operation program measurements: maximum finalized configuration bytes and maximum kernel-binary bytes.

The estimate does not establish that pooling saves memory. Device benchmarks compare `actual_reservation_bytes` with allocator-reported physical reservations from separate allocation and record both values with preparation time.

## C++ Contracts

The architecture-neutral placement API is declared in [`SRAMAllocator.h`](../../include/ttlang/Dialect/TTL/Transforms/SRAMAllocator.h). `SRAMAllocationProblem` contains region extents, the complete conflict graph, alignment, a reserved prefix, and a byte budget. `SRAMAllocator::allocate` validates this input, calls the selected strategy, validates the complete solution, and returns offsets plus the arena high-water mark without modifying compiler IR. `createSRAMAllocator` selects a registered strategy by its stable name. The exact and greedy implementations use the same contract.

The runtime ownership interfaces are:

- `tt::tt_metal::experimental::retained_buffer_view::create`: creates a bounded SRAM view, preserves uniform, range-lockstep, or per-core addressing, and retains its source allocation.
- `ttnn::experimental::create_sharded_tensor_view`: applies a `TensorSpec` and tensor topology to that retained view.
- `tt::tt_metal::experimental::program_preparation::prepare`: performs non-dispatch compilation and finalization and reports program-memory use.
- `ttnn::experimental::prepare_generic_op`: exposes preparation for TT-Lang's generic operation descriptor.

The TT-Lang runtime depends only on these common interfaces. Wormhole and Blackhole address rules and program limits remain behind TT-Metal APIs.

## Assumptions and Constraints

Persistent declarations support BF16 and FP32 tiled tensors with height, width, or block sharding. Uniform and per-core allocation are supported. Per-core persistent tensors permit direct local access; general tensor access and multicast still require a uniform base.

One joint storage owner supplies one completion domain. Selected device domains, PipeNet-owned storage, Metal DFB reconfiguration, synchronized reset resources inside an invocation, fabric routes, and opaque runtime-resource factories are rejected during operation preparation because their storage and completion requirements are not represented in the joint plan. External launchers of persistent tensor accesses remain supported through `storage.submit`.

Device close, reset, mesh topology changes, and sub-device-manager changes must not race storage use. A closed or replaced device does not transfer persistent contents to another device.

## Follow-On Work

Joint placement can incorporate PipeNet, DFB reconfiguration, and selected device domains after their resources expose immutable requirements and completion contracts through the same preparation interface. Read-only effect proofs can permit concurrent borrowing by retaining conflicts between arenas whose executions may overlap. Conditional reservation against complete per-core free-interval snapshots can place pools into fragmented gaps without relying on a stale occupancy query.
