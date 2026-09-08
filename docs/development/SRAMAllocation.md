# Compiler-Managed SRAM Allocation

## Purpose

TT-Lang normally assigns concurrently live dataflow buffers (DFBs) to TT-Metal descriptor indices; DFBs with disjoint lifetimes can reuse an index. Wormhole B0 provides 32 indices and Blackhole provides 64. An operation can therefore exhaust indices while sufficient SRAM remains.

`--ttl-memory-model=compiler-sram` replaces descriptor-indexed storage with compiler-assigned SRAM byte ranges. The Python DFB API and its producer/consumer semantics remain unchanged. `metal-cb` remains the default.

An *arena* is the node-local SRAM reservation for one operation execution. A *control record* holds one storage owner's producer and consumer counters. Two payloads have *noninterfering lifetimes* when enforced completion order prevents simultaneous use. An *address-bearing descriptor* supplies a buffer address and geometry to generated device code without a physical DFB index. "SRAM" names the backend; "L1" names the device memory and its capacity budget.

| Property | `metal-cb` | `compiler-sram` |
| --- | --- | --- |
| Allocated identity | TT-Metal DFB index | Logical DFB index and compiler storage owner |
| Storage address | TT-Metal descriptor | Compiler arena or tensor base plus byte offset |
| Capacity limit | SRAM capacity and 32 or 64 descriptor indices | SRAM capacity, control records, and alignment |
| Payload reuse | Requires the Metal descriptor and backing-storage contracts | Requires noninterfering completed lifetimes or an explicit validated allocation group |
| Producer/consumer state | TT-Metal DFB interface state | Two 32-bit page-sequence counters per storage owner |
| Tensor-backed storage | Installed through a TT-Metal descriptor | Addressed directly through the tensor runtime argument |
| Allocation groups | Reuse a physical descriptor and its storage contract | Share one validated storage owner and control record |
| Reset and reconfiguration | Blackhole TT-Metal interface reset and runtime descriptor reconfiguration | Blackhole address-based state reset with compiler-fixed geometry |
| External C++ DFB access | Numeric index or typed descriptor bound to a TT-Metal DFB | Typed descriptor bound to compiler-assigned storage |
| PipeNet receiver | TT-Metal DFB descriptor or tensor-backed address | Finalized arena or tensor-backed address for local and generated fabric transfers; no TT-Metal DFB descriptor |

Shared terminology is defined in the [TT-Lang specification glossary](../sphinx/specs/TTLangSpecification.md#appendix-a-glossary). The DFB protocol and lifecycle rules are defined in [DFB Management](DFBManagement.md).

## Allocation Model

An invocation of a compiled Python `ttl.operation` with a nonempty allocation plan owns one compiler-managed arena on each participating worker node. Every arena uses the same relative layout. Kernels receive the node-local arena base as one common runtime argument, so the argument count does not depend on the number of logical DFBs.

The arena has two sections:

```text
0                                                   arenaBytes
+----------------------+----------------------------------+
| 8-byte state records | aligned, reusable payload ranges |
+----------------------+----------------------------------+
```

Each storage owner has one 8-byte record. An ungrouped logical DFB is its own storage owner. A validated allocation group has one storage owner shared by its members. The first 32-bit word is the published-page sequence and the second is the consumed-page sequence. Allocation-group validation requires one element type and therefore one page size. Page units preserve one cursor interpretation when members use different pages-per-block and block-count values. Separate words allow the producer and consumer to update state without an atomic read-modify-write operation.

Payload storage can overlap when the compiler proves that the corresponding lifetimes cannot be active concurrently. Control records are independent except within an explicit allocation group whose validation proves state ownership transfer.

For `S` storage owners and target alignment `A`, the payload section begins at:

```text
controlEnd = roundUp(8 * S, A)
```

For compiler-owned storage with page size `P`, pages per block `T`, and block count `B`, the payload extent is:

```text
extent = roundUp(P * T * B, A)
```

Packed-format metadata is included in `P`. An allocation group reserves the largest compiler-owned payload extent required by any member. Tensor-backed storage uses the tensor's existing node-local SRAM address and adds no payload bytes to the arena. Its control record remains in the arena. The complete arena size is the maximum of the control section end and every compiler-owned payload end. An empty allocation plan needs no arena; lifecycle synchronization scratch remains a separate runtime allocation.

## Conflict Analysis

Allocation consumes the existing logical-identity, allocation-group, and completion-aware lifetime analyses. The compiler validates every allocation group and builds the complete conflict relation before changing IR. Unknown launch domains, unproved completion, concurrent lifetimes, and incompatible storage ownership remain conflicts.

The shared storage conflict analysis accepts an explicit storage mode. Metal storage includes conflicts caused by runtime descriptor installation and Metal-managed backing changes. Compiler-managed storage excludes those conflicts because each logical DFB's page size, pages per block, block count, and storage capacity remain constant during execution, and each validated storage owner has a control record. This distinction permits byte reuse across a reconfiguration boundary after the prior lifecycle ends while preserving DFBs that remain live across the boundary.

Validated allocation-group members are collapsed into one storage owner. The owner conflicts with another owner if any member pair conflicts. This preserves all lifecycle conflicts while allowing the explicit ownership transfer represented by the group. The existing group validator proves ordering, capacity, cursor continuity, and storage compatibility; the SRAM allocator does not duplicate those checks.

Control records are node-local, so group members on disjoint nodes may share a storage index and control offset while using different payload backing. On a shared node, a change between tensor and arena payloads, or between tensor payloads, requires a synchronized reconfiguration that ends the earlier member's lifetime and resets the owner's control record. The finalizer records the earlier DFB, later DFB, node, and reset boundary for each proved change. EmitC and host preflight require that exact handoff; a reset involving either DFB alone does not authorize a backing change.

Tensor-backed DFBs require an exact, non-empty launch-node domain. On a shared launch node, partial byte-range overlap is rejected because it does not represent a complete storage ownership transfer. Identical byte ranges are permitted for the same storage owner or for nonconflicting lifetimes. Disjoint ranges do not alias.

This design reuses one lifetime model for both memory backends. The allocator cannot serialize operations or remove a conflict to make a program fit.

```text
buildStorageConflicts(lifetimes, storageMode):
    conflicts = empty graph
    for each unordered pair (left, right):
        for each worker node where both may be active:
            if the node association or completion order is unknown:
                add conflict(left, right)
            else if neither lifetime completes before the other begins:
                add conflict(left, right)
            if backing-storage ownership is incompatible on this node:
                add conflict(left, right)
            if storageMode is metal-cb and either DFB uses reconfiguration backing or descriptor installation can overwrite live state:
                add conflict(left, right)

    return conflicts
```

```text
buildStorageOwners(regions, allocationGroups, conflicts):
    validate allocationGroups with the shared allocation-group validator
    create one owner for each group and each ungrouped region

    for each owner:
        owner.extent = maximum compiler-owned extent among its members

    for each unordered owner pair (left, right):
        ownerConflict(left, right) = any member pair in conflicts

    validate tensor-backed byte-range aliases on every shared launch node
    return owners and ownerConflict
```

Possible launch domains use the same rules as exact domains and remain conservative. The compiler authorizes overlap only when every common worker node has a proven completion order.

## Placement Interface and Algorithms

Placement uses a reusable C++ allocator API. The caller converts storage owners, conflicts, target alignment, and capacity into an immutable allocation problem; strategies do not inspect IR. Each owner with a compiler-owned payload contributes one allocator region. Tensor-backed owners consume control records but add no allocator regions. A solution contains one byte offset per allocator region and the payload high-water mark.

Every allocator result passes the same validation before IR mutation. Validation requires the correct offset count, target alignment, offsets at or above the payload base, intervals within the SRAM budget, disjoint intervals for every conflict, and an exact payload high-water mark. Allocation policy cannot weaken these invariants. The domain allocation entry point maps offsets to storage owners and retains the control prefix when computing the arena size.

### C++ Allocator Contract

The reusable placement API is declared in [SRAMAllocator.h](../../lib/Dialect/TTL/Transforms/SRAMAllocator.h). The caller builds a `SRAMAllocationProblem` with aligned, nonzero `regionBytes`, an `InterferenceGraph` whose edges prohibit overlap, `alignmentBytes`, the first usable `payloadBaseOffset`, and the total `budgetBytes`. The conflict graph is symmetric with no self-edges. Region indices are stable and map to storage owners in the caller.

```cpp
FailureOr<std::unique_ptr<SRAMAllocator>> createSRAMAllocator(
    llvm::StringRef name, const SRAMAllocatorOptions &options,
    std::string &failureReason);

FailureOr<SRAMAllocationSolution> SRAMAllocator::allocate(
    const SRAMAllocationProblem &problem,
    SRAMPlacementFailure &failureDetail) const;

FailureOr<llvm::SmallVector<SRAMAllocationDomainSolution>>
SRAMAllocator::allocateDomains(
    llvm::ArrayRef<SRAMAllocationDomainProblem> domains,
    SRAMAllocationDomainFailure &failureDetail) const;
```

`SRAMAllocatorOptions::minimumArenaSearchLimit` is positive and bounds the exact strategy; the factory rejects zero. A solution contains one arena-relative byte offset per region and `arenaBytes`, the maximum payload end (zero for no regions). `allocate` validates the immutable problem, calls the strategy, and validates its solution before the caller changes IR. Validation checks offset count, alignment, budget, conflict disjointness, and the exact high-water mark. On failure, `SRAMPlacementFailure` reports the category, reason, and optional region index. A strategy implements `getName()` and private `allocateImpl()`; the factory gives it a stable option name. All strategies share the same input and validation contract.

`allocateDomains` applies that contract to independently addressable layouts. Each domain maps its region indices to storage owners; the caller proves that domain bindings do not overlap. It validates all domains before placing any of them, then returns one placement and arena size per domain. A failure returns no partial result and identifies the domain, failure category, and affected storage owner when available. A control-only domain retains its aligned control prefix. The exact strategy's work limit applies to each domain separately.

| Strategy | Selection | Guarantee |
| --- | --- | --- |
| `multi-order-decreasing` (default) | Run stable and degree-aware first-fit; retain the smaller arena, using stable placement on a tie. | Never larger than stable first-fit for the same problem. |
| `first-fit-decreasing` | Place larger regions first at the lowest aligned legal offset. | Deterministic feasible placement when it fits. |
| `best-fit-decreasing` | Place larger regions first in the finite legal gap with the least unused space. | Deterministic feasible placement when it fits. |
| `minimum-arena` | Enumerate relevant aligned offsets with bounded branch-and-bound search. | Proves the minimum or reports infeasibility or an exhausted search limit. |

All greedy strategies sort by decreasing extent; stable region order resolves remaining ties. For each region, the allocator sorts already placed conflicting intervals by address. First-fit chooses the first gap that fits. Best-fit chooses the finite fitting gap with the least unused space, resolving ties by lower address; if none exists, it appends the region. The scan advances beyond overlapping blockers, so each new interval avoids every conflicting interval. Nonconflicting regions may share an address.

```text
placeGreedy(problem, order, gapSelection):
    placed = []
    for region in order:
        blockers = sort(placed regions that conflict with region, by offset)
        gaps = aligned free intervals between blockers
        offset = choose first fitting gap or smallest fitting finite gap
        if no finite gap fits: offset = aligned end of blockers
        record offset and append region to placed
    return offsets and maximum payload end
```

Degree-aware order resolves equal extents by decreasing conflict count. Running both orders retains first-fit's deterministic result whenever the arena sizes tie.

```text
placeMultiOrder(problem):
    stable = placeGreedy(problem, decreasing extent then region index, first-fit)
    degreeAware = placeGreedy(problem, decreasing extent then conflict count, first-fit)
    return degreeAware if its arena is smaller, otherwise stable
```

Minimum-arena search separates disconnected conflict components. They have no mutual disjointness requirement, so their placements overlay at the same payload base and the total minimum is the largest component minimum. A greedy placement supplies an upper bound; a weighted conflict clique supplies a lower bound. An optimal placement can be shifted left until each nonzero offset touches the end of a conflicting interval. Thus candidate offsets are subset sums of aligned region extents, not every address. The search visits candidates in deterministic order, rejects overlaps and results above the current best, and stops when the lower bound is reached.

```text
placeMinimumArena(problem, workLimit):
    greedy = better of first-fit and best-fit
    for component in connectedComponents(problem.conflicts):
        lower = weightedCliqueBound(component)
        upper = greedy component placement if it fits the budget
        candidates = aligned subset sums of component extents below upper or budget
        search placements over candidates, pruning conflicts and results above upper
        if workLimit is exhausted: report inconclusive with the work count
        if no placement fits: report proven budget infeasibility
        retain the minimum component placement
    overlay component placements at payloadBaseOffset
    return offsets and maximum component payload end
```

Greedy placement takes `O(P^2 log P)` time for `P` compiler-owned regions; multi-order repeats it twice. The exact search can take exponential time, so its candidate-generation and partial-placement work share one explicit limit. Exhausting that limit is not a proof of infeasibility, even if a greedy placement fits. The allocator uses normalized byte requirements; target alignment and capacity enter through the problem, while MLIR diagnostics, storage-owner mapping, and runtime allocation remain in the caller.

## Reset and Reconfiguration

Blackhole selected reset, reset-all, and DFB reconfiguration use the existing DFB-interface synchronization LLK with zero Metal DFB masks. Each distinct synchronization boundary receives one 16-byte scratch record containing three arrival words and one coordinator release word. The combined scratch allocation is rounded once by the runtime allocator; the cost is per boundary, not per DFB.

```text
resetBoundary(selectedDFBs, synchronizationRecord):
    synchronize the DFB interface owners; the second data-movement processor coordinates release
    for each selected DFB:
        store zero to its published and consumed words
    if selectedDFBs is not empty:
        synchronize the processors again
```

The first synchronization completes earlier asynchronous work before state changes. The second prevents any participant from beginning later DFB work before all selected records are cleared. DFB interface owners perform idempotent zero stores; the MATH processor does not access the records.

Reconfiguration uses the same algorithm. Lifetime analysis records which logical DFB lifecycles terminate at each boundary. Only those control records are cleared. A logical DFB that remains live across the boundary retains its record and payload. When no lifecycle terminates, the boundary requires only the first synchronization.

Reset calls carry DFB identities for node selection but use only arena control addresses. A kernel that only synchronizes DFB state therefore does not require tensor payload arguments.

Wormhole continues to support ordinary compiler-managed allocation, transfer, and compute. Synchronized reset and reconfiguration remain Blackhole-only because their current LLK protocol depends on Blackhole processor synchronization behavior. Wormhole compilation rejects those operations with a target-specific diagnostic before allocation.

## Compute Tile Dimensions

Metal compute uses DFB descriptors to configure tile formats and dimensions. Address-based compute has no descriptor, so each operand type carries its format, byte size, height, width, and direct-to-destination choice. BF16 and FP32 support 1x32, 2x32, 4x32, 8x32, 16x16, 16x32, 32x16, and 32x32 tiles on Blackhole and Wormhole. The shared compute-target interface validates these dimensions; the target adapter supplies architecture-specific LLK arguments.

A kernel may use different tile dimensions in successive operations. The compute context tracks input formats, page sizes, face row heights, and face counts, plus output format and dimensions. An output dimension change requires PACK reconfiguration even when two tiles occupy the same number of bytes.

```text
configureCompute(inputA, inputB, output):
    if no configuration exists:
        initialize UNPACK, MATH, and PACK
    else:
        reconfigure each changed input format, page size, face row height, or face count
        reconfigure PACK if its format, page size, or tile dimensions changed
        if output tile dimensions changed:
            initialize PACK while preserving address modifiers
    record the active configuration
```

Shared compute helpers depend on tile properties, not SRAM offsets, capacities, or runtime-argument indices. This allows multiple DFBs with the same tile properties to use one helper implementation and bounds generated program size as the number of logical DFBs grows.

## External C++ Interface

`ttl.dfb_descriptor(dfb)` lowers to a C++ template type containing page size, pages per block, block count, shared storage capacity, state offset, payload offset, and an optional tensor common-argument index. Its `bind()` method obtains the state address from the arena. The payload address comes from either the arena or the tensor's existing common runtime argument. External functions therefore require no Metal DFB index and no additional runtime argument per DFB.

On a compute thread, the generated `ComputeDFBDescriptor` also carries the tile format, height, width, and direct-to-destination choice. Its storage offsets remain fixed compile-time values; compute helpers specialize only on tile properties. In finalized allocation metadata, `l1_payload_offset` is absolute within the arena. The generated descriptor stores the difference between that address and its control-record address.

External compute descriptors accept BF16, FP32, BFP4_B, and BFP8_B tiles. The target interface requires complete 32x32 tiles for block-float formats. Native generated compute remains limited to BF16 and FP32; an external C++ kernel supplies the mixed-format compute sequence.

An external compute adapter selects Metal numeric-index operations or address-based target operations using `TTLANG_DFB_STORAGE_COMPILER_SRAM` (0 for Metal, 1 for compiler-managed SRAM). Generated device code defines the marker before including the external header; architecture-specific operations remain behind `ttlang::l1::target`. Opaque C++ bodies are outside compiler compute analysis, so the enclosing operation declares any required compute configuration. [External functions](../sphinx/reference/external-functions.md#template-arguments) specifies the C++ interface.

```text
bind(descriptor):
    stateAddress = target.arenaBase() + descriptor.stateOffset
    if descriptor has tensor backing:
        payloadAddress = target.commonArg(descriptor.tensorArg) + descriptor.payloadOffset
    else:
        payloadAddress = stateAddress + descriptor.payloadOffset
    return AddressDFB(stateAddress, payloadAddress, descriptor.geometry, descriptor.storageCapacity)
```

External calls can declare `DFBEffect` entries for protocol operations. These effects participate in lifetime and conflict analysis; a DFB dependency without effects remains live until completion is proved. Compiler-managed storage rejects unknown DFB access, numeric DFB template arguments, and DFB function arguments. External C++ code uses `ttl.dfb_descriptor(dfb)` as a template argument to bind compiler-managed storage.

## PipeNet Transfers

Local PipeNets use the existing transfer schedule, transport, capacity, and synchronization protocols. Finalized DFB storage determines a receiver's address: compiler-owned payloads use the arena base plus `l1_payload_offset`; tensor-backed payloads use the retained tensor base plus their declared byte offset. Neither requires a TT-Metal DFB descriptor. Producer and wait launch domains include explicit DFB operations and external-call `DFBEffect` declarations through the shared DFB access interface.

```text
bindLocalPipeReceivers(allocation, computedReceivers):
    arena = reserveZeroedArena(allocation.arenaBytes)
    pipeResources = allocatePipeScratchAndSynchronization()
    for each receiver in computedReceivers:
        if receiver has tensor backing:
            address[receiver] = retainedTensorBase(receiver) + receiver.byteOffset
        else:
            address[receiver] = arena.base + receiver.l1_payload_offset
        require address[receiver] fits the device address type
    append tensor addresses, receiver addresses, pipe resource addresses, arena.base
    retain arena and pipeResources until device completion
```

The finalized `ttl.crta_indices` list determines the tensor-address prefix even when a tensor-backed DFB outlives its original function operand. Cache identity includes tensor-backed receiver addresses, while ownership accounting excludes caller-owned tensors. The 32- or 64-index TT-Metal DFB limit does not apply to these logical DFBs; PipeNet transport and semaphore limits are unchanged.

Generated inter-device transfers use the same receiver address rule. The arena is one TT-Metal mesh buffer with a common address on participating devices, so its payload offset identifies the destination allocation. A tensor-backed receiver has a computed address only when every participating node has the same tensor base and byte offset. Generated fabric requires a computed receiver address and diagnoses other cases before transfer lowering. Local PipeNets can use receiver publication, including when Metal shared descriptors can change backing storage. Fabric routing resolves device targets independently of these storage decisions.

```text
for each logicalDevice:
    descriptor = buildProgramDescriptor(receiverAddresses, pipeResources)
    bindings[logicalDevice] = planFabricRoutes(descriptor, logicalDevice)
for each logicalDevice:
    apply bindings[logicalDevice] to its descriptor
dispatch the mesh program
```

Planning all bindings before applying any of them prevents an invalid route from partially configuring the mesh program.

## Target Interfaces

Common allocation and lowering contain no architecture branches. `compiler_l1_target.h` provides arena-base access, SRAM loads and stores, producer/consumer completion, and processor ownership. `compiler_l1_compute.h` implements address-based compute operations; `compiler_l1_compute_target.h` adapts their LLK calls to Wormhole and Blackhole signatures.

## Runtime Arena

The runtime allocates the control records and compiler-owned payloads as a row-major, height-sharded TTNN SRAM tensor with one equal-length row per participating worker node. Height sharding directly represents one arena row per node. Width sharding provides no capacity benefit, and block sharding introduces an unused partition dimension. Tensor-backed payloads retain their existing height-, width-, or block-sharded TTNN allocations. A mesh arena uses TT-Metal's lockstep allocation, which assigns the same SRAM address on every selected device. Device-domain descriptors therefore combine device-specific logical coordinates with one common arena base.

Before reserving an arena, the runtime validates each tensor-backed segment against its actual tensor's type, tile, shard nodes, and byte range. Tensor-backed ring capacity must equal the declared DFB capacity so the runtime validates every address the ring can use. It also rejects undeclared overlap at the tensors' current addresses. Identical ranges are permitted for one storage owner or for reuse of the same declared tensor backing after compiler-proved lifetime separation.

The runtime passes the arena as an auxiliary `generic_op` input without changing the user output position. Each invocation waits for device completion before releasing its arena, including after descriptor preparation or dispatch fails. A synchronization failure retains the arena for the process lifetime because completion is unknown. This wait adds host latency to each invocation with a nonempty allocation plan.

The runtime zero-initializes synchronization scratch. Existing semaphore, runtime-argument, define, and external-resource ownership contracts compose with the arena. Runtime resource caching includes the allocation metadata, reset count, and validated backing handoffs, so different storage contracts do not share resources.

Finalization records `ttl.memory_model`, `ttl.l1_arena_bytes`, and one entry per logical DFB in `ttl.dfb_allocations`. Entries are ordered by `dfb_index`, which equals each entry's array position. Each entry identifies its storage owner (`storage_index`), shared capacity (`storage_capacity_pages`), and arena-relative control-record offset (`l1_offset`). Members of an allocation group share these values. Compiler-owned payloads also record an arena-relative offset (`l1_payload_offset`) and aligned extent (`l1_allocation_bytes`); tensor-backed payloads instead retain their tensor segment metadata and consume no arena payload. Before code generation, EmitC checks record ownership, bounds, target alignment, shared capacity, and agreement with each DFB's element type and page count. A generated kernel's compile-time argument 0 identifies the common runtime argument containing its local arena base; subsequent DFB compile-time arguments identify allocation entries. The C++ `PayloadOffset` template parameter is relative to the control record for arena payloads. `ttl.compiler_sram_reconfiguration_resets` records which control records are cleared at each reconfiguration boundary.

Uniform allocation reserves the largest required arena on every participating worker node. This can waste capacity when activity is sparse. Per-node layouts require node-specific allocation metadata and are an extension of this design.

## Allocation Report

`--ttl-sram-allocation-report` emits schema-versioned JSON records to stderr with the prefix `ttlang-sram-report: `. It is disabled by default and emits nothing for `metal-cb`. The compiler record is emitted after allocation validation and before IR mutation. A runtime record is emitted for each arena reservation, including calls that reuse a compiled operation. Reporting does not change placement.

The compiler record identifies storage `owners`, logical DFB `regions`, overlapping owner pairs in `reused_ranges`, and the existing `logical_conflicts` and `lifetimes` analysis evidence. Conflict reasons are recorded before allocation-group ownership is applied. Lifetime event IDs identify partial-order events, not elapsed time or a total execution order. Overlap byte counts are not additive when more than two owners share the same range.

| Compiler field | Meaning |
| --- | --- |
| `arena_bytes_per_node` | Planned control prefix and payload high-water mark. |
| `control_record_bytes`, `control_padding_bytes` | Control state and alignment padding. |
| `payload_extent_sum_bytes` | Sum of distinct compiler-owned storage-owner extents after allocation-group consolidation. |
| `payload_union_bytes` | Number of distinct payload addresses occupied by those extents. |
| `payload_reuse_bytes` | Extent sum minus union; excludes sharing within an allocation group. |
| `payload_gap_bytes` | Payload high-water mark minus union; an unused address gap, not distance from optimal placement. |

The runtime record has `phase: "runtime"` and `scope: "arena-reference-device"`. It reports the requested arena bytes per node and the actual reserved bytes derived from the arena buffer's aligned page size and uniform page count. `node_count` and `reserved_bytes_on_reference_device` describe only the participating nodes on the reference device. Existing tensor payloads, PipeNet scratch, external resources, and program storage are outside this measurement. A control-only arena may omit trailing compiler alignment padding from the runtime request.

A compiler-only report can be inspected with:

```sh
ttlang-opt test/ttlang/Dialect/TTL/Transforms/compiler_sram_multi_order.mlir \
  -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-sram sram-allocation-report=true})' \
  > /tmp/allocated.mlir 2> /tmp/sram-report.log
```

## Memory Utilization

Storage efficiency comes from six decisions:

1. Completion-aware conflicts permit payload overlap across sequential lifetimes, formats, and reconfiguration epochs.
2. Greedy strategies search aligned gaps; the default compares two placement orders and retains the smaller arena.
3. Payloads are ordered by decreasing size so large extents have more legal offsets available.
4. Allocation groups share one payload envelope and one control record after the shared validator proves ownership transfer.
5. Tensor-backed payloads remain in their existing L1 tensors and consume no duplicate arena payload storage.
6. The arena uses one runtime argument, independent of logical DFB count.

The fixed control cost is `roundUp(8 * S, A)` for `S` storage owners. For 96 ungrouped one-page BF16 DFBs, simultaneous lifetimes require 196,608 payload bytes and 768 control bytes before final arena alignment. If all 96 lifetimes are sequential, they reuse one 2,048-byte payload range and retain 768 control bytes. The control records are then 27% of the 2,816 bytes before final alignment. A validated allocation group reduces both payload envelopes and control records because its members explicitly transfer ownership. Further reduction would require inferring state ownership transfer without an allocation group or packing independently written producer and consumer state, which would weaken the ownership contract or require atomic updates.

The bounded `minimum-arena` strategy measures the gap between the selected greedy placement and the proven optimum for small problems. Its exhaustive search may cost substantially more compilation time; the default multi-order strategy retains the first-fit result unless the second order reduces the arena.

## Implemented Contract

- One uniform worker-node arena layout at a lockstep address across the selected devices.
- Compiler-owned static payloads and existing height-, width-, or block-sharded tensor-backed payloads.
- Validated allocation groups with one shared state record and the largest required compiler-owned payload envelope.
- One-block transactions and complete-capacity tensor publication or consumption, with positive capacity below `2^31` pages.
- Consumer-owned replacement writes into the acquired read window without changing occupancy or sequence state.
- BF16 and FP32 address-based compute for 1x32, 2x32, 4x32, 8x32, 16x16, 16x32, 32x16, and 32x32 tiles.
- Address-based tensor transfer, elementwise compute, matmul, reductions, broadcast, transpose, and selected activation operations covered by the implementation tests.
- Typed external C++ calls with explicit DFB effects, compiler-owned or tensor-backed payloads, BF16/FP32 elementwise multiplication and block matmul, and BF16 by BFP4_B/BFP8_B block matmul.
- Device-domain and mesh program placement with declarative external runtime resources.
- Blackhole selected reset, reset-all, and reconfiguration.
- Local and generated inter-device PipeNet transfers with compiler-owned or tensor-backed receivers.
- Wormhole allocation, transfer, compute, external descriptors, and local PipeNet compilation without reset or reconfiguration.

## Validation

| Scenario | Evidence |
| --- | --- |
| [Sub-tile compute](../../test/python/test_subtile_compute.py) | 424 Blackhole device-correctness cases cover BF16/FP32, DRAM/SRAM tensors, both storage backends, both compiler allocation strategies, tensor-backed multi-page expressions, geometry changes within one kernel, and typed external descriptors. |
| Blackhole transfer and compute | Device correctness across BF16/FP32, DRAM/TTNN L1 tensors, repeated executions, counter wraparound, 96 live DFBs, arithmetic with 66 allocated DFBs, matmul, reductions, residual, MLP, attention, and expert merge |
| External calls and lifecycle boundaries | 20 Blackhole device cases across BF16/FP32 and DRAM/TTNN L1, including repeated selected reset, reset-all, reconfiguration, live state preservation, payload reuse, and reset of allocation index 65 |
| External C++ compute | [Elementwise](../../test/python/test_external_dfb_reuse.py) passes 98 Blackhole BF16/FP32 cases, including a 70-DFB composition. [Block matmul](../../test/python/test_external_matmul.py) passes 148 Blackhole cases: 118 BF16/FP32 cases plus 30 BF16 by BFP4_B/BFP8_B cases across Metal DFB and compiler-managed SRAM storage, including DRAM, interleaved SRAM, and three sharded tensor layouts. |
| Tensor-backed storage | 46 Blackhole BF16/FP32 device cases cover compiler-owned and tensor-backed storage, height/width/block sharding, shard orientation, byte offsets, replacement, and repeated execution. |
| Allocation groups | Four Blackhole BF16/FP32 device cases cover shared-state handoff and different member capacities. |
| Local PipeNet | 46 Blackhole BF16/FP32 device cases cover DRAM/SRAM tensors, transfer protocols, reset and reconfiguration, repeated invocation, typed external calls, and receiver indices above the Metal descriptor limit; Wormhole support is compile-only. |
| Allocation | An independent oracle checks 5,184 four-region cases and 160 larger cases; default placement has 99.69% aggregate efficiency and 71.42% worst-case efficiency against the proven minimum. Another 2,432 compile-only placements cover both target alignments, reuse modes, and all strategies. Eight contract cases reject invalid inputs and results. Forty-eight Blackhole device cases cover BF16/FP32, DRAM/SRAM, repeated calls, and 96 simultaneous or reusable DFBs. |
| Wormhole | N150 device correctness for 424 sub-tile cases and 12 native/external 70-DFB compositions; local PipeNet remains compile-only, and reset/reconfiguration is rejected. |
| Runtime placement and resources | Runtime-unit evidence for one-device and device-domain descriptors, replicated mesh placement, lockstep arena binding, external fabric bindings, PipeNet resource composition, resource lifetimes, and program hashes; 18 Blackhole device-correctness cases for typed external calls with semaphores, runtime arguments, defines, repeated invocations, BF16/FP32, DRAM/SRAM, generic/specialized kernels, and both memory models |
| Invalid contracts | Compiler diagnostics for malformed metadata, unsupported transactions and tile forms, unknown external effects, numeric external DFB indices, storage ownership, and budget overflow |

## Extensions

- Per-node arena layouts require node-specific allocation metadata and ownership. Multicast receivers additionally require a shared payload address.
- Cross-operation reuse of PipeNet scratch requires enforced completion through destination consumption.
- Row-major operations require matching geometry, stride, and capacity rules in the address-based interface.
- Wormhole reset and reconfiguration require a target synchronization protocol validated on device.

The intended dependency order after generated fabric support is:

1. Qualify additional external C++ kernels against the typed descriptor interface. Extend the target interface only for operations whose address, geometry, or completion requirements it does not yet express.
2. Add row-major metadata, partial-block and general contiguous multi-block transactions, and the corresponding address, stride, capacity, and wrap rules.
3. Add Wormhole reset and reconfiguration after defining and device-qualifying a Wormhole synchronization protocol behind the existing target interface.
4. Qualify complete model layers, then measure device cycles, arena high-water usage, initialization cost, compile time, and generated code size against `metal-cb`.

Each extension must preserve the fail-before-mutation rule, architecture isolation, explicit ownership, and compiler-managed descriptor independence.
