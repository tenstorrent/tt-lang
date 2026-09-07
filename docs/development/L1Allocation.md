# Compiler-Managed L1 Allocation

## Purpose

TT-Lang normally assigns each logical dataflow buffer (DFB) a TT-Metal DFB descriptor. Wormhole B0 provides 32 descriptor indices and Blackhole provides 64. A program can therefore exhaust descriptor indices while sufficient L1 storage remains.

`--ttl-memory-model=compiler-l1` replaces descriptor-indexed storage with compiler-assigned L1 byte ranges. The Python DFB API and its producer/consumer semantics remain unchanged. `metal-cb` remains the default.

| Property | `metal-cb` | `compiler-l1` |
| --- | --- | --- |
| Allocated identity | TT-Metal DFB index | Logical DFB index and compiler storage owner |
| Storage address | TT-Metal descriptor | Compiler arena or tensor base plus byte offset |
| Capacity limit | L1 capacity and 32 or 64 descriptor indices | L1 capacity, control records, and alignment |
| Payload reuse | Requires the Metal descriptor and backing-storage contracts | Requires noninterfering completed lifetimes or an explicit validated allocation group |
| Producer/consumer state | TT-Metal DFB interface state | Two 32-bit page-sequence counters per storage owner |
| Tensor-backed storage | Installed through a TT-Metal descriptor | Addressed directly through the tensor runtime argument |
| Allocation groups | Reuse a physical descriptor and its storage contract | Share one validated storage owner and control record |
| Reset and reconfiguration | Blackhole TT-Metal interface reset and runtime descriptor reconfiguration | Blackhole address-based state reset with compiler-fixed geometry |
| External C++ DFB access | Numeric index or typed descriptor bound to a TT-Metal DFB | Typed descriptor bound to compiler-assigned storage |

Shared terminology is defined in the [TT-Lang specification glossary](../sphinx/specs/TTLangSpecification.md#appendix-a-glossary). The DFB protocol and lifecycle rules are defined in [DFB Management](DFBManagement.md).

## Allocation Model

One compiler-managed arena exists on each participating worker core for each invocation of a compiled Python `ttl.operation`. Every arena uses the same relative layout. Kernels receive the core-local arena base as one common runtime argument, so the argument count does not depend on the number of logical DFBs.

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

Packed-format metadata is included in `P`. An allocation group reserves the largest payload extent required by any member. Tensor-backed storage uses the tensor's node-local L1 address and adds no payload bytes to the arena. Its control record remains in the arena. The complete arena size is the maximum of the control section end and every compiler-owned payload end. Empty programs allocate no arena.

## Conflict Analysis

Allocation consumes the existing logical-identity, allocation-group, and completion-aware lifetime analyses. The compiler validates every allocation group and builds the complete conflict relation before changing IR. Unknown launch domains, unproved completion, concurrent lifetimes, and incompatible storage ownership remain conflicts.

The shared storage conflict analysis accepts an explicit storage mode. Metal storage includes conflicts caused by runtime descriptor installation and Metal-managed backing changes. Compiler-managed storage excludes those conflicts because each logical DFB retains fixed compile-time geometry and each validated storage owner has a control record. This distinction permits byte reuse across a reconfiguration boundary after the prior lifecycle ends while preserving DFBs that remain live across the boundary.

Validated allocation-group members are collapsed into one storage owner. The owner conflicts with another owner if any member pair conflicts. This preserves all lifecycle conflicts while allowing the explicit ownership transfer represented by the group. The existing group validator proves ordering, capacity, cursor continuity, and storage compatibility; the compiler-L1 allocator does not duplicate or weaken those checks.

Tensor-backed DFBs require an exact, non-empty launch-node domain. On a shared launch node, partial byte-range overlap is rejected because it does not represent a complete storage ownership transfer. Identical byte ranges are permitted for the same storage owner or for nonconflicting lifetimes. Disjoint ranges do not alias.

This design reuses one lifetime model for both memory backends. The allocator cannot serialize operations or remove a conflict to make a program fit.

```text
buildStorageConflicts(lifetimes, storageMode):
    conflicts = empty graph
    for each unordered pair (left, right):
        for each worker core where both may be active:
            if the core association or completion order is unknown:
                add conflict(left, right)
            else if neither lifetime completes before the other begins:
                add conflict(left, right)

        if storageMode is metal-cb and either descriptor installation or backing-storage ownership can overlap:
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

Possible launch domains use the same rules as exact domains and remain conservative. The compiler authorizes overlap only when every common worker core has a proven completion order.

## Placement Interface and Algorithms

Placement is independent of MLIR and target-specific code. Storage-owner construction produces one allocator region for each owner with a compiler-owned payload. Tensor-backed owners consume control records but add no allocator regions. The immutable allocation problem contains each payload extent, a symmetric conflict matrix, target alignment, the payload base after all control records, and the L1 budget. An allocator returns one byte offset per allocator region and the payload high-water mark.

Every allocator result passes the same validation before IR mutation. Validation requires the correct offset count, target alignment, offsets at or above the payload base, intervals within the L1 budget, disjoint intervals for every conflict, and an exact payload high-water mark. Allocation policy cannot weaken these invariants. The caller maps the returned offsets to storage owners and computes the arena size as the maximum of the control-section end and the payload high-water mark.

### C++ Allocator Contract

The compiler-internal interface is:

```cpp
namespace mlir::tt::ttl {

struct CompilerL1AllocationProblem {
  llvm::SmallVector<uint64_t> regionBytes;
  llvm::SmallVector<llvm::BitVector> conflicts;
  uint64_t alignmentBytes;
  uint64_t payloadBaseOffset;
  uint64_t budgetBytes;
};

struct CompilerL1AllocationSolution {
  llvm::SmallVector<uint64_t> offsets;
  uint64_t arenaBytes;
};

class CompilerL1Allocator {
public:
  virtual ~CompilerL1Allocator() = default;
  virtual llvm::StringRef getName() const = 0;

private:
  friend FailureOr<CompilerL1AllocationSolution>
  solveCompilerL1Allocation(const CompilerL1Allocator &allocator,
                            const CompilerL1AllocationProblem &problem,
                            std::optional<unsigned> &failureRegionIndex,
                            std::string &failureReason);

  virtual FailureOr<CompilerL1AllocationSolution>
  allocate(const CompilerL1AllocationProblem &problem,
           std::string &failureReason) const = 0;
};

FailureOr<std::unique_ptr<CompilerL1Allocator>>
createCompilerL1Allocator(llvm::StringRef name, std::string &failureReason);

FailureOr<CompilerL1AllocationSolution> solveCompilerL1Allocation(
    const CompilerL1Allocator &allocator,
    const CompilerL1AllocationProblem &problem,
    std::optional<unsigned> &failureRegionIndex,
    std::string &failureReason);

} // namespace mlir::tt::ttl
```

`regionBytes[i]` is the nonzero, aligned extent of allocator region `i`. The caller retains the mapping from allocator-region indices to compiler-owned storage-owner indices. `conflicts` is a square, symmetric bit matrix with a clear diagonal. `payloadBaseOffset` is aligned and does not exceed `budgetBytes`. The problem is immutable after construction. Region order defines deterministic equal-size ordering.

`solveCompilerL1Allocation` is the only caller of the private strategy method. It validates the problem, invokes the selected strategy, and validates the solution. On success, `offsets` has one entry per allocator region and `arenaBytes` is the exact maximum payload end, or zero when no allocator regions exist. On failure, `failureReason` contains diagnostic text and `failureRegionIndex` identifies an allocator region only when the error applies to one region. The allocator layer does not emit diagnostics or modify IR.

`createCompilerL1Allocator` maps stable compiler-option names to implementations. A new implementation derives from `CompilerL1Allocator`, implements `getName()` and `allocate()`, and registers its name in the factory. It cannot change conflict construction or bypass common validation.

| Strategy | Gap selection | Use |
| --- | --- | --- |
| `first-fit-decreasing` | Lowest aligned legal offset | Default deterministic low-address placement. |
| `best-fit-decreasing` | Finite legal gap with the least unused space; lower offset resolves ties | Reduces fragmentation when differently sized lifetimes leave reusable gaps. |

Both strategies place larger regions first because large extents fit in fewer gaps. Storage-owner order resolves equal-size ties. Both are greedy heuristics and can produce different arena sizes; neither proves optimality.

```text
allocateDecreasing(problem, gapSelection):
    placementOrder = stableSort(problem.regions, decreasing extent)
    placed = empty list

    for region in placementOrder:
        blockers = placed regions that conflict with region
        blockers = sort(blockers, increasing payload start)
        offset = selectOffset(problem, region, blockers, gapSelection)
        record offset for region
        append region to placed

    arenaBytes = maximum payload end, or zero for an empty problem
    return offsets and arenaBytes
```

First-fit selects an offset as follows:

```text
selectFirstFit(problem, region, blockers):
    candidate = problem.payloadBase
    for blocker in blockers:
        if [candidate, candidate + region.extent) ends before blocker:
            return candidate
        if candidate lies before blocker.end:
            candidate = roundUp(blocker.end, problem.alignment)
    return candidate
```

Best-fit evaluates every finite gap before the unbounded space after the final blocker:

```text
selectBestFit(problem, region, blockers):
    candidate = problem.payloadBase
    best = none
    for blocker in blockers:
        if [candidate, candidate + region.extent) ends before blocker:
            unused = blocker.start - candidate - region.extent
            best = minimum(best, (unused, candidate))
        if candidate lies before blocker.end:
            candidate = roundUp(blocker.end, problem.alignment)
    return best.offset if best exists, otherwise candidate
```

For each new region, either selection scans gaps between sorted blockers and advances beyond every blocker that intersects the current candidate. The selected interval therefore overlaps no conflicting interval. Applying this argument in placement order proves disjoint storage for every conflict edge. All other overlap is authorized by the lifetime analysis.

For `P` compiler-owned storage owners, placement takes `O(P^2 log P)` time after conflict construction and uses `O(P)` placement storage. Conflict adjacency for `N` logical DFBs uses `O(N^2)` bits. A budget failure reports that the selected strategy failed; it does not claim that no feasible placement exists.

The allocator interface contains no MLIR operations, DFB identities, architecture identities, tensor identities, or target branches. It receives normalized alignment and budget values through the allocation problem. Adding a strategy requires an implementation of the placement interface and a stable factory name. Conflict construction, storage-owner mapping, target queries, validation, metadata emission, and runtime allocation remain unchanged.

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

Wormhole continues to support ordinary compiler-managed allocation, transfer, and compute. Synchronized reset and reconfiguration remain Blackhole-only because their current LLK protocol depends on Blackhole processor synchronization behavior. Wormhole compilation rejects those operations with a target-specific diagnostic before allocation.

## External C++ Interface

`ttl.dfb_descriptor(dfb)` lowers to a C++ template type containing page size, pages per block, block count, shared storage capacity, state offset, payload offset, and an optional tensor common-argument index. Its `bind()` method obtains the state address from the arena. The payload address comes from either the arena or the tensor's existing common runtime argument. External functions therefore require no Metal DFB index and no additional runtime argument per DFB.

```text
bind(descriptor):
    stateAddress = target.arenaBase() + descriptor.stateOffset
    if descriptor has tensor backing:
        payloadAddress = target.commonArg(descriptor.tensorArg) + descriptor.payloadOffset
    else:
        payloadAddress = stateAddress + descriptor.payloadOffset
    return AddressDFB(stateAddress, payloadAddress, descriptor.geometry, descriptor.storageCapacity)
```

External calls that access DFBs must provide explicit `DFBEffect` entries. These effects participate in lifetime and conflict analysis. Unknown DFB access and numeric `dfb_index` template arguments are rejected for compiler-managed storage.

## Target Interfaces

Common allocation and lowering contain no architecture branches. `compiler_l1_target.h` provides arena-base access, L1 loads and stores, producer/consumer completion, and processor ownership. `compiler_l1_compute_target.h` provides LLK address conversion, format configuration, and address-based compute operations. Wormhole and Blackhole differences remain inside these target interfaces.

## Runtime Arena

The runtime allocates the control records and compiler-owned payloads as a row-major, height-sharded TTNN L1 tensor with one equal-length row per participating worker core. Height sharding directly represents one arena row per core. Width sharding provides no capacity benefit, and block sharding introduces an unused partition dimension. Tensor-backed payloads retain their existing height-, width-, or block-sharded TTNN allocations. A mesh arena uses TT-Metal's lockstep allocation, which assigns the same L1 address on every selected device. Device-domain descriptors therefore combine device-specific logical coordinates with one common arena base.

The arena is passed as an auxiliary `generic_op` input so TTNN retains it through device execution while preserving the user output position. Arena and synchronization scratch are zero-initialized. Declarative runtime resources compose with the arena: semaphore descriptors, per-kernel runtime arguments, compile-time defines, external fabric bindings, and their lifetime owners retain their existing validation and program-hash contracts. Runtime resource caching includes the allocation metadata and reset count, so incompatible layouts do not share resources.

Uniform allocation reserves the largest required arena on every participating core. This can waste capacity when activity is sparse. Per-core layouts require target-independent per-node allocation metadata and are follow-on work.

## Memory Utilization

Storage efficiency comes from six decisions:

1. Completion-aware conflicts permit payload overlap across sequential lifetimes, formats, and reconfiguration epochs.
2. Both allocation strategies search aligned gaps instead of using a monotonic offset.
3. Payloads are ordered by decreasing size to reduce fragmentation from early small placements.
4. Allocation groups share one payload envelope and one control record after the shared validator proves ownership transfer.
5. Tensor-backed payloads remain in their existing L1 tensors and consume no duplicate arena payload storage.
6. The arena uses one runtime argument, independent of logical DFB count.

The fixed control cost is `roundUp(8 * S, A)` for `S` storage owners. For 96 ungrouped one-page BF16 DFBs, simultaneous lifetimes require 196,608 payload bytes and 768 control bytes before final arena alignment. If all 96 lifetimes are sequential, they reuse one 2,048-byte payload range and retain 768 control bytes. The control records are then 27% of the 2,816 bytes before final alignment. A validated allocation group reduces both payload envelopes and control records because its members explicitly transfer ownership. Further reduction would require inferring state ownership transfer without an allocation group or packing independently written producer and consumer state, which would weaken the ownership contract or require atomic updates.

Monotonic allocation with explicit execution-phase overlays was considered. It cannot reuse an aligned gap between active allocations and requires explicit phase boundaries. TT-Lang instead uses its completion-aware conflict graph and searches reusable gaps, which permits overlap within a phase and across different extents. Best-fit addresses fragmentation without exponential search. An exact or bounded-search strategy can use the same allocator interface if measurements justify its compile-time cost.

## Implemented Contract

- One uniform worker-core arena layout at a lockstep address across the selected devices.
- Compiler-owned static payloads and existing height-, width-, or block-sharded tensor-backed payloads.
- Validated allocation groups with one shared state record and the largest required compiler-owned payload envelope.
- One-block transactions and complete-capacity tensor publication or consumption, with positive capacity below `2^31` pages.
- Consumer-owned replacement writes into the acquired read window without changing occupancy or sequence state.
- Full 32x32 BF16 and FP32 tiles for address-based compute.
- Address-based tensor transfer, elementwise compute, matmul, reductions, broadcast, transpose, and selected activation operations covered by the implementation tests.
- Typed external C++ calls with explicit DFB effects and either compiler-owned or tensor-backed payloads.
- Device-domain and mesh program placement with declarative external runtime resources.
- Blackhole selected reset, reset-all, and reconfiguration.
- Wormhole allocation, transfer, compute, and external descriptors without reset or reconfiguration.

PipeNet transfers and computed-address DFBs are outside this contract and are rejected before runtime-resource construction. The compiler does not fall back to Metal descriptors.

## Validation

| Scenario | Evidence |
| --- | --- |
| Blackhole transfer and compute | Device correctness across BF16/FP32, DRAM/L1 tensors, repeated executions, counter wraparound, 96 live DFBs, arithmetic with 66 allocated DFBs, matmul, reductions, residual, MLP, attention, and expert merge |
| Tensor-backed storage | 46 Blackhole device-correctness cases across BF16/FP32, compiler-owned scratch and tensor-backed storage, height/width/block sharding, row/column shard orientation, nonzero byte offsets, complete-capacity publication, replacement, and repeated execution; compile-only metadata checks cover both allocator strategies |
| Allocation groups | Four compiler-L1 Blackhole device-correctness cases across BF16/FP32 and DRAM/L1 tensors for repeated shared-state handoff with different member capacities; compile-only checks cover both allocator strategies, tensor-backed ownership, rejection with reuse disabled, and tensor byte-range alias diagnostics |
| External calls and lifecycle boundaries | 20 Blackhole device cases across BF16/FP32 and DRAM/L1, including repeated selected reset, reset-all, reconfiguration, live state preservation, payload reuse, and reset of allocation index 65 |
| Allocation | 20,888 compile-only generated placements covering both strategies, conflicts, alignment, reuse enabled and disabled, determinism, and exact budget boundaries; a focused fragmented graph verifies distinct strategy results |
| Wormhole | Compile-only allocation, typed external descriptor, and UNPACK/MATH/PACK target compilation; negative reset and reconfiguration diagnostics |
| Runtime placement and resources | Runtime-unit evidence for one-device and device-domain descriptors, replicated mesh placement, lockstep arena binding, external fabric bindings, resource lifetimes, program hashes, and retained PipeNet rejection; 18 Blackhole device-correctness cases for typed external calls with semaphores, runtime arguments, defines, repeated invocations, BF16/FP32, DRAM/L1, generic/specialized kernels, and both memory models |
| Invalid contracts | Compiler diagnostics for malformed metadata, unsupported transactions and tile forms, unknown external effects, numeric external DFB indices, storage ownership, and budget overflow |

Relevant tests are [transfer and allocator device tests](../../test/python/test_compiler_l1.py), [compute device tests](../../test/python/test_compiler_l1_compute.py), [lifecycle and external-call device tests](../../test/python/test_compiler_l1_lifecycle.py), [runtime placement tests](../../test/python/test_kernel_runner.py), [external runtime-resource device tests](../../test/python/test_operation_runtime_resources.py), and [generated allocator stress tests](../../test/ttlang/Dialect/TTL/Transforms/compiler_l1_stress.py).

## Follow-on PRs

The intended dependency order after tensor-backed storage and allocation groups is:

1. Add PipeNet and computed-address transfers. Represent compiler-managed endpoint addresses in the immutable transport plan, validate the complete graph before allocation, and extend lifetime completion through remote transfer and destination consumption.
2. Add multi-device fabric execution. Resolve source and destination device domains, allocate per-device resources through the runtime placement interface, and verify inter-device completion before storage reuse.
3. Qualify representative external C++ kernels against the typed descriptor interface and add common adapters for required address, geometry, and completion operations.
4. Add sub-tile and row-major metadata, partial-block and general contiguous multi-block transactions, and the corresponding address, stride, capacity, and wrap rules.
5. Add Wormhole reset and reconfiguration after defining and device-qualifying a Wormhole synchronization protocol behind the existing target interface.
6. Qualify complete model layers, then measure device cycles, arena high-water usage, initialization cost, compile time, and generated code size against `metal-cb`.

Each extension must preserve the fail-before-mutation rule, architecture isolation, explicit ownership, and compiler-managed descriptor independence.
