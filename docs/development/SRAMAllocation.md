# Compiler-Managed SRAM Allocation

## Purpose

TT-Lang normally assigns each logical dataflow buffer (DFB) a TT-Metal DFB descriptor. Wormhole B0 provides 32 descriptor indices and Blackhole provides 64. A program can therefore exhaust descriptor indices while sufficient SRAM storage remains.

`--ttl-memory-model=compiler-l1` replaces descriptor-indexed storage with compiler-assigned SRAM byte ranges. The Python DFB API and its producer/consumer semantics remain unchanged. `metal-cb` remains the default.

| Property | `metal-cb` | `compiler-l1` |
| --- | --- | --- |
| Allocated identity | TT-Metal DFB index | Logical DFB index and compiler storage owner |
| Storage address | TT-Metal descriptor | Compiler arena or tensor base plus byte offset |
| Capacity limit | SRAM capacity and 32 or 64 descriptor indices | SRAM capacity, control records, and alignment |
| Payload reuse | Requires the Metal descriptor and backing-storage contracts | Requires noninterfering completed lifetimes or an explicit validated allocation group |
| Producer/consumer state | TT-Metal DFB interface state | Two 32-bit page-sequence counters per storage owner |
| Tensor-backed storage | Installed through a TT-Metal descriptor | Addressed directly through the tensor runtime argument |
| Allocation groups | Reuse a physical descriptor and its storage contract | Share one validated storage owner and control record |
| Reset and reconfiguration | Blackhole TT-Metal interface reset and runtime descriptor reconfiguration | Blackhole address-based state reset; page size, pages per block, block count, and storage capacity remain unchanged |
| External C++ DFB access | Numeric index or typed descriptor bound to a TT-Metal DFB | Typed descriptor bound to compiler-assigned storage |
| PipeNet receivers | TT-Metal descriptor with computed or receiver-published addressing | Compiler arena or tensor-backed computed address for intra-device and generated inter-device transfers; no TT-Metal DFB descriptor |

Shared terminology is defined in the [TT-Lang specification glossary](../sphinx/specs/TTLangSpecification.md#appendix-a-glossary). The DFB protocol and lifecycle rules are defined in [DFB Management](DFBManagement.md).

## Allocation Model

One compiler-managed arena exists on each participating worker core for each invocation of a compiled Python `ttl.operation`. `--ttl-sram-allocation-mode=uniform` uses the same relative layout on every core. `per-core` allocates independent layouts, except that multicast receivers share one layout and base address. Kernels receive the core-local arena base as one common runtime argument, so the argument count does not depend on the number of logical DFBs.

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

Packed-format metadata is included in `P`. An allocation group reserves the largest payload extent required by any member. Tensor-backed storage uses the tensor's node-local SRAM address and adds no payload bytes to the arena. Its control record remains in the arena. The complete arena size is the maximum of the control section end and every compiler-owned payload end. Empty programs allocate no arena.

## Conflict Analysis

Allocation consumes the existing logical-identity, allocation-group, and completion-aware lifetime analyses. The compiler validates every allocation group and builds the complete conflict relation before changing IR. Unknown launch domains, unproved completion, concurrent lifetimes, and incompatible storage ownership remain conflicts.

The shared storage conflict analysis accepts an explicit storage mode. Metal storage includes conflicts caused by runtime descriptor installation and Metal-managed backing changes. Compiler-managed storage excludes those conflicts because each logical DFB's page size, pages per block, block count, and storage capacity remain constant during execution, and each validated storage owner has a control record. This distinction permits byte reuse across a reconfiguration boundary after the prior lifecycle ends while preserving DFBs that remain live across the boundary.

Validated allocation-group members are collapsed into one storage owner. The owner conflicts with another owner if any member pair conflicts. This preserves all lifecycle conflicts while allowing the explicit ownership transfer represented by the group. The existing group validator proves ordering, capacity, cursor continuity, and storage compatibility; the compiler-managed SRAM allocator does not duplicate or weaken those checks.

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

Placement is independent of MLIR and target-specific code. Storage-owner construction produces one allocator region for each owner with a compiler-owned payload. Tensor-backed owners consume control records but add no allocator regions. The immutable allocation problem contains each payload extent, a symmetric conflict matrix, target alignment, the payload base after all control records, and the SRAM budget. An allocator returns one byte offset per allocator region and the payload high-water mark.

Every allocator result passes the same validation before IR mutation. Validation requires the correct offset count, target alignment, offsets at or above the payload base, intervals within the SRAM budget, disjoint intervals for every conflict, and an exact payload high-water mark. Allocation policy cannot weaken these invariants. The domain allocation entry point maps offsets to storage owners and retains the control prefix when computing the arena size.

### C++ Allocator Contract

The compiler-internal interface is:

```cpp
namespace mlir::tt::ttl {

struct SRAMAllocationProblem {
  llvm::SmallVector<uint64_t> regionBytes;
  InterferenceGraph conflicts{0};
  uint64_t alignmentBytes;
  uint64_t payloadBaseOffset;
  uint64_t budgetBytes;
};

struct SRAMAllocatorOptions {
  uint64_t exactSearchLimit;
};

struct SRAMAllocationSolution {
  llvm::SmallVector<uint64_t> offsets;
  uint64_t arenaBytes;
};

struct SRAMAllocationDomainProblem {
  SRAMAllocationProblem allocation;
  llvm::SmallVector<unsigned> storageIndices;
};

struct SRAMStoragePlacement {
  unsigned storageIndex;
  uint64_t offset;
};

struct SRAMAllocationDomainSolution {
  llvm::SmallVector<SRAMStoragePlacement> placements;
  uint64_t arenaBytes;
};

struct SRAMAllocationDomainFailure {
  unsigned domainIndex;
  std::optional<unsigned> storageIndex;
  std::string reason;
};

class SRAMAllocator {
public:
  virtual ~SRAMAllocator() = default;
  virtual llvm::StringRef getName() const = 0;

  FailureOr<SRAMAllocationSolution>
  allocate(const SRAMAllocationProblem &problem,
           std::optional<unsigned> &failureRegionIndex,
           std::string &failureReason) const;

  FailureOr<llvm::SmallVector<SRAMAllocationDomainSolution>>
  allocateDomains(llvm::ArrayRef<SRAMAllocationDomainProblem> domains,
                  SRAMAllocationDomainFailure &failureDetail) const;

private:
  virtual FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &problem,
               std::string &failureReason) const = 0;
};

FailureOr<std::unique_ptr<SRAMAllocator>>
createSRAMAllocator(llvm::StringRef name, const SRAMAllocatorOptions &options,
                  std::string &failureReason);

} // namespace mlir::tt::ttl
```

`regionBytes[i]` is the nonzero, aligned extent of allocator region `i`. The caller retains the mapping from allocator-region indices to compiler-owned storage-owner indices. `conflicts` is the shared undirected resource-interference graph. `payloadBaseOffset` is aligned and does not exceed `budgetBytes`. The problem is immutable after construction. Region order supplies the final deterministic tie-break.

`SRAMAllocator::allocate` is the public, nonvirtual entry point. It validates the problem, invokes the private strategy method, and validates the solution. This structure keeps strategy selection replaceable while enforcing one correctness contract. On success, `offsets` has one entry per allocator region and `arenaBytes` is the exact maximum payload end, or zero when no allocator regions exist. On failure, `failureReason` contains diagnostic text and `failureRegionIndex` identifies an allocator region only when the error applies to one region. The allocator layer does not emit diagnostics or modify IR.

`allocateDomains` applies the same strategy and validation to independently addressable layouts. Domain order is stable; `storageIndices` maps each domain-local region to a caller-owned storage identity. Identities must be unique within a domain and may recur across domains. Each domain supplies its own conflicts, alignment, control prefix, and budget. Core membership and the proof that domain bindings are disjoint belong to the caller, not the placement strategy. The compiler supplies one domain in uniform mode and one domain per independent core or multicast receiver group in per-core mode.

All domain inputs are validated before any strategy executes. Failure returns no partial placement; `failureDetail` identifies the domain and, when available, the storage owner. A control-only domain retains `payloadBaseOffset` as its arena size. Exact search has the configured work limit separately for each domain. Independent exact minima minimize total reservation for a fixed domain partition and positive replication counts; this does not optimize the partition itself or account for host allocation granularity.

```text
allocateDomains(domains):
    validate every domain problem and storage-identity mapping
    for each domain in input order:
        placement = selected strategy(domain.problem)
        validate placement using the common allocator rules
        map local region offsets to storage identities
        arenaBytes = max(control-prefix end, payload high-water mark)
    return all domain placements, or failure without a partial result
```

`SRAMAllocatorOptions` contains limits that affect strategy execution but do not change the allocation problem. `exactSearchLimit` bounds the exact strategy's combined subset-sum candidates and partial placements separately for each allocation domain. With `D` domains, total search work can reach `D` times the configured limit. `createSRAMAllocator` maps stable compiler-option names to implementations and supplies these options. `getName()` identifies the implementation in validation diagnostics. A new implementation derives from `SRAMAllocator`, implements `getName()` and `allocateImpl()`, and registers its name in the factory. It cannot change conflict construction or bypass common validation.

The common interface and validation are in `SRAMAllocator.h` and `SRAMAllocator.cpp`. `SRAMAllocator_Greedy.cpp` shares ordering and gap placement across the three greedy strategies; `SRAMAllocator_Exact.cpp` contains exact search. Private declarations in `SRAMAllocator_Internal.h` connect the factory and allow exact search to reuse greedy upper bounds.

| Strategy | Placement rule | Result |
| --- | --- | --- |
| `multi-order-decreasing` (default) | Run first-fit decreasing with stable and degree-aware equal-size ordering; retain the smaller arena and the stable layout on ties. | Deterministic placement no larger than first-fit decreasing. |
| `first-fit-decreasing` | Place decreasing extents at the lowest aligned legal offset. | Deterministic feasible placement. |
| `best-fit-decreasing` | Place decreasing extents in the finite legal gap with the least unused space; lower offset resolves ties. | Deterministic feasible placement that can reduce fragmentation. |
| `exact` | Search aligned subset-sum offsets with branch-and-bound. | Proven minimum arena, or a precise inconclusive or infeasible diagnostic. |

The decreasing strategies place larger regions first because large extents fit in fewer gaps. Storage-owner order resolves equal-size ties in the individual first-fit and best-fit strategies. They are greedy heuristics and can produce different arena sizes.

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

The default uses two decreasing-size orders. The second resolves equal extents by decreasing conflict degree (the number of conflicting regions), placing more constrained regions first. It selects the smaller arena; equal arena sizes retain the original offsets. Since the original placement remains a candidate, the result cannot increase its arena size. Both candidates use the same gap-placement algorithm and the selected result passes common validation. Two placements preserve the `O(P^2 log P)` bound with additional constant-factor work.

```text
allocateMultiOrder(problem):
    stable = firstFit(problem, order by decreasing extent then owner order)
    degreeAware = firstFit(problem, order by decreasing extent then decreasing conflict degree then owner order)
    if degreeAware succeeds and (stable fails or degreeAware.arenaBytes < stable.arenaBytes):
        return degreeAware
    return stable
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

For each new region, either decreasing strategy scans gaps between sorted blockers and advances beyond every blocker that intersects the current candidate. The selected interval therefore overlaps no conflicting interval. Applying this argument in placement order proves disjoint storage for every conflict edge. All other overlap is authorized by the lifetime analysis.

For `P` compiler-owned storage owners, decreasing placement takes `O(P^2 log P)` time after conflict construction and uses `O(P)` placement storage. Conflict adjacency for `N` logical DFBs uses `O(N^2)` bits. A decreasing-strategy budget failure reports only that the selected strategy failed.

Exact placement first computes the better decreasing-strategy result as a feasible upper bound. It partitions the conflict graph into connected components because components have no disjointness constraints and can use the same addresses. A decreasing-weight clique in each component supplies a proven lower bound. Each component is then searched independently, and the component minima are overlaid at the common payload base.

Only subset sums of aligned region extents need consideration. In an optimal placement chosen to minimize the sum of offsets among all minimum-arena placements, every region at a nonzero offset must touch the end of a conflicting region below it. Otherwise it could move left while preserving validity. Repeating this argument reaches offset zero, so every selected offset is a sum of distinct region extents.

```text
allocateExact(problem, workLimit):
    incumbent = better of first-fit-decreasing and best-fit-decreasing
    components = connectedComponents(problem.conflicts)
    resultOffsets = empty vector

    for component in components:
        lowerBound = maximum weight found by deterministic greedy cliques
        if lowerBound exceeds the available payload capacity:
            report proven infeasibility

        componentIncumbent = incumbent restricted to component, if it fits
        if componentIncumbent equals lowerBound:
            record componentIncumbent
            continue

        candidates = sorted subset sums below the incumbent, or within capacity when no incumbent fits
        search assignments in decreasing conflict-degree and extent order
        reject overlap with each assigned conflicting region
        reject assignments whose high-water mark cannot improve the incumbent

        if the work limit is reached:
            report limit exhaustion and include any feasible incumbent size
            fail without returning an unproved placement
        if no feasible assignment exists after exhaustive search:
            report proven infeasibility
        record the best component assignment; exhaustion proves it minimal

    overlay component assignments at the payload base
    return the maximum component end as the proven minimum arena size
```

The proof above establishes completeness of the candidate offsets. Exhaustive enumeration below the incumbent establishes that no smaller placement exists. Component overlay is valid because different components have no conflict edges. The search is exponential in the worst case. `--ttl-l1-exact-allocation-search-limit` bounds generated subset sums and visited partial assignments per domain; reaching the limit fails compilation rather than returning an unproved result.

The allocator interface contains no MLIR operations, DFB identities, architecture identities, tensor identities, or target branches. It receives normalized alignment and budget values through the allocation problem. Adding a strategy requires an implementation of the placement interface and a stable factory name. Conflict construction, storage-owner mapping, target queries, validation, metadata emission, and runtime allocation remain unchanged.

## Allocation Domains

Per-core mode starts with one domain per worker core in the exact launch grid. A multicast carries one destination address, so all receivers of each multicast belong to one domain. Overlapping receiver sets merge transitively using LLVM `EquivalenceClasses`. This constraint can produce larger domains than individual multicast rectangles; it preserves existing NOC transfers without changing LLKs.

```text
buildDomains(operation):
    initialize one disjoint set per worker core
    for each multicast receiver set:
        union all receiver cores
    for each resulting domain, in stable core order:
        include every storage owner that may be active on any member core
        build storage conflicts using completion ordering on member cores
        add an owner conflict if any pair of their logical DFB members conflicts
        retain the common control prefix and target alignment
    allocate and validate all domains before changing IR
```

Buffers can share storage only when the compiler proves that their lifetimes do not overlap on any core sharing the allocation layout. Overlap on cores outside that group does not prevent reuse. If activity or completion is uncertain, the compiler retains potentially needed storage and prevents reuse wherever it cannot prove that sharing is safe.

Control records remain at fixed offsets on every core, including cores without that owner's payload, so reset and allocation-group ownership retain their existing contracts. These choices bound the current savings; independent placement does not imply an optimal domain partition or minimum total device reservation.

The existing core-specialization pass creates one kernel instance per core. Finalized metadata supplies each instance's payload offsets and each computed PipeNet argument's destination DFB, core, and logical device. Transport finalization preserves those identities when it removes unused receiver arguments. The runtime validates domain membership, shared layouts, storage aliases, and receiver bindings before creating resources.

```text
bindDomains(invocation):
    validate finalized domain metadata and specialized kernel core sets
    allocate one zero-initialized SRAM tensor per domain
        singleton: use TT-Metal per-core allocation
        multiple cores: use TT-Metal lockstep allocation
    for each selected device and specialized kernel:
        bind that device/core's arena base and payload offsets
        bind computed receiver bases using destination device/core identities
    retain arena tensors through operation completion
```

Per-core host allocation requires `TT_METAL_ALLOCATOR_MODE_HYBRID=1` before opening devices.

## Allocation Report

`--ttl-sram-allocation-report` emits JSON records to stderr, each prefixed by `ttlang-sram-report: `. Reporting is disabled by default and inactive with `metal-cb`. The compiler record appears on compilation; the runtime record appears on each invocation that allocates an arena, including compiled-artifact cache hits. Reporting does not change placement or lifetime proofs.

Per-core mode emits one compiler and runtime record per allocation domain. `cores` lists domain members; compiler records also include `domain` and `allocation_mode`. For singleton per-core arenas, runtime records use the allocated tensor's `buffer_aligned_page_size()` and verify one buffer page per core; `accounting_source` identifies this as `tensor-buffer-geometry`. Lockstep domains use `buffer-report`. The installed TT-Metal buffer inventory omits per-core allocations. These counts describe backing extents, not free-space fragmentation or the largest remaining allocation.

The compiler record has `schema_version: 1` and `phase: "compiler"`. `owners` maps shared storage and control offsets to logical DFBs. `regions` includes declaration locations, core domains, and fixed tensor byte ranges. `logical_conflicts` reuses existing reason names and source evidence; these are analysis facts before allocation-group ownership is applied. `reuse_enabled: false` separately explains policy-disabled reuse. `reused_ranges` lists overlapping compiler-owned owner pairs; its byte counts are not additive when more than two owners reuse a range. `lifetimes` records known and possible core membership, completion proof status, entry locations, and entry/completion event IDs. Event IDs identify partial-order analysis events, not elapsed time or a total execution order.

| Compiler metric | Meaning |
| --- | --- |
| `arena_bytes_per_core` | Planned control prefix plus payload high-water mark. |
| `control_record_bytes`, `control_padding_bytes` | Control state and alignment padding, reported separately. |
| `payload_extent_sum_bytes` | Sum of distinct compiler-owned storage-owner extents, after allocation-group consolidation. |
| `payload_union_bytes` | Number of distinct payload addresses occupied by those extents. |
| `payload_reuse_bytes` | Extent sum minus union; excludes sharing already represented by allocation groups. |
| `payload_gap_bytes` | Payload high-water mark minus union; unused address gaps, not excess over an optimal allocation. |

The runtime record has `phase: "runtime"` and `scope: "arena-reference-device"`. It reports TTNN's `max_size_per_bank` as `reserved_bytes_per_core`, the participating `core_count`, their product as `reserved_bytes_on_reference_device`, and `reservation_padding_bytes_per_core` beyond `requested_bytes_per_core`. The requested extent is reconstructed from finalized DFB descriptors; a control-only arena can omit trailing compiler alignment padding from this request. It measures the arena reservation on the mesh reference device; it is not a mesh-wide total or total program SRAM use. Existing tensor payloads, PipeNet scratch, and external resources are outside this runtime total.

A compiler-only report can be obtained with:

```sh
ttlang-opt test/ttlang/Dialect/TTL/Transforms/compiler_l1_multi_order.mlir \
  -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 sram-allocation-report=true})' \
  > /tmp/allocated.mlir 2> /tmp/sram-report.log
```

## Requested Allocator Capabilities

The initial scope is one compiled `ttl.operation`, including its tensor-backed and compiler-owned DFBs. The requested static/dynamic distinction remains to be defined; it is not assumed to mean runtime-dependent sizes.

| Request | Implemented | Missing |
| --- | --- | --- |
| Late allocation and global minimum | One immutable problem per `ttl.operation`; optional exact minimum for each compiler-owned allocation domain. | Joint placement of tensor-backed and compiler-owned storage within that operation. Existing tensor addresses are already assigned. |
| Full lockstep, ranged lockstep, and per-core allocation | Uniform mode uses one shared layout; per-core mode uses domain-specific completion conflicts and merges multicast receivers that require equal addresses. | Explicit user-selected partitions. Multicast constraints can force larger shared domains. |
| Unified tensor and DFB allocation | Shared ownership, lifetime, allocation-group, and alias validation; tensor-backed DFBs avoid duplicate payload storage. | Shared physical placement. TTNN owns existing tensor allocations; the compiler currently owns only its arena. |
| Lifetime inspection and reuse hints | Automatic completion-aware reuse and the allocation report above. | A user-facing guidance contract that preserves asynchronous completion. |

### Planned Implementation

1. Lifetime guidance. Build on the allocation report. Placement preferences may change ordering but cannot remove conflicts. Reuse existing ownership-transfer operations for semantic lifetime boundaries; validate producer publication and consumer completion, including remote and external users.
2. Allocation-domain refinement. Accept explicit core partitions, validate multicast receiver address equality, and reuse domain-specific conflict construction. Measure whether finer domains reduce actual reservation enough to justify additional host allocations and kernel specialization.
3. Unified host placement. Describe tensor and DFB storage with common ownership, alias, lifetime, alignment, domain, and fixed/movable constraints. Preserve caller-owned addresses. Reserve the validated plan transactionally and construct tensor views over owned storage, retaining owners through completion. Reuse TTNN/TT-Metal host facilities where their contracts suffice; extend host APIs where required.
4. Late joint placement within one operation. Extend the existing immutable allocation problem and its oracle to fixed tensor intervals and domain-specific movable storage. Assign offsets only after sizes, ownership, domains, and completion conflicts are known. Minimize uniform arena size or total domain reservation subject to each core's capacity. Optimality remains relative to the supplied requirements and fixed addresses.

Reporting supplies the measurements for later work. Domain representation precedes unified placement. Lifetime guidance can proceed alongside domain work. These host/compiler extensions preserve common validation and address-based LLK interfaces.

Cross-launch persistence, cross-launch reuse, and runtime-dependent sizes are possible extensions proposed during design discussion, not requirements inferred from the original requests. Persistence retains contents across launches; reuse releases storage after completion. They require separate ownership and lifetime contracts.

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

## Compute Tile Metadata

TT-Metal compute APIs normally read the data format and tile dimensions from a DFB descriptor. Compiler-managed storage creates no such descriptor, so each generated compute operand type contains the data format, page size, tile height, and tile width. The common compute interface derives face dimensions from those values. Native compiler-managed compute accepts BF16 and FP32 tiles supported by the shared compute-target contract: 1x32, 2x32, 4x32, 8x32, 16x16, 16x32, 32x16, and 32x32 on Wormhole and Blackhole. Typed external compute descriptors additionally accept BFP4_B and BFP8_B with 32x32 tiles. The external function supplies the operation-specific mixed-format compute sequence; native operations remain restricted to formats qualified by their generated implementations.

Compute setup is shared by operands with equal formats, page sizes, tile dimensions, and direct-to-destination settings. Storage offsets and capacities remain properties of each DFB. Separating hardware properties from storage identity prevents repeated setup code from exhausting kernel instruction storage in large DFB compositions.

A generated compute kernel can use different tile dimensions during one execution. `ComputeContext` records the currently programmed input formats, page sizes, and face dimensions, together with the output format and tile dimensions. The first operation configures UNPACK, MATH, and PACK. Later operations reconfigure only state that differs. A PACK tile-dimension change requires data-format reconfiguration followed by pack initialization that preserves the existing address modifiers. This sequence follows the TT-Metal LLK contract and avoids repeating hardware configuration.

```text
configureCompute(inputA, inputB, output):
    if no compute configuration exists:
        configure UNPACK and MATH for inputA and inputB
        configure PACK for output
    else:
        for each input:
            if page size or face dimensions changed:
                reconfigure its format and face dimensions
            else if its format changed:
                reconfigure its format
        if output tile dimensions changed:
            reconfigure its format and dimensions
            initialize PACK while preserving address modifiers
        else if output format or page size changed:
            reconfigure its format
    record the active input and output configuration
```

`compiler_l1_compute.h` contains this common policy and passes derived parameters to `compiler_l1_compute_target.h`. The target header adapts those parameters to the Blackhole and Wormhole LLK signatures. Sub-tile lowering adds no architecture branch to compiler conversion or allocation.

## External C++ Interface

`ttl.dfb_descriptor(dfb)` lowers to a C++ template type containing page size, pages per block, block count, shared storage capacity, state offset, payload offset, and an optional tensor common-argument index. A compute-thread descriptor also contains the data format, tile height, tile width, and direct-to-destination choice:

```cpp
namespace ttlang::l1 {

template <uint32_t Format, uint32_t PageBytes, uint32_t TileHeight,
          uint32_t TileWidth, uint32_t PagesPerBlock, uint32_t BlockCount,
          uint32_t StorageCapacityPages, uint32_t StateOffset,
          uint32_t PayloadOffset, int32_t PayloadCommonArgIndex,
          bool DirectToDestination>
class ComputeDFBDescriptor;

} // namespace ttlang::l1
```

The template parameters are compile-time constants copied from finalized allocation and compute metadata:

| Parameter | Contract |
| --- | --- |
| `Format` | TT-Metal data-format value stored in SRAM. |
| `PageBytes` | Bytes per tile and per DFB page. It is a positive multiple of 16. |
| `TileHeight`, `TileWidth` | Compute-target tile dimensions used to configure UNPACK, MATH, and PACK. |
| `PagesPerBlock`, `BlockCount` | Transaction size and number of transactions in the logical DFB. |
| `StorageCapacityPages` | Capacity of the shared storage owner. It is at least `PagesPerBlock * BlockCount`. |
| `StateOffset` | Byte offset of the storage owner's 8-byte control record from the arena base. |
| `PayloadOffset` | Byte offset from the control-record address for arena storage, or from the tensor base for tensor-backed storage. |
| `PayloadCommonArgIndex` | Common runtime-argument index containing the tensor base; `-1` selects arena storage. |
| `DirectToDestination` | Whether FP32 input is unpacked directly to FP32 destination registers instead of conversion to TF32. |

`bind()` resolves the control-record and payload addresses from these constants. External functions therefore require no Metal DFB index and no additional runtime argument per DFB. [Compute kernel configuration](ComputeKernelConfiguration.md) defines direct-to-destination selection.

Generated C++ emits storage descriptor definitions before the external header, allowing one source adapter to bind either Metal or compiler-owned storage. Compiler-owned adapters use `ttlang::l1::target`, which contains target-specific address operations. Opaque C++ bodies do not participate in compute analysis, so the enclosing operation declares required compute configuration. [External functions](../sphinx/reference/external-functions.md#template-arguments) defines the exact C++ interface.

```text
bind(descriptor):
    stateAddress = target.arenaBase() + descriptor.stateOffset
    if descriptor has tensor backing:
        payloadAddress = target.commonArg(descriptor.tensorArg) + descriptor.payloadOffset
    else:
        payloadAddress = stateAddress + descriptor.payloadOffset
    return AddressDFB(stateAddress, payloadAddress, descriptor.pageSize,
                      descriptor.pagesPerBlock, descriptor.blockCount,
                      descriptor.storageCapacity)
```

External calls can provide explicit `DFBEffect` entries for protocol operations. These effects participate in lifetime and conflict analysis; a dependency without effects remains conservatively live until a synchronization boundary proves completion. Numeric DFB references through `ttl.get_dfb_id` or a DFB in `func_args` are rejected for compiler-managed storage.

## PipeNet Integration

Intra-device and generated inter-device PipeNets retain the existing transfer plan and synchronization protocols. Compiler-managed allocation changes how a receiver address is obtained. A compiler-owned receiver uses the arena base plus its finalized payload offset. A tensor-backed receiver uses the tensor base plus its finalized byte offset. No TT-Metal DFB descriptor is created for either case.

Generated inter-device transfers compute receiver addresses from the destination device and core. Uniform mode supplies a common lockstep arena base. Per-core mode binds the destination domain's actual allocation address and payload offset. Tensor-backed receivers retain their validated tensor byte ranges. Fabric binding independently resolves logical device coordinates to physical routing targets.

The Metal backend retains receiver publication when one physical DFB index can refer to different storage across reconfiguration epochs. Compiler-managed allocation assigns each finalized DFB index one arena or tensor base for the compiled operation, so that base remains valid for every transfer occurrence.

Producer and wait launch domains are validated for explicit DFB operations and external-call `DFBEffect` declarations. Treating both through the DFB access interface prevents an external producer from being omitted from PipeNet deadlock analysis.

Receiver addresses precede PipeNet scratch and semaphore addresses in the common runtime argument layout. The finalized `ttl.crta_indices` metadata defines the tensor-argument prefix because tensor-backed DFBs can retain tensors that are no longer function operands. The arena base remains the final compiler-managed storage argument. This ordering matches lowering and does not depend on the number of TT-Metal DFB descriptors.

```text
buildPipeRuntimeArguments(allocations, computedReceivers, tensors):
    resources = allocatePipeScratchAndSemaphores()
    arena = allocateZeroedArena(allocations.arenaBytes)

    for each receiver in computedReceivers:
        if receiver has tensor backing:
            base[receiver] = tensors[receiver.tensorIndex].base + receiver.byteOffset
        else:
            base[receiver] = arena.base + receiver.payloadOffset
        require base[receiver] fits uint32

    for each kernel:
        arguments = retainedTensorAddresses(kernel)
        append base addresses selected by kernel.computedReceivers
        append resources in finalized PipeNet order
        append arena.base
    return arguments
```

```text
bindGeneratedFabricRoutes(deviceDomain, kernels, routes):
    plans = empty map
    for each logical device in deviceDomain:
        program = buildProgramDescriptor(kernels, logicalDevice)
        plans[logicalDevice] = planFabricBindings(program, routes, logicalDevice)

    for each logical device in deviceDomain:
        apply plans[logicalDevice] to its program descriptor
    return one mesh program descriptor containing every device program
```

Planning every device before applying any binding prevents a later invalid route from leaving earlier program descriptors partially configured.

Runtime-resource cache identity includes tensor-backed receiver addresses, so a new tensor allocation cannot reuse a stale receiver base. Cache ownership includes only SRAM allocations created by the runtime; caller-owned tensor addresses remain part of the available-SRAM calculation.

The supported contract includes computed-capacity, computed receiver-post, published receiver-post, and global-counter protocols; ready-receive selection; grouped transfers; generated inter-device routes; and compiler-managed reset and reconfiguration boundaries. PipeNet lifetime operations, including remote completion, continue to participate in the existing completion-aware conflict analysis before placement.

## Target Interfaces

Common allocation and lowering contain no architecture branches. `compiler_l1_target.h` provides arena-base access, SRAM loads and stores, producer/consumer completion, and processor ownership. `compiler_l1_compute_target.h` provides LLK address conversion, format configuration, and address-based compute operations. Wormhole and Blackhole differences remain inside these target interfaces.

## Runtime Arena

The runtime represents each domain arena as a row-major, height-sharded TTNN SRAM tensor, with one row per member core. This directly represents equal-length storage within a domain. Uniform mode uses TT-Metal lockstep allocation across all selected cores and devices. Per-core mode uses independent allocation for singleton domains and lockstep allocation for multicast receiver domains; kernel descriptors bind the actual address on each selected device. Tensor-backed payloads retain their existing height-, width-, or block-sharded allocations.

The arena is passed as an auxiliary `generic_op` input so TTNN retains it through device execution while preserving the user output position. Arena and synchronization scratch are zero-initialized. Declarative runtime resources compose with the arena: semaphore descriptors, per-kernel runtime arguments, compile-time defines, external fabric bindings, and their lifetime owners retain their existing validation and program-hash contracts. Runtime resource caching includes the allocation metadata and reset count, so incompatible layouts do not share resources.

Uniform allocation reserves the largest required arena on every participating core. This can waste capacity when activity is sparse.

## Memory Utilization

Storage efficiency comes from seven decisions:

1. Completion-aware conflicts permit payload overlap across sequential lifetimes, formats, and reconfiguration epochs.
2. Both decreasing strategies search aligned gaps instead of using a monotonic offset.
3. Payloads are ordered by decreasing size to reduce fragmentation from early small placements.
4. Allocation groups share one payload envelope and one control record after the shared validator proves ownership transfer.
5. Tensor-backed payloads remain in their existing SRAM tensors and consume no duplicate arena payload storage.
6. The arena uses one runtime argument, independent of logical DFB count.
7. Exact placement can prove the minimum high-water mark when greedy placement leaves fragmentation.

The fixed control cost is `roundUp(8 * S, A)` for `S` storage owners. For 96 ungrouped one-page BF16 DFBs, simultaneous lifetimes require 196,608 payload bytes and 768 control bytes before final arena alignment. If all 96 lifetimes are sequential, they reuse one 2,048-byte payload range and retain 768 control bytes. The control records are then 27% of the 2,816 bytes before final alignment. A validated allocation group reduces both payload envelopes and control records because its members explicitly transfer ownership. Further reduction would require inferring state ownership transfer without an allocation group or packing independently written producer and consumer state, which would weaken the ownership contract or require atomic updates.

Monotonic allocation with explicit execution-phase overlays was considered. It cannot reuse an aligned gap between active allocations and requires explicit phase boundaries. TT-Lang instead uses its completion-aware conflict graph and searches reusable gaps, which permits overlap within a phase and across different extents. Best-fit reduces fragmentation at predictable compile time. Exact placement removes remaining fragmentation when its bounded exhaustive search completes.

Allocation quality excludes the fixed control prefix. For a nonempty problem, payload high-water mark `H` is `arenaBytes - payloadBaseOffset`; `Hmin` is the proven minimum of the same quantity. Absolute fragmentation is `H - Hmin`, relative fragmentation is `(H - Hmin) / Hmin`, and packing efficiency is `Hmin / H`. Exact placement has efficiency 1 when it completes. Control overhead is measured separately as `payloadBaseOffset`, and total per-core SRAM use remains `arenaBytes`.

Regression tests compare exact placement with an independent exhaustive byte-offset oracle for all 5,184 combinations of four-region conflict graphs and three aligned extent sizes. The individual first-fit and best-fit strategies are optimal in 5,035 cases (97.13%). Across all cases, `sum(Hmin) / sum(H)` is 99.26%. Across the 149 suboptimal cases, the same payload-weighted efficiency is 78.74%, the worst-case efficiency is 66.66%, the average excess is 1.32 alignment units, and the maximum excess is three alignment units. The worst case is a four-region chain with equal extents: stable owner order uses three address levels, while alternating the chain endpoints uses two. Best-fit cannot improve this case because equal extents present the same gaps as first-fit. Exact placement finds the two-level result. The combined default reduces suboptimal cases to 68 and total excess from 196 to 80 alignment units, with approximately 99.70% aggregate efficiency and 71.43% worst-case efficiency. These synthetic cases provide a stable regression baseline rather than a workload distribution.

Compiler-level tests compare 616 lifetime-derived placements with a separate exhaustive oracle, retain a fixed BF16 fragmentation case, and verify deterministic metadata. In the fixed case, decreasing placement uses 45,056 payload bytes while exact placement uses 32,768 bytes: 72.73% efficiency and 12,288 excess bytes. Device tests verify that reuse and 96 simultaneously live DFBs preserve data while using no TT-Metal descriptors. Domain tests independently derive conflicts from per-core execution schedules and check 8,688 placements across both target alignments and all strategies, including exact minima. A two-core copy test verifies one-page temporal reuse on one core while retaining simultaneous storage on the other, across BF16/FP32 and DRAM/SRAM inputs.

The two-core uneven-demand regression assigns 16 tiles to one core and one tile to the other. Uniform allocation reserves 65,664 bytes for BF16 and 131,200 bytes for FP32; per-core allocation uses 34,944 and 69,760 bytes respectively, including the common 64-byte control prefix on each core. The test checks exact output equality with DRAM and SRAM tensors and verifies backing extents through runtime reports. This measures reduced reservation for that workload, not execution speed or a universal improvement over another allocator.

## Implemented Contract

- Uniform arena allocation or independent per-core domains with shared multicast receiver layouts.
- Compiler-owned static payloads and existing height-, width-, or block-sharded tensor-backed payloads.
- Validated allocation groups with one shared state record and the largest required compiler-owned payload envelope.
- One-block transactions and complete-capacity tensor publication or consumption, with positive capacity below `2^31` pages.
- Consumer-owned replacement writes into the acquired read window without changing occupancy or sequence state.
- Native BF16 and FP32 address-based compute for 1x32, 2x32, 4x32, 8x32, 16x16, 16x32, 32x16, and 32x32 tiles, plus typed external BFP4_B and BFP8_B descriptors for 32x32 tiles.
- Address-based tensor transfer, elementwise compute, matmul, reductions, broadcast, transpose, and selected activation operations covered by the implementation tests.
- Typed external C++ calls with explicit DFB effects, compiler-owned or tensor-backed payloads, elementwise multiply, and multi-tile block matmul.
- Device-domain and mesh program placement with declarative external runtime resources.
- Blackhole selected reset, reset-all, and reconfiguration.
- Wormhole allocation, transfer, compute, external descriptors, and local PipeNet compilation without reset or reconfiguration.
- Local intra-device PipeNet transfers with compiler-owned or tensor-backed computed receiver addresses and no TT-Metal DFB descriptors.
- Generated inter-device PipeNet transfers with compiler-owned or tensor-backed computed receiver addresses and per-device fabric binding.

## Validation

| Scenario | Evidence |
| --- | --- |
| Sub-tile compute | 220 compiler-managed SRAM Blackhole device-correctness cases across BF16/FP32, DRAM/SRAM tensors, both decreasing strategies, supported tile dimensions, a tensor-backed multi-page expression, elementwise operations, broadcast, matmul, transpose, reductions, mixed dimensions in one compute kernel, equal-byte-size width transitions, and typed external descriptors; 192 Metal device-correctness cases preserve existing behavior |
| Blackhole transfer and compute | Device correctness across BF16/FP32, DRAM/SRAM tensors, repeated executions, counter wraparound, 96 live DFBs, arithmetic with 66 allocated DFBs, matmul, reductions, residual, MLP, attention, and expert merge |
| [Allocation domains](../../test/python/sram_domains.py) | Blackhole device correctness for per-core copies, mixed extents, compute, reset/reconfiguration, external calls and PipeNet receivers; host tests cover distinct mesh-device addresses, selected-device placement, shared-layout validation and generated-runner metadata. Wormhole domain allocation is compile-only. |
| Tensor-backed storage | 46 Blackhole device-correctness cases across BF16/FP32, compiler-owned scratch and tensor-backed storage, height/width/block sharding, row/column shard orientation, nonzero byte offsets, complete-capacity publication, replacement, and repeated execution; compile-only metadata checks cover all three allocator strategies |
| Allocation groups | Four compiler-managed SRAM Blackhole device-correctness cases across BF16/FP32 and DRAM/SRAM tensors for repeated shared-state handoff with different member capacities; compile-only checks cover all three allocator strategies, tensor-backed ownership, rejection with reuse disabled, and tensor byte-range alias diagnostics |
| External calls and lifecycle boundaries | 20 Blackhole device cases across BF16/FP32 and DRAM/SRAM, including repeated selected reset, reset-all, reconfiguration, live state preservation, payload reuse, and reset of allocation index 65 |
| External descriptor and elementwise compute | 98 Blackhole device-correctness cases across BF16/FP32, compiler-owned and tensor-backed storage, TT-Metal and compiler-managed storage, both decreasing strategies, reset and reconfiguration, repeated invocation, and a 70-DFB composition |
| External block matmul | 148 Blackhole device-correctness cases across one tile through a 2x2-tile result, BF16/FP32, BFP4_B/BFP8_B weights, compiler-owned and single-core height/width/block-sharded tensor-backed storage, TT-Metal and compiler-managed storage, both decreasing strategies, selected reset, reconfiguration with payload reuse, repeated invocation, and a gated-MLP composition with native normalization, activation, and residual operations |
| Allocation | 16 Blackhole default/combined-strategy device cases cover BF16/FP32 transfers, DRAM/SRAM inputs, reuse, and 96 live DFBs; 160 graphs with 8-512 regions check non-regression against first-fit, stable ties, and exact-budget determinism; 12 Blackhole exact-strategy device-correctness cases cover BF16/FP32, DRAM/SRAM inputs, repeated invocation, compiler-owned storage reuse, 96 simultaneous DFBs, and 96 sequential DFBs; 5,184 arbitrary conflict and extent combinations compare exact placement with an independent byte-offset oracle and record greedy excess; 31,948 compile-only generated placements cover all four strategies, Wormhole and Blackhole alignment, conflicts, reuse enabled and disabled, determinism, and exact budget boundaries; 616 exact cases are checked against a separate exhaustive oracle; an adversarial fragmented graph verifies that exact placement improves both decreasing strategies and proves the minimum; negative tests distinguish proven infeasibility from work-limit exhaustion |
| Wormhole | Compile-only allocation, typed external descriptor, local PipeNet pipeline, and UNPACK/MATH/PACK target compilation; negative reset and reconfiguration diagnostics |
| Runtime placement and resources | Runtime-unit evidence for one-device and device-domain descriptors, replicated mesh placement, lockstep arena binding, external fabric bindings, resource lifetimes, program hashes, tensor-address cache identity, and owned-allocation accounting; 18 Blackhole device-correctness cases for typed external calls with semaphores, runtime arguments, defines, repeated invocations, BF16/FP32, DRAM/SRAM, generic/specialized kernels, and both memory models |
| Local PipeNet execution | Blackhole device correctness across BF16/FP32, DRAM/SRAM, both decreasing strategies, four synchronization protocols, ready-receive selection, grouped transfers, reset, reconfiguration, typed external DFB calls, repeated invocation, two-axis matmul distribution, and receiver logical indices above the Metal limit; compile-only coverage includes exact placement |
| Generated fabric PipeNet execution | Compile-only full-pipeline coverage preserves generated routes, routing-plane operations, compiler-owned receiver offsets, and descriptor independence; runtime-unit coverage verifies per-device route binding for compiler-owned and tensor-backed receiver addresses |
| Invalid contracts | Compiler diagnostics for malformed metadata, unsupported transactions and tile forms, unknown external effects, numeric external DFB indices, storage ownership, and budget overflow |

Relevant tests are [transfer and allocator device tests](../../test/python/test_compiler_l1.py), [compute device tests](../../test/python/test_compiler_l1_compute.py), [sub-tile compute device tests](../../test/python/test_subtile_compute.py), [lifecycle and external-call device tests](../../test/python/test_compiler_l1_lifecycle.py), [external elementwise device tests](../../test/python/test_external_dfb_reuse.py), [external matmul device tests](../../test/python/test_external_matmul.py), [local PipeNet device tests](../../test/python/pipe/test_compiler_l1_pipenet.py), [generated fabric device tests](../../test/python/fabric/test_ccl.py), [runtime placement tests](../../test/python/test_kernel_runner.py), [external runtime-resource device tests](../../test/python/test_operation_runtime_resources.py), and [generated allocator stress tests](../../test/ttlang/Dialect/TTL/Transforms/compiler_l1_stress.py).

## Other Follow-on Work

The allocator extensions and their dependencies are outlined above. Other backend work comprises:

- Qualify additional external C++ kernels against the typed descriptor interface; add target operations only when required address or completion primitives are absent.
- Add row-major metadata, partial-block and general contiguous multi-block transactions, with explicit stride, capacity, and wrap rules.
- Define and device-qualify Wormhole reset and reconfiguration behind the common target interface.
- Qualify complete model layers and measure device cycles, reserved SRAM, initialization cost, compile time, and generated code size against `metal-cb`.

Each extension preserves validation before mutation, architecture isolation, explicit ownership, and compiler-managed descriptor independence.
