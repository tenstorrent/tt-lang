# Compiler-Managed SRAM Allocation

## Overview

TT-Lang normally assigns each logical dataflow buffer (DFB) a TT-Metal DFB descriptor. Wormhole B0 provides 32 descriptor indices and Blackhole provides 64. A program can therefore exhaust descriptor indices while sufficient SRAM storage remains.

`--ttl-memory-model=compiler-l1` replaces descriptor-indexed storage with compiler-assigned SRAM byte ranges. The Python DFB API and its producer/consumer semantics remain unchanged. `metal-cb` remains the default.

| Property | `metal-cb` | `compiler-l1` |
| --- | --- | --- |
| Allocated identity | TT-Metal DFB index | Logical DFB index and compiler storage owner |
| Storage address | TT-Metal descriptor | Compiler arena or tensor base plus byte offset |
| Capacity limit | SRAM capacity and 32 or 64 descriptor indices | SRAM capacity, control records, and alignment |
| Payload reuse | Requires the Metal descriptor and backing-storage contracts | Requires noninterfering completed lifetimes or an explicit validated allocation group |
| Producer/consumer state | TT-Metal DFB interface state | Two four-byte SRAM sequence slots per storage owner; no DFB index or local semaphore id |
| Tensor-backed storage | Installed through a TT-Metal descriptor | Addressed directly through the tensor runtime argument |
| Allocation groups | Reuse a physical descriptor and its storage contract | Share one validated storage owner and control record |
| Reset and reconfiguration | Blackhole TT-Metal interface reset and runtime descriptor reconfiguration | Blackhole address-based state reset; page size, pages per block, block count, and storage capacity remain unchanged |
| External C++ DFB access | Numeric index or typed descriptor bound to a TT-Metal DFB | Typed descriptor bound to compiler-assigned storage |
| PipeNet receivers | TT-Metal descriptor with computed or receiver-published addressing | Compiler arena or tensor-backed computed address for intra-device and generated inter-device transfers; no TT-Metal DFB descriptor |

Shared terminology is defined in the [TT-Lang specification glossary](../sphinx/specs/TTLangSpecification.md#appendix-a-glossary). The DFB protocol and lifecycle rules are defined in [DFB Management](DFBManagement.md).

## Design

The compiler separates lifetime and ownership proofs from byte placement. It collects an immutable allocation problem, validates all domain placements, and only then emits offsets and runtime bindings. A strategy can improve placement without changing program scheduling or weakening completion requirements.

### Storage Ownership and Arena Layout

Each compiled Python `ttl.operation` contributes compiler-planned scratch storage on its participating worker cores. That storage remains live through one completed invocation. `--ttl-sram-allocation-mode=uniform` uses the same relative layout on every core. `per-core` creates one layout for each multicast-derived core domain; disjoint domains have independent layouts. Kernels receive the core-local arena base as one common runtime argument, so the argument count does not depend on the number of logical DFBs.

The arena has two sections:

```text
0                                                   arenaBytes
+----------------------+----------------------------------+
| 8-byte state records | aligned, reusable payload ranges |
+----------------------+----------------------------------+
```

Each storage owner has one 8-byte DFB control record. An ungrouped logical DFB is its own storage owner. A validated allocation group has one storage owner shared by its members. The first 32-bit slot is the published-page sequence and the second is the consumed-page sequence. A capacity of at most 32,768 pages permits native 16-bit loads and stores because the sequence modulus is twice the capacity; larger capacities use all 32 bits. Allocation-group validation requires one element type and therefore one page size. Page units preserve one cursor interpretation when members use different pages-per-block and block-count values. Separate slots allow the producer and consumer to update state without an atomic read-modify-write operation.

#### DFB Sequence Protocol

The following input-DFB example uses a six-page payload ring and transfers three pages per transaction. The declared `(1, 3)` block contains three pages, and `block_count=2` provides two blocks for producer-consumer overlap. A sequence counter records page progress modulo twice the DFB capacity. `published` records producer progress and advances after payload writes complete. `consumed` records consumer progress and advances after payload reads complete. Their difference gives the number of ready pages. `reserve_back` waits for free capacity and captures the published sequence that addresses the write window. `wait_front` waits for ready pages and captures the consumed sequence that addresses the read window.

[Open the latest interactive diagram](https://gist.github.com/brnorris03/a24ddc3de1e4d675ff45965c5ca17cb7/raw/dfb-sequence-counters.svg?raw=1) to select stages by mouse or keyboard. The embedded view shows the static overview. The [Gist file view](https://gist.github.com/brnorris03/a24ddc3de1e4d675ff45965c5ca17cb7#file-dfb-sequence-counters-svg) always identifies the latest revision.

![Compiler-managed SRAM DFB sequence protocol](https://gist.github.com/brnorris03/a24ddc3de1e4d675ff45965c5ca17cb7/raw/dfb-sequence-counters.svg?raw=1)

The final diagram state shows a later transaction reusing pages 0 through 2 of the same DFB after pages 3 through 5 complete. This ring-slot reuse is distinct from allocator reuse. Allocator reuse assigns the same SRAM byte range to different storage owners only when completion analysis proves that their lifetimes cannot overlap.

For capacity `N`, both sequences are in `[0, 2N)`. Occupancy is `(published - consumed) mod 2N` and remains in `[0, N]`. Payload addressing uses `sequence mod N`. With six payload pages, `published = 0, consumed = 0` is empty, while `published = 6, consumed = 0` is full. Both select payload page 0, but the twelve-value counter range keeps the states distinct. The extra counter states do not allocate more payload pages. The current single-producer/single-consumer contract makes each counter single-writer and avoids atomic read-modify-write operations.

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

### Implemented Contract

Each compiled `ttl.operation` has a scratch layout for one invocation. Persistent declarations retain stable storage across prepared operation launches until the owning `SRAMStorage` releases them. Joint placement reserves persistent payloads with the scratch arenas of all prepared operations and permits scratch overlap only between completion-ordered launches. Compiler-owned payload sizes are static; tensor-backed payloads retain their existing height-, width-, or block-sharded allocations. Uniform and per-core placement share the same ownership and completion rules.

DFB transactions operate on one block, or publish/consume a tensor-backed DFB's complete capacity. Capacity is positive and below `2^31` pages. Consumer-owned replacement writes remain within the acquired read window and do not change occupancy or sequence counters. Compute formats and tile dimensions, reset synchronization, and external/transport bindings are specified in the backend subsections below.

`wait_front` and `reserve_back` capture the acquired sequence after their availability checks succeed. Every address-bearing operation in that transaction derives its payload address from the captured sequence, so later counter updates cannot move an active transaction to another payload window. Initialization and configuration use storage metadata without acquiring a window. A release completes outstanding payload access before updating its sequence counter. When the immediately preceding `ttl.wait` already proves completion of the exact transaction, the release reuses that directional NoC barrier instead of issuing a second full NoC barrier.

The runtime clears every control record before dispatch. Within a generated kernel, one `Buffer` object retains the sequence advanced by its processor and reloads only the peer sequence needed to test availability. A function call, opaque DFB user, reset, reconfiguration, or external descriptor prevents this optimization because another object may advance the same interface. Peer loads retain the target memory fence.

The 32-index Wormhole B0 and 64-index Blackhole limits apply only to TT-Metal DFB descriptors. A compiler-managed logical DFB does not allocate one of those descriptors. Its local producer/consumer protocol polls two SRAM sequence slots and does not allocate a local semaphore id. Logical DFB count is therefore limited by SRAM use, generated code and configuration size, runtime arguments, and Metal program capacity instead of the hardware descriptor count.

PipeNet synchronization is a separate resource. TT-Lang currently has 16 local hardware semaphore ids. Generated PipeNet counters use available local ids and then use host-created `GlobalSemaphore` SRAM words; exhaustion of the 16 local ids does not restore a 16-DFB limit. Global counters add SRAM allocations and runtime arguments and remain subject to the combined SRAM and program-capacity checks.

### Shared Runtime Requirements

The compiler placement problem describes relative offsets inside compiler-owned arenas. The runtime also needs one representation that can describe those movable arenas together with fixed tensor storage. Before creating runtime resources, it prepares an immutable requirement for each physical owner. The requirement records ownership, lifetime, alignment, one interval per device/core location, and selected equal-base groups. Each location interval records its extent, whether it contains payload storage, and an optional fixed base. An equal-base group requires its member intervals to start at one physical address; their extents may differ. Locations outside a group remain independently addressable. An empty device coordinate denotes the device domain selected when storage is bound.

For example, suppose owners `A` and `B` overlap in lifetime. `A` requires the same base on two cores for remote access, but its capacity differs by core. `B` exists only on core `(0,0)` and has no address relationship with `A`:

```text
              0 KiB        2       3       4
core (0,0):  [ A: 2 KiB ][B:1 KiB]
core (1,0):  [ A: 4 KiB                 ]
              ^ A starts at the same address on both cores
```

The model represents these as three owner-location intervals. One equal-base group connects only the two `A` intervals. The independent `B` interval conflicts with `A` on core `(0,0)` without constraining core `(1,0)`.

The canonical property definitions are `SRAMStorageRequirement`, `SRAMLocationRequirement`, and `SRAMEqualBaseGroup` in [`_sram_requirements.py`](../../python/ttl/_sram_requirements.py). The C++ placement subset is declared in [`SRAMAllocator.h`](../../include/ttlang/Dialect/TTL/Transforms/SRAMAllocator.h).

| Property | Values | Meaning |
| --- | --- | --- |
| Owner | Compiler arena, tensor argument, or persistent declaration | Identifies one physical storage allocation and its aliases. |
| Placement ownership | Fixed or movable | Every location of fixed storage supplies an existing address that the allocator must preserve. Movable storage supplies no address; the allocator selects its offsets. |
| Lifetime | External, persistent, or invocation | External storage follows caller ownership, persistent storage remains live until `SRAMStorage.close()`, and invocation storage remains live through one completed operation launch. |
| Locations | One or more device/core pairs | States where this owner requires storage. It does not require those locations to use the same address. |
| Location extent | Positive byte count | Gives this owner's required capacity at one location. Extents may differ by location. |
| Payload presence | Present or absent | Distinguishes a payload-bearing location from a location that contains only the owner's control state. |
| Address relation | Independent or member of an equal-base group | An equal-base group requires only its selected locations to use the same numeric address. |
| Placement constraint | Optional fixed base | Supplies the immutable address of fixed storage. Movable storage has no fixed base. |
| Alignment | Power-of-two byte count | Constrains every selected or supplied address for the owner. |

Lifetime and address constraints are independent. A persistent owner may cover one core, selected cores, or all cores. Multicast adds an equal-base group only for the addressed owner and receiver locations. Invocation scratch remains invocation-scoped with or without an equal-base constraint. These facts do not require separate persistent, multicast, and general-purpose arenas. A placement strategy may combine storage when its conflicts and address constraints permit reuse.

`SRAMAddressing` is selected when the runtime realizes requirements as TTNN reservations. Uniform addressing requests one physical base for every pool location; per-core addressing permits independent physical bases. It is not a requirement property because a single-location owner has no address-equality relation. Equal-capacity groups are also backend constraints: they tell the allocator that one retained pool reserves its largest location high-water mark at every member location.

DFB control and payload ranges are uses of a requirement, not additional owners. Several tensor-backed DFBs that reference one tensor therefore produce one fixed requirement and several byte-range uses. The fixed requirement covers the tensor's complete logical shard extent and shard grid, including bytes and cores not referenced by those DFBs. Compiler control and payload ranges reference their arena requirement. Each use is validated against the extent at every participating location. The runtime derives arena sizes and core groups from this prepared record, so allocation and binding use the same validated information.

Persistent declarations use the same requirement type. They remain movable until `SRAMStorage.allocate()` jointly places them with the arenas of every prepared operation. Caller-supplied tensors retain their existing addresses and remain outside these owned pools. The storage owner enforces completion between prepared operations before allowing their scratch arenas to reuse bytes.

### Completion and Storage Conflicts

Payload completion and storage lifetime are separate proofs. The release optimization applies only to one acquired transaction whose complete storage-use set contains one asynchronous copy and its direct wait. The transfer direction must match the release: a NoC read completes producer writes before `push_back`, and a NoC write completes consumer reads before `pop_front`. The acquire and release must cover the same tile count, and the wait must immediately precede the release. Multiple copies, indirect handles, additional storage users, ambiguous ownership, partial releases, and control-flow-mediated transfer handles retain the target completion barrier.

```text
reserve -> NoC read  -> read barrier  -> publish sequence
wait    -> NoC write -> write barrier -> consume sequence
                              |
                              +-- exact transaction proof permits state update

Without the proof:
payload accesses -> target completion barrier -> sequence update
```

```text
provePayloadComplete(release):
    require exactly one structural acquire owner
    require acquire and release tile counts to match
    collect every storage use owned by the acquired transaction
    require exactly one ttl.copy and one ttl.wait
    require the wait to consume that copy's only transfer handle
    require the copy direction and SRAM operand to match the transaction
    require the wait to immediately precede the release
    return true
```

Allocation consumes the existing logical-identity, allocation-group, and completion-aware lifetime analyses. The compiler validates every allocation group and builds the complete conflict relation before changing IR. Unknown launch domains, unproved completion, concurrent lifetimes, and incompatible storage ownership remain conflicts.

The shared storage conflict analysis accepts an explicit storage mode. Metal storage includes conflicts caused by runtime descriptor installation and Metal-managed backing changes. Compiler-managed storage excludes those conflicts because each logical DFB's page size, pages per block, block count, and storage capacity remain constant during execution, and each validated storage owner has a control record. This distinction permits byte reuse across a reconfiguration boundary after the prior lifecycle ends while preserving DFBs that remain live across the boundary.

Validated allocation-group members are collapsed into one storage owner. The owner conflicts with another owner if any member pair conflicts. This preserves all lifecycle conflicts while allowing the explicit ownership transfer represented by the group. The existing group validator proves ordering, capacity, cursor continuity, and storage compatibility; the compiler-managed SRAM allocator does not duplicate or weaken those checks.

Tensor-backed DFBs require an exact, non-empty launch-node domain. On a shared launch node, partial byte-range overlap is rejected because it does not represent a complete storage ownership transfer. Identical byte ranges are permitted for the same storage owner or for nonconflicting lifetimes. Disjoint ranges do not alias.

This design reuses one lifetime model for both memory backends. The allocator cannot serialize operations or remove a conflict to make a program fit.

```text
buildStorageConflicts(lifetimes, storageMode, selectedCores = all cores):
    conflicts = empty graph
    for each unordered pair (left, right):
        if either launch domain or access completion is unproved:
            conservatively retain the conflict
            continue
        for each selected core where both may be active:
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

### Core Allocation Domains

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

## Placement API and Algorithms

Placement is a reusable, compiler-internal C++ library and API with no dependency on MLIR operations or target identities. Compiler analyses construct immutable storage requirements; a selected strategy returns byte offsets without inspecting IR or changing scheduling. New strategies implement the common interface without changing analysis, target code generation, or result validation. Architecture adapters supply alignment and per-location budgets through the same API.

The API supports two inputs. `SRAMAllocationProblem` places one list of regions in one address space and remains the interface for compiler operation arenas. `SRAMLocationAllocationProblem` places owner-location intervals across several independently bounded address spaces, enforces selected equal offsets, and records locations whose backing allocation uses one common capacity. The latter represents joint persistent and scratch placement without defining persistent, multicast, and ordinary scratch as separate arena types.

Every allocator result passes the same validation before IR mutation. Validation requires the correct offset count, target alignment, offsets at or above the payload base, intervals within the SRAM budget, disjoint intervals for every conflict, and an exact payload high-water mark. Allocation policy cannot weaken these invariants. The domain allocation entry point maps offsets to storage owners and retains the control prefix when computing the arena size.

### C++ Strategy Contract

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

struct SRAMAllocationLocation {
  uint64_t payloadBaseOffset;
  uint64_t budgetBytes;
};

struct SRAMAllocationRegion {
  unsigned ownerIndex;
  unsigned locationIndex;
  uint64_t bytes;
  std::optional<uint64_t> fixedOffset;
};

struct SRAMEqualOffsetGroup {
  llvm::SmallVector<unsigned> regionIndices;
};

struct SRAMEqualCapacityGroup {
  llvm::SmallVector<unsigned> locationIndices;
};

struct SRAMLocationAllocationProblem {
  llvm::SmallVector<SRAMAllocationLocation> locations;
  llvm::SmallVector<SRAMAllocationRegion> regions;
  InterferenceGraph conflicts{0};
  llvm::SmallVector<SRAMEqualOffsetGroup> equalOffsetGroups;
  uint64_t alignmentBytes;
  llvm::SmallVector<SRAMEqualCapacityGroup> equalCapacityGroups;
};

struct SRAMLocationAllocationSolution {
  llvm::SmallVector<uint64_t> offsets;
  llvm::SmallVector<uint64_t> highWaterBytes;
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

  FailureOr<SRAMLocationAllocationSolution>
  allocateLocations(const SRAMLocationAllocationProblem &problem,
                    std::optional<unsigned> &failureRegionIndex,
                    std::string &failureReason) const;

private:
  virtual FailureOr<SRAMAllocationSolution>
  allocateImpl(const SRAMAllocationProblem &problem,
               std::string &failureReason) const = 0;

  virtual FailureOr<SRAMLocationAllocationSolution>
  allocateLocationsImpl(const SRAMLocationAllocationProblem &problem,
                        std::string &failureReason) const = 0;
};

FailureOr<std::unique_ptr<SRAMAllocator>>
createSRAMAllocator(llvm::StringRef name, const SRAMAllocatorOptions &options,
                  std::string &failureReason);

} // namespace mlir::tt::ttl
```

`regionBytes[i]` is the nonzero, aligned extent of allocator region `i`. The caller retains the mapping from allocator-region indices to compiler-owned storage-owner indices. `conflicts` is the shared undirected resource-interference graph. `payloadBaseOffset` is aligned and does not exceed `budgetBytes`. The problem is immutable after construction. Region order supplies the final deterministic tie-break.

`SRAMAllocator::allocate` is the public, nonvirtual entry point. It validates the problem, invokes the private strategy method, and validates the solution. This structure keeps strategy selection replaceable while enforcing one correctness contract. On success, `offsets` has one entry per allocator region and `arenaBytes` is the exact maximum payload end, or zero when no allocator regions exist. On failure, `failureReason` contains diagnostic text and `failureRegionIndex` identifies an allocator region only when the error applies to one region. The allocator layer does not emit diagnostics or modify IR.

`allocateLocations` applies the same validation structure to owner-location intervals. Each `(ownerIndex, locationIndex)` pair is unique. Conflicts can connect only intervals at the same location. An equal-offset group contains at most one interval per location and requires every member to start at one physical byte offset; members can have different extents. A fixed interval retains its supplied offset. An equal-capacity group states that its backing allocation reserves the largest member high-water mark at every member location. Locations outside these groups contribute their individual high-water marks. The result reports one offset per interval and the exact high-water mark at every location.

```text
relative offset       0       64      128      192      256
core (0,0)            [persist][----- scratch -----]
core (1,0)            [------ persistent ------][local]
                       ^
                       same base for this persistent owner

The persistent intervals start together. Their ends differ. Local storage is
constrained only by conflicts at its own core.
```

The allocator first combines each equal-offset group into one placement variable. A candidate offset must fit every member against that location's budget, fixed intervals, and conflicting placed intervals. Ungrouped intervals form one-member variables. Equal-capacity groups affect the physical-reservation objective and require their common capacity to fit every member budget; they do not change interval addresses. This representation keeps address equality and backing-allocation capacity as separate constraints.

```text
allocateLocations(problem):
    validate locations, owner-location intervals, conflicts, fixed offsets,
        equal-offset groups, and equal-capacity groups
    variables = equal-offset groups plus every ungrouped interval
    place fixed variables
    for variable in strategy order:
        candidates = strategy candidate offsets
        retain candidates that fit every member location
        select a candidate according to the strategy
    compute the exact high-water mark at every location
    compute physical reservation from independent and equal-capacity locations
    validate all offsets, conflicts, equalities, fixed offsets, and budgets
    return one offset per owner-location interval
```

### Domain Allocation Contract

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

### Adding a Strategy

`SRAMAllocatorOptions` contains limits that affect strategy execution but do not change the allocation problem. `exactSearchLimit` bounds exact search. With `D` calls, total search work can reach `D` times the configured limit. `createSRAMAllocator` maps stable compiler-option names to implementations and supplies these options. `getName()` identifies the implementation in validation diagnostics. A new implementation derives from `SRAMAllocator`, implements `getName()`, `allocateImpl()`, and `allocateLocationsImpl()`, and registers its name in the factory. It cannot change conflict construction or bypass common validation.

The public interface and validation are in [SRAMAllocator.h](../../include/ttlang/Dialect/TTL/Transforms/SRAMAllocator.h) and [SRAMAllocator.cpp](../../lib/Dialect/TTL/Transforms/SRAMAllocator.cpp). [SRAMAllocator_Location.cpp](../../lib/Dialect/TTL/Transforms/SRAMAllocator_Location.cpp) implements placement-variable construction, fit checks, ordering, and objective calculation shared by the location strategies. [SRAMAllocator_Greedy.cpp](../../lib/Dialect/TTL/Transforms/SRAMAllocator_Greedy.cpp) and [SRAMAllocator_Location_Greedy.cpp](../../lib/Dialect/TTL/Transforms/SRAMAllocator_Location_Greedy.cpp) implement greedy placement. [SRAMAllocator_Exact.cpp](../../lib/Dialect/TTL/Transforms/SRAMAllocator_Exact.cpp) and [SRAMAllocator_Location_Exact.cpp](../../lib/Dialect/TTL/Transforms/SRAMAllocator_Location_Exact.cpp) implement exact search. Private headers connect the implementations without exposing strategy details through the public API.

### Greedy Placement

| Strategy | Placement rule | Result |
| --- | --- | --- |
| `multi-order-decreasing` (default) | Run first-fit decreasing with stable and degree-aware equal-size ordering; retain the smaller arena and the stable layout on ties. | Deterministic placement no larger than first-fit decreasing. |
| `first-fit-decreasing` | Place decreasing extents at the lowest aligned legal offset. | Deterministic feasible placement. |
| `best-fit-decreasing` | Place decreasing extents in the finite legal gap with the least unused space; lower offset resolves ties. | Deterministic feasible placement that can reduce fragmentation. |
| `exact` | Search aligned subset-sum offsets with branch-and-bound. | Proven minimum arena, or a precise inconclusive or infeasible diagnostic. |

The decreasing strategies place larger regions first because large extents fit in fewer gaps. Storage-owner order resolves equal-size ties in the individual first-fit and best-fit strategies. They are greedy heuristics and can produce different arena sizes.

For a location problem, first-fit selects the lowest location base or conflicting-interval end that fits every member of a placement variable. Best-fit minimizes the increase in physical reservation. Multi-order compares stable and conflict-degree-aware orders by the same measure. Exact search constructs the bounded closure of every location base, fixed boundary, and aligned interval extent, then searches those candidates to minimize physical reservation. An independent location contributes its high-water mark. An equal-capacity group contributes its member count multiplied by its largest high-water mark. The closure permits an earlier placement variable to start above a later variable; candidates derived only from already placed intervals would not prove a minimum.

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

### Exact Placement

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

## Backend Integration

### Architecture Boundary

Common allocation and lowering contain no architecture branches. `compiler_l1_target.h` provides arena-base access, SRAM loads and stores, producer/consumer completion, and processor ownership. `compiler_l1_compute_target.h` provides LLK address conversion, format configuration, and address-based compute operations. Wormhole and Blackhole differences remain inside these target interfaces.

### Runtime Allocation and Binding

The runtime represents each domain arena as a row-major, height-sharded TTNN SRAM tensor, with one row per member core. This directly represents equal-length storage within a domain. Uniform mode uses TT-Metal lockstep allocation across all selected cores and devices. Per-core mode uses independent allocation for singleton domains and lockstep allocation for multicast receiver domains; kernel descriptors bind the actual address on each selected device. Tensor-backed payloads retain their existing height-, width-, or block-sharded allocations.

The arena is passed as an auxiliary `generic_op` input so TTNN retains it through device execution while preserving the user output position. Arena and synchronization scratch are zero-initialized. Declarative runtime resources compose with the arena: semaphore descriptors, per-kernel runtime arguments, compile-time defines, external fabric bindings, and their lifetime owners retain their existing validation and program-hash contracts. Runtime resource caching includes the allocation metadata and reset count, so incompatible layouts do not share resources.

The existing core-specialization pass creates one kernel instance per core. Finalized metadata supplies each instance's payload offsets and each computed PipeNet argument's destination DFB, core, and logical device. Transport finalization preserves those identities when it removes unused receiver arguments. The runtime validates domain membership, shared layouts, storage aliases, and receiver bindings before creating resources. Independently addressed tensor backing requires direct local access on every executing core; general tensor access and multicast require one common base address and are rejected.

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

### Reset and Reconfiguration

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

### Compute Configuration

TT-Metal compute APIs normally read the data format and tile dimensions from a DFB descriptor. Compiler-managed storage creates no such descriptor, so each generated compute operand type contains the data format, page size, tile height, and tile width. The common compute interface derives face dimensions from those values. Native compiler-managed compute accepts BF16 and FP32 tiles supported by the shared compute-target contract: 1x32, 2x32, 4x32, 8x32, 16x16, 16x32, 32x16, and 32x32 on Wormhole and Blackhole. Typed external compute descriptors additionally accept BFP4_B and BFP8_B with 32x32 tiles. The external function supplies the operation-specific mixed-format compute sequence; native operations remain restricted to formats supported by their generated implementations.

Compute setup is shared by operands with equal formats, page sizes, tile dimensions, and direct-to-destination settings. Storage offsets and capacities remain properties of each DFB. Separating hardware properties from storage identity prevents repeated setup code from exhausting kernel instruction storage in large DFB compositions.

A generated compute kernel can use different tile dimensions during one execution. `ComputeContext` records the currently programmed input formats, page sizes, and face dimensions, together with the output format and tile dimensions. An exact packed identity makes the common unchanged-configuration check constant-sized. The first operation configures UNPACK, MATH, and PACK. Later operations reconfigure only state that differs. A PACK tile-dimension change requires data-format reconfiguration followed by pack initialization that preserves the existing address modifiers. This sequence follows the TT-Metal LLK contract and avoids repeating hardware configuration. Address-based copy helpers are inlined so the RISC compiler can retain loop-invariant addresses.

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

### External C++ Descriptors

`ttl.dfb_descriptor(dfb)` lowers to a C++ template type containing page size, pages per block, block count, shared storage capacity, state offset, payload offset, and an optional tensor common-argument index. A compute-thread descriptor also contains the data format, tile height, tile width, and direct-to-destination choice.

`bind()` resolves the control-record and payload addresses from the finalized metadata. External functions therefore require no Metal DFB index and no additional runtime argument per DFB. [Compute kernel configuration](ComputeKernelConfiguration.md) defines direct-to-destination selection.

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

### PipeNet Receiver Binding

Intra-device and generated inter-device PipeNets retain the existing transfer plan and synchronization protocols. Compiler-managed allocation changes how a receiver address is obtained. A compiler-owned receiver uses the arena base plus its finalized payload offset. A tensor-backed receiver uses the tensor base plus its finalized byte offset. No TT-Metal DFB descriptor is created for either case.

Generated inter-device transfers compute receiver addresses from the destination device and core. Uniform mode supplies a common lockstep arena base. Per-core mode binds the destination domain's actual allocation address and payload offset. Tensor-backed receivers retain their validated tensor byte ranges. Fabric binding independently resolves logical device coordinates to physical routing targets.

The Metal backend retains receiver publication when one physical DFB index can refer to different storage across reconfiguration epochs. Compiler-managed allocation assigns each finalized DFB index one arena or tensor base for the compiled operation, so that base remains valid for every transfer occurrence.

Producer and wait launch domains are validated for explicit DFB operations and external-call `DFBEffect` declarations. Treating both through the DFB access interface prevents an external producer from being omitted from PipeNet deadlock analysis.

Receiver addresses precede PipeNet scratch and semaphore addresses in the common runtime argument layout. The finalized `ttl.crta_indices` metadata defines the tensor-argument prefix because tensor-backed DFBs can retain tensors that are no longer function operands. The arena base remains the final compiler-managed storage argument. This ordering matches lowering and does not depend on the number of TT-Metal DFB descriptors.

```text
bindReceiverAddress(receiver, destinationDevice, destinationCore):
    if receiver has tensor backing:
        base = tensorBase(receiver.tensor, destinationDevice, destinationCore)
    else:
        domain = allocationDomain(destinationCore)
        base = arenaBase(domain, destinationDevice)
    address = base + finalizedReceiverOffset(receiver, destinationCore)
    require address fits uint32
    return address
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

## Allocation Quality and Validation

### Metrics and Regression Baseline

Allocation quality excludes the fixed control prefix. For a nonempty problem, let `H = arenaBytes - payloadBaseOffset` and let `Hmin` be its proven minimum for the same extents, conflicts, alignment, and domain partition. Excess allocation is `H - Hmin`; relative excess is `(H - Hmin) / Hmin`; packing efficiency is `Hmin / H`. Address gaps are not necessarily excess over the optimum. Control bytes and host reservation padding are measured separately.

The independent exhaustive byte-offset oracle covers all 5,184 four-region graphs and three aligned extent sizes. Its synthetic distribution is a reproducible regression baseline, not a workload distribution:

| Strategy | Aggregate payload efficiency, `sum(Hmin) / sum(H)` | Worst-case efficiency | Total excess, in alignment units |
| --- | --- | --- | --- |
| First-fit or best-fit decreasing | 99.26% | 66.67% | 196 |
| Multi-order decreasing (default) | Approximately 99.70% | 71.43% | 80 |
| Exact | 100% | 100% | 0 |

For equal-sized regions forming a four-vertex conflict chain, an unfavorable stable order uses three address levels although two suffice. Best-fit sees the same equal-sized gaps and cannot improve that placement. Degree-aware ordering addresses this ordering defect; retaining the stable candidate prevents regressions relative to first-fit. Neither greedy ordering proves optimality. A separate BF16 regression requires 45,056 payload bytes with the individual decreasing strategies and 32,768 with exact placement: 12,288 excess bytes and 72.73% efficiency.

The fixed state cost can dominate heavily reused payloads. For 96 ungrouped one-page BF16 DFBs, simultaneous lifetimes require 196,608 payload bytes and 768 control bytes. Fully sequential lifetimes reuse 2,048 payload bytes but retain 768 control bytes, or 27% of the 2,816-byte total before final alignment. Reducing that state cost requires proven ownership transfer, as provided by allocation groups; independent producer and consumer counters cannot simply share a word without changing their update protocol.

Domain placement addresses a different source of waste: reserving the busiest core's layout everywhere. In the two-core 16-tile/one-tile regression, uniform allocation reserves 65,664 bytes for BF16 and 131,200 for FP32. Per-core allocation reserves 34,944 and 69,760 respectively, including control prefixes. These are measured backing extents, not execution-speed results or a claim of globally optimal host placement.

A matched Blackhole benchmark copies one 4x4-tile block 256 times per dispatch through read, compute, and write kernels. Each result below contains 200 measured dispatches per backend after 20 paired warmups; exact outputs pass before and after measurement. Compilation and allocation are outside the measured interval.

| Type and run | Compiler-managed SRAM | Metal DFB | Compiler / Metal | Compiler-managed p95 |
| --- | ---: | ---: | ---: | ---: |
| BF16, first | 220.155 us | 220.558 us | 0.99817 | 220.834 us |
| BF16, repeat | 220.179 us | 220.486 us | 0.99861 | 220.992 us |
| FP32, first | 417.492 us | 443.448 us | 0.94147 | 418.543 us |
| FP32, repeat | 417.514 us | 443.447 us | 0.94152 | 418.733 us |

All four comparisons meet the benchmark's eligibility requirements and have no material order effect. One BF16 run reports serial dependence; its median and ratio agree with the independent repeat. This establishes parity for the measured block-copy workload: compiler-managed SRAM is 0.14% to 0.18% faster for BF16 and approximately 5.85% faster for FP32. It does not establish parity for every operation. The [performance plan](https://gist.github.com/brnorris03/51f10d0f049a4477166317b6cf15f1c9#file-sramperformanceplan-md) records the exact candidate, confidence intervals, retained mechanisms, and validation scope.

The final Blackhole validation covers 180 direct-address compute cases, 220 sub-tile cases, 20 lifecycle cases, 8 external reconfiguration cases, 12 variants of a 70-logical-DFB composition, and 12 cases with 96 simultaneously live logical DFBs. The largest 70-DFB binary has 48,924 bytes of combined TRISC text. Wormhole compile-only validation covers BF16 and FP32 reader, writer, unpack, math, and pack processors.

### Validation Responsibilities

The tests separate placement optimality, lifetime-proof correctness, runtime address binding, and device correctness. Numerical output tests alone cannot detect missed reuse or excessive reservation.

| Contract | Regression evidence |
| --- | --- |
| Legal and efficient placement | [Generated allocator tests](../../test/ttlang/Dialect/TTL/Transforms/compiler_l1_stress.py) check conflicts, alignment, budgets, determinism, reuse modes, both target alignments, and independent exact oracles. Larger graphs check that the default never exceeds successful first-fit placement; negative cases distinguish infeasibility from search-limit exhaustion. |
| Domain-specific reuse | [Domain tests](../../test/python/sram_domains.py) cover uneven demand, per-core lifetime differences, multicast address equality, and reported reservation. Host tests cover distinct device addresses and malformed bindings. |
| Data and lifecycle preservation | [Allocator](../../test/python/test_compiler_l1.py), [compute](../../test/python/test_compiler_l1_compute.py), [transaction](../../test/python/test_compiler_l1_transaction_patterns.py), [sub-tile](../../test/python/test_subtile_compute.py), and [lifecycle](../../test/python/test_compiler_l1_lifecycle.py) device tests cover BF16/FP32, DRAM/SRAM inputs, repeated multi-page blocks, unequal DFB capacities, reuse, tensor backing, groups, reset/reconfiguration, repeated invocations, and logical DFB counts above Metal limits. |
| External and transport integration | [External elementwise](../../test/python/test_external_dfb_reuse.py), [external matmul](../../test/python/test_external_matmul.py), [local PipeNet](../../test/python/pipe/test_compiler_l1_pipenet.py), and [runtime resource](../../test/python/test_operation_runtime_resources.py) tests exercise address-based descriptors and completion contracts. [Runtime tests](../../test/python/test_kernel_runner.py) check ownership, cache identity, mesh placement, and generated fabric binding. |

Blackhole evidence includes device correctness. Wormhole evidence is compile-only for allocation, transfer, compute, typed external descriptors, and local PipeNet lowering; it includes rejection of reset/reconfiguration. Generated inter-device PipeNet evidence is compile-only plus runtime-unit binding checks. These checks do not establish multi-device execution performance.

### Allocation Report

`--ttl-sram-allocation-report` emits JSON records to stderr, each prefixed by `ttlang-sram-report: `. Reporting is disabled by default and inactive with `metal-cb`. The compiler record appears on compilation; the runtime record appears on each invocation that allocates an arena, including compiled-artifact cache hits. Reporting does not change placement or lifetime proofs.

Per-core mode emits one compiler and runtime record per allocation domain. `cores` lists domain members; compiler records also include `domain` and `allocation_mode`. Each runtime record derives the per-core reservation from the allocated arena tensor's total page count and aligned page size, and verifies that pages divide uniformly across the domain. `accounting_source` identifies this as `tensor-buffer-geometry`. Tensor geometry avoids ambiguous address matching because disjoint domains may allocate different buffers at the same SRAM address. These counts describe backing extents, not free-space fragmentation or the largest remaining allocation.

The compiler record has `schema_version: 1` and `phase: "compiler"`. `owners` maps shared storage and control offsets to logical DFBs. `regions` includes declaration locations, core domains, and fixed tensor byte ranges. `logical_conflicts` reuses existing reason names and source evidence; these are analysis facts before allocation-group ownership is applied. `reuse_enabled: false` separately explains policy-disabled reuse. `reused_ranges` lists overlapping compiler-owned owner pairs; its byte counts are not additive when more than two owners reuse a range. `lifetimes` records known and possible core membership, completion proof status, entry locations, and entry/completion event IDs. Event IDs identify partial-order analysis events, not elapsed time or a total execution order.

| Compiler metric | Meaning |
| --- | --- |
| `arena_bytes_per_core` | Planned control prefix plus payload high-water mark. |
| `control_record_bytes`, `control_padding_bytes` | Control state and alignment padding, reported separately. |
| `payload_extent_sum_bytes` | Sum of distinct compiler-owned storage-owner extents, after allocation-group consolidation. |
| `payload_union_bytes` | Number of distinct payload addresses occupied by those extents. |
| `payload_reuse_bytes` | Extent sum minus union; excludes sharing already represented by allocation groups. |
| `payload_gap_bytes` | Payload high-water mark minus union; unused address gaps, not excess over an optimal allocation. |

The runtime record has `phase: "runtime"` and `scope: "arena-reference-device"`. It derives `reserved_bytes_per_core` from the arena buffer's aligned page size and uniform page count, reports the participating `core_count`, their product as `reserved_bytes_on_reference_device`, and `reservation_padding_bytes_per_core` beyond `requested_bytes_per_core`. The requested extent is reconstructed from finalized DFB descriptors; a control-only arena can omit trailing compiler alignment padding from this request. It measures the arena reservation on the mesh reference device; it is not a mesh-wide total or total program SRAM use. Existing tensor payloads, PipeNet scratch, and external resources are outside this runtime total.

A compiler-only report can be obtained with:

```sh
ttlang-opt test/ttlang/Dialect/TTL/Transforms/compiler_l1_multi_order.mlir \
  -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 sram-allocation-report=true})' \
  > /tmp/allocated.mlir 2> /tmp/sram-report.log
```

## Requested Capabilities

The current model names lifetimes directly: persistent storage remains live until its owner closes, while invocation storage remains live through one completed launch. It does not expose static and dynamic as separate allocation categories, and extents cannot depend on runtime values.

| Request | Implemented | Missing |
| --- | --- | --- |
| Late allocation and global minimum | Allocation is deferred until operation requirements are finalized. The optional exact strategy minimizes each conflict component. `SRAMStorage` jointly places persistent declarations and every prepared operation arena. | A device-wide optimum across unrelated storage owners and existing tensors. Existing tensor addresses remain fixed. |
| Full lockstep, ranged lockstep, and per-core allocation | The requirement model and allocator API record independent per-location extents and selected equal-base groups. | Compiler DFB placement still produces one layout per multicast-derived domain. The owned-pool realization still accepts only one all-location group or fully independent locations. |
| Unified tensor and DFB allocation | One requirement model represents fixed tensors, movable persistent tensors, compiler arenas, aliases, lifetimes, per-location extents, and equal-base groups. Joint owned pools pack persistent tensor payloads with prepared operation arenas, including DFB control and payload ranges. | Relocation or packing of existing caller-owned tensors. Runtime-dependent extents also require a separate contract. |
| Lifetime inspection and reuse hints | Automatic completion-aware reuse and the allocation report above. | A user-facing guidance contract that preserves asynchronous completion. |
| Persistent per-core SRAM across launches | `SRAMStorage` owns uniform or per-core BF16/FP32 tiled tensors, preserves their addresses until close, initializes them once, and places them with completion-ordered scratch arenas. External launchers participate through the same ownership protocol. | Concurrent borrowing, device reset or migration, and joint preparation of PipeNet storage, DFB reconfiguration, selected device domains, fabric routes, or opaque runtime-resource factories. |

## Future Work

### Implementation Direction

1. Lifetime guidance. Build on the allocation report. Placement preferences may change ordering but cannot remove conflicts. Reuse existing ownership-transfer operations for semantic lifetime boundaries; validate producer publication and consumer completion, including remote and external users.
2. Compiler owner-location placement. Replace complete multicast-domain allocation requests with one interval per storage owner and core. Project conflicts per core, apply equal-base constraints only to the selected multicast locations, and derive each core's arena extent from the resulting high-water mark.
3. Existing-allocation integration. Obtain complete per-core free intervals from the host allocator and reserve selected intervals conditionally. Extend the immutable allocation problem and its oracle with fixed tensor intervals so owned pools can occupy fragmented gaps without relocating caller-owned tensors or relying on stale occupancy snapshots.
4. Cross-owner optimization. Prepare requirements and completion relations from multiple storage owners before reservation. Optimize total physical reservation while preserving concurrency between owners, then commit every reservation through one rollback-capable transaction.

[Persistent SRAM Storage](PersistentStorage.md) defines ownership and completion across launches. Its [joint-placement implementation](PersistentStorage.md#joint-placement) packs persistent declarations with prepared operation arenas and reuses scratch only after enforced device completion. [Program-capacity validation](PersistentStorage.md#program-capacity) checks code and configuration limits before initialization and publication. Runtime-dependent sizes require a further allocation contract.

### Backend Extensions and Validation

Partial-block and general contiguous multi-block transactions require explicit stride, capacity, and wrap rules; row-major compute requires corresponding metadata. Wormhole reset and reconfiguration require a target synchronization protocol and device correctness testing. Additional external kernels reuse the typed descriptor interface, adding target primitives only where needed.

Complete-layer benchmarks must measure device cycles, actual SRAM reservation, initialization cost, compile time, and generated code size against `metal-cb`. Allocation optimality alone does not establish runtime performance.

### Non-SPSC Protocols

The current control record requires one active writer for each sequence: the producer writes `published`, and the consumer writes `consumed`. Each aligned 32-bit slot is read and written indivisibly and made visible across processors. The two slots are not updated as one atomic 8-byte value, and the protocol does not use atomic read-modify-write operations.

Multiple logical producers or consumers can retain this record only when the compiler proves a total ownership order and device completion before every ownership transfer. True concurrency requires a different protocol. Concurrent producers need an atomic reservation position plus ordered publication or per-slot readiness state. Work-sharing consumers need atomic claims plus completion tracking before reclamation. Broadcast consumers need progress state for each consumer or per-slot acknowledgements. An atomic increment alone is insufficient because claiming a slot does not prove that its payload access completed. Target-specific atomic operations belong behind the common target interface; these protocols can require more than eight control bytes.

The sequence range does not increase payload allocation. A DFB with capacity `N` owns `N` payload pages, while each counter has `2N` values and addresses page `sequence modulo N`. The extra counter states distinguish full from empty after the payload address wraps. The compiler reserves the declared capacity of `pages per block * block count`; it does not infer a smaller maximum occupancy. A declaration whose execution never uses all blocks can therefore reserve unused pages. Reducing that capacity requires proof that the smaller ring preserves progress, including cyclic dataflow and external users, and should account for the performance benefit of producer-consumer overlap.
