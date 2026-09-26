# Compiler-Managed SRAM Allocation

## Purpose

TT-Lang normally assigns concurrently live dataflow buffers (DFBs) to TT-Metal descriptor indices; DFBs with disjoint lifetimes can reuse an index. Wormhole B0 provides 32 indices and Blackhole provides 64. An operation can therefore exhaust indices while sufficient SRAM remains.

`--ttl-memory-model=compiler-sram` replaces descriptor-indexed storage with compiler-assigned SRAM byte ranges. The Python DFB API and its producer/consumer semantics remain unchanged. `metal-cb` remains the default.

An *arena* is the node-local SRAM reservation for one operation execution. A *control record* holds one logical DFB's producer and consumer counters. Two payloads have *noninterfering lifetimes* when enforced completion order prevents simultaneous use. An *address-bearing descriptor* supplies a buffer address and geometry to generated device code without a physical DFB index. "SRAM" names the backend; "L1" names the device memory and its capacity budget.

| Property | `metal-cb` | `compiler-sram` |
| --- | --- | --- |
| Allocated identity | TT-Metal DFB index | Compiler allocation index used only in compiler metadata |
| Storage address | TT-Metal descriptor | Arena base plus compiler-assigned byte offset |
| Capacity limit | SRAM capacity and 32 or 64 descriptor indices | SRAM capacity, control records, and alignment |
| Payload reuse | Requires the Metal descriptor and backing-storage contracts | Requires noninterfering completed lifetimes |
| Producer/consumer state | TT-Metal DFB interface state | Two 32-bit sequence counters per logical DFB |
| Reset and reconfiguration | Blackhole TT-Metal interface reset and runtime descriptor reconfiguration | Rejected until synchronized counter reset is supported. |
| External C++ DFB access | Numeric index or descriptor metadata | Typed address-bearing descriptor |

Shared terminology is defined in the [TT-Lang specification glossary](../sphinx/specs/TTLangSpecification.md#appendix-a-glossary). The DFB protocol and lifecycle rules are defined in [DFB Management](DFBManagement.md).

## Allocation Model

For a nonempty allocation plan, each invocation of a compiled Python `ttl.operation` owns one arena on each participating worker node. Every arena uses the same relative layout. Kernels receive the node-local arena base as one common runtime argument, so the argument count does not depend on the number of logical DFBs.

The arena has two sections:

```text
0                                                   arenaBytes
+----------------------+----------------------------------+
| 8-byte DFB records   | aligned, reusable payload ranges |
+----------------------+----------------------------------+
```

Payload offsets and the start of the payload section are aligned to 32 bytes on Wormhole B0 and 64 bytes on Blackhole. A tensor-to-DFB copy can read from DRAM into these addresses. The pinned TT-Metal revision specifies these [Wormhole B0](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h#L291-L310) and [Blackhole](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h#L374-L394) DRAM-read alignments; its [NoC sanitizer](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/hw/inc/internal/debug/sanitize.h#L513-L538) applies the source alignment to the local L1 destination. The arena base is allocated by TT-Metal's [DRAM-aligned L1 allocator](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/tt_metal/impl/allocator/bank_manager.cpp#L140-L159). A 16-byte L1 alignment alone does not satisfy DRAM-to-L1 reads.

Each logical DFB owns one 8-byte record for the duration of the operation execution. The first 32-bit word is the published-block sequence and the second is the consumed-block sequence. Separate words allow the producer and consumer to update state without an atomic read-modify-write operation.

Payload storage can overlap when the compiler proves that the corresponding lifetimes cannot be active concurrently. Control records do not overlap because payload completion does not prove that sequence state can change ownership.

For `N` logical DFBs and target DRAM-read alignment `A`, the payload section begins at:

```text
controlEnd = roundUp(8 * N, A)
```

For a tiled DFB with page size `P`, pages per block `T`, and block count `B`, its payload extent is:

```text
extent = roundUp(P * T * B, A)
```

Packed-format metadata is included in `P`. The complete arena size is the maximum assigned payload end. Empty programs allocate no arena.

## Conflict Analysis

Allocation consumes the existing logical-identity and completion-aware lifetime analyses. The compiler builds the complete conflict relation before changing IR. Unknown worker nodes on which a kernel may execute, unproved completion, concurrent lifetimes, and incompatible storage ownership remain conflicts.

The shared storage conflict analysis includes writes from Metal descriptor installation and incompatible backing ownership. Compiler-managed reset and reconfiguration are rejected before allocation, so an accepted compiler-managed program has no such installation writes.

This design reuses one lifetime model for both memory backends. The allocator cannot serialize operations or remove a conflict to make a program fit.

```text
buildStorageConflicts(lifetimes):
    conflicts = empty graph
    for each unordered pair (left, right):
        for each worker node where both may be active:
            if the node association or completion order is unknown:
                add conflict(left, right)
            else if neither lifetime completes before the other begins:
                add conflict(left, right)
            if descriptor installation can overwrite live state or backing-storage ownership is incompatible on this node:
                add conflict(left, right)

    return conflicts
```

When the compiler knows only a set of possible worker nodes, it applies the same rule to every member. It authorizes overlap only when every common worker node has a proven completion order.

## Placement Interface and Algorithms

Placement is a reusable C++ library that does not inspect compiler IR or target-specific code. Conflict analysis produces an immutable allocation problem containing each payload extent, a symmetric conflict matrix, target alignment, the payload base after control records, and the SRAM budget. An allocator returns one byte offset per payload and the arena high-water mark.

Every allocator result passes the same validation before IR mutation. Validation requires the correct offset count, target alignment, offsets at or above the payload base, intervals within the SRAM budget, disjoint intervals for every conflict, and an exact arena high-water mark. Allocation policy cannot weaken these invariants.

### C++ Allocator Contract

The interface is declared in [CompilerL1Allocator.h](../../lib/Dialect/TTL/Transforms/CompilerL1Allocator.h). Its input, output, and failure fields are:

| Type | Fields and meaning |
| --- | --- |
| `CompilerL1AllocationProblem` | `regionBytes`: aligned, nonzero payload extents; `conflicts`: symmetric matrix of pairs that must not overlap; `alignmentBytes`: target alignment; `payloadBaseOffset`: first usable payload byte after control records; `budgetBytes`: usable L1 capacity. |
| `CompilerL1AllocationSolution` | `offsets`: one arena-relative byte offset per input region; `arenaBytes`: exact maximum payload end, or zero for empty input. |
| `SRAMPlacementFailure` | `kind`: invalid problem, strategy failure, invalid solution, or exhausted budget; `regionIndex`: optional failing region; `reason`: diagnostic text. |

Strategies implement the following C++ contract:

```cpp
namespace mlir::tt::ttl {

class CompilerL1Allocator {
public:
  virtual ~CompilerL1Allocator() = default;
  virtual llvm::StringRef getName() const = 0;

private:
  virtual FailureOr<CompilerL1AllocationSolution>
  allocate(const CompilerL1AllocationProblem &problem,
           std::string &failureReason) const = 0;
};

FailureOr<std::unique_ptr<CompilerL1Allocator>>
createCompilerL1Allocator(llvm::StringRef name, std::string &failureReason);

FailureOr<CompilerL1AllocationSolution> solveCompilerL1Allocation(
    const CompilerL1Allocator &allocator,
    const CompilerL1AllocationProblem &problem,
    SRAMPlacementFailure &failureDetail);

} // namespace mlir::tt::ttl
```

Region vector order defines region indices and deterministic equal-size ordering. The conflict matrix has a clear diagonal; `payloadBaseOffset` is aligned and does not exceed `budgetBytes`. The problem is passed as `const` after construction.

`solveCompilerL1Allocation` is the only caller of the private strategy method. It validates the problem, invokes the selected strategy, and validates the solution. On success, `offsets` has one entry per region and `arenaBytes` is the exact maximum payload end, or zero when no regions exist. On failure, `reason` contains diagnostic text; `regionIndex` identifies a region only when the error applies to one region, and `kind` distinguishes budget exhaustion from invalid input, strategy failure, and invalid output. The allocator layer does not emit diagnostics or modify IR.

`createCompilerL1Allocator` maps stable compiler-option names to implementations. A new implementation derives from `CompilerL1Allocator`, implements `getName()` and `allocate()`, and registers its name in the factory. It cannot change conflict construction or bypass common validation.

| Strategy | Gap selection | Use |
| --- | --- | --- |
| `first-fit-decreasing` | Lowest aligned legal offset | Default; preserves deterministic low-address placement. |
| `best-fit-decreasing` | Finite legal gap with the least unused space; lower offset resolves ties | Reduces fragmentation when differently sized lifetimes leave reusable gaps. |

Both strategies place larger regions first because large extents have fewer usable gaps. Declaration order resolves equal-size ties. Both are greedy heuristics and can produce different arena sizes; neither proves optimality.

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

Placement takes `O(N^2 log N)` time after conflict construction and uses `O(N)` placement storage. Conflict adjacency uses `O(N^2)` bits. A budget failure reports that the selected strategy failed; it does not claim that no feasible placement exists.

The allocator interface contains no MLIR operations, DFB identities, architecture identities, or target branches. It receives normalized alignment and budget values through the allocation problem. Adding a strategy requires an implementation of the placement interface and a stable factory name. Conflict construction, target queries, validation, metadata emission, and runtime allocation remain unchanged.

## External C++ Interface

`ttl.dfb_descriptor(dfb)` lowers to a C++ template type containing page size, pages per block, block count, state offset, and the payload offset relative to the state record. Its `bind()` method obtains the arena base through the target interface and constructs the address-based buffer. Generated descriptor definitions precede external C++ headers that use them. External functions require no Metal DFB index or additional runtime argument per DFB.

```text
bind(descriptor):
    stateAddress = target.arenaBase() + descriptor.stateOffset
    return AddressDFB(stateAddress, descriptor.payloadOffset, descriptor.geometry)
```

External calls that access DFBs must provide explicit `DFBEffect` entries. These effects participate in lifetime and conflict analysis. Unknown DFB access and numeric `dfb_index` template arguments are rejected for compiler-managed storage.

## Target Interfaces

Common allocation and lowering contain no architecture branches. `compiler_l1_target.h` provides arena-base access, SRAM loads and stores, producer/consumer completion, and processor ownership. `compiler_l1_compute.h` implements address-based compute operations; `compiler_l1_compute_target.h` adapts their LLK calls to Wormhole and Blackhole signatures.

## Runtime Arena

For each invocation with a nonempty allocation plan, the runtime allocates a zero-initialized arena as a row-major, height-sharded TTNN L1 tensor with one equal-length row per participating worker node. Height sharding directly represents one arena row per node. Width sharding provides no capacity benefit, and block sharding introduces an unused partition dimension.

The runtime passes the arena as an auxiliary `generic_op` input without changing the user output position. Each invocation waits for device completion before releasing its arena, including after descriptor preparation or dispatch fails. A synchronization failure retains the arena for the process lifetime because completion is unknown. This wait adds host latency to each invocation with a nonempty allocation plan.

Finalization records `ttl.memory_model`, `ttl.l1_arena_bytes`, and one entry per logical DFB in `ttl.dfb_allocations`. Entries are ordered by `dfb_index`, which equals each entry's array position. Each entry gives the arena-relative control-record offset (`l1_offset`), arena-relative payload offset (`l1_payload_offset`), and aligned payload extent (`l1_allocation_bytes`). Before code generation, EmitC checks bounds, target alignment, and agreement between each DFB type and its allocation's element type, page size, and total page count. A generated kernel's compile-time argument 0 identifies the common runtime argument containing its local arena base; subsequent DFB compile-time arguments identify allocation entries. The C++ `PayloadOffset` template parameter is relative to the control record: `l1_payload_offset - l1_offset`.

Uniform allocation reserves the largest required arena on every participating node. This can waste capacity when activity is sparse. Per-node layouts require node-specific allocation metadata and are an extension of this design.

## Memory Utilization

Storage efficiency comes from four decisions:

1. Completion-aware conflicts permit payload overlap across sequential lifetimes and formats.
2. Both allocation strategies search aligned gaps instead of using a monotonic offset.
3. Payloads are ordered by decreasing size to reduce fragmentation from early small placements.
4. The arena uses one runtime argument, and each logical DFB adds only its fixed control record rather than a Metal descriptor.

The fixed control cost is `roundUp(8 * N, A)`. For 96 BF16 DFBs with one page per block and one block each, simultaneous lifetimes require 196,608 payload bytes and 768 control bytes before final arena alignment. If all 96 lifetimes are sequential, they reuse one 2,048-byte payload range and retain 768 control bytes. The control records are then 27% of the 2,816 bytes before final alignment. Reducing that cost would require shared state ownership transitions or packed atomic updates, both of which add synchronization and target requirements.

Monotonic allocation with explicit execution-phase overlays was considered. It cannot reuse an aligned gap between active allocations and requires explicit phase boundaries. TT-Lang instead uses its completion-aware conflict graph and searches reusable gaps, which permits overlap within a phase and across different extents. Best-fit addresses fragmentation without exponential search. An exact or bounded-search strategy can use the same allocator interface if measurements justify its compile-time cost.

## Implemented Contract

- One device and one uniform worker-node arena.
- Compiler-owned static storage. Tensor-backed DFBs and allocation groups are rejected.
- Full-block transactions with positive capacity below `2^31` pages.
- Full 32x32 BF16 and FP32 tiles for address-based compute.
- Address-based tensor transfer, elementwise compute, matmul, reductions, broadcast, transpose, and loop-carried L1 packer accumulation. SFPU and initializer operations without DFB operands or results use their existing lowering.
- Scalar device printing. Destination-register printing changes pack state on Blackhole; DFB, tile, and tensor printing require physical DFB descriptors. These modes are rejected before lowering.
- Typed external C++ calls with explicit DFB effects.
- Wormhole and Blackhole allocation, transfer, compute, and external descriptors without synchronized reset or reconfiguration.

PipeNet transfers, computed-address DFBs, device-domain placement, multi-device execution, and external runtime resources are outside this contract and are rejected before device execution. The compiler does not fall back to Metal descriptors.

## Validation

[Device tests](../../test/python/test_compiler_l1.py) cover transfer, allocation, reuse, and descriptor-count stress. [Compute tests](../../test/python/test_compiler_l1_compute.py) and [accumulation tests](../../test/python/test_accumulation_strategies.py) cover BF16/FP32, DRAM/SRAM inputs, and typed external calls. [Generated placement tests](../../test/ttlang/Dialect/TTL/Transforms/compiler_l1_stress.py) check alignment, conflicts, budgets, deterministic strategies, and reuse against independent expected placements. Negative compiler tests check unsupported contracts before device execution.

## Extensions

- Per-node and multi-device placement require node-specific layout metadata and ownership for each arena. Multicast receivers additionally require a shared payload address.
- Tensor-backed DFBs and allocation groups require fixed external byte ranges and explicit alias/ownership constraints in the allocation problem.
- PipeNet transfers require completion evidence through destination consumption before scratch ranges can be reused.
- Sub-tile and row-major operations require matching geometry, stride, and capacity rules in the address-based compute interface.
- Synchronized reset and reconfiguration require a processor-wide completion barrier before clearing selected control records and another barrier before subsequent DFB access. Blackhole can use its existing DFB-interface synchronization LLK. Wormhole requires a target synchronization protocol validated on device.

These extensions preserve complete pre-mutation validation, explicit ownership, and descriptor-independent allocation.
