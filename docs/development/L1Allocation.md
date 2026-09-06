# Compiler-Managed L1 Allocation

`--ttl-memory-model=compiler-l1` assigns byte offsets to logical DFBs within a
runtime-owned L1 arena. Python declarations and acquisition semantics are
unchanged. Generated transfer kernels use addresses and L1 control words; the
program creates zero Metal DFB descriptors. `metal-cb` remains the default.

Both backends already use completion-aware lifetime analysis to reuse storage.
The difference is the allocated resource and its runtime interface:

| Property | Normal DFB allocator (`metal-cb`) | Byte allocator (`compiler-l1`) |
|---|---|---|
| Assignment | Physical DFB indices plus backing-storage groups. | Payload byte intervals plus distinct control records in one arena. |
| Limiting resources | L1 bytes and architecture-dependent descriptor indices (32 or 64). | L1 bytes, including 16 control bytes per logical region and padding. |
| Reuse | Compatible lifetimes share indices; distinct descriptors can also share backing storage. | Noninterfering payload intervals may overlap; control records remain distinct. |
| Placement | Descriptor and storage conflict graphs drive index/group assignment. | Storage interference drives decreasing-size, aligned first-fit placement. |
| Runtime access | Metal descriptor-indexed pointers, formats, and counters. | Arena base plus constant offsets, compile-time formats, and L1 control words. |
| Execution coverage | Existing compute and transfer backend. | Local tiled transfers in this POC. |

Shared terminology is defined in the [specification glossary](../sphinx/specs/TTLangSpecification.md#appendix-a-glossary).
[DFB Management](DFBManagement.md#compiler-managed-storage-protocol) defines the
acquisition and completion contract. This document covers allocation and binding.

## Allocation Inputs and Contract

The allocator receives module-wide logical identities and completion-aware
lifetimes from [DFBLogicalIdentityAnalysis](../../include/ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h)
and [DFBConcurrentKernelLivenessAnalysis](../../lib/Dialect/TTL/Transforms/DFBConcurrentKernelLivenessAnalysis.h).

| Allocation term | Meaning |
|---|---|
| Region | One logical DFB's payload allocation and permanent control record. |
| Region ordinal | Declaration-order index used to identify a region; it is not a Metal descriptor index. |
| Payload extent | Encoded page bytes multiplied by pages per block and block count, rounded to target alignment. |
| Interference | A restriction requiring two payload intervals to remain disjoint. |
| Arena | One per-invocation allocation containing all regions, with identical offsets on participating cores. |

All offsets and extents use bytes and half-open intervals. Target alignment comes
from [TargetInfo](../../include/ttlang/Target/TargetInfo.h), keeping architecture
selection outside the allocator. Page sizing uses
[getDFBPageSizeBytes](../../include/ttlang/Dialect/TTL/IR/TTLOpsUtils.h) and
[getDFBAllocationSizeBytes](../../lib/Dialect/TTL/Transforms/DFBAllocationLimits.cpp),
including packed-format metadata rather than scalar element widths alone.

The accepted-input contract is:

- Declarations of a logical DFB have identical types and capacities. Storage
  ownership is static. The planner rejects tensor backing, explicit allocation
  groups, resets, and reconfiguration before assigning offsets.
- Lifetimes include asynchronous access completion. Unknown ordering produces
  interference; unreachable execution remains distinct from unknown execution.
- The transfer backend accepts whole-block acquisitions with positive capacity
  below `2^31` pages and the ownership/alternation contract in DFB Management.
  Its operation check rejects unrecognized operations and pre-lowered C++ effects.
- Launches use one device and a uniform height-sharded arena. The runtime rejects
  device-domain placement and external runtime resources. Device qualification
  covers BF16/FP32 tiled transfers with interleaved DRAM/L1 tensors on Blackhole.
  Trace replay and independent command queues have no qualification evidence.

The budget is the existing target/runtime usable-L1 limit or an explicit override.
It includes control records and alignment. The runtime allocator must also fit
this arena alongside live tensors and runtime reservations; an override does not
reserve additional memory.

## Region Collection

[CompilerL1Allocation.cpp](../../lib/Dialect/TTL/Transforms/CompilerL1Allocation.cpp)
builds and validates the complete plan before changing declarations. Collection
retains the declarations to update; application executes that plan without
recomputing placement.

```text
collectRegions(module, identityAnalysis, alignment):
    reject reset or reconfiguration operations
    regions = insertion-ordered map
    for assignment in identityAnalysis, in declaration order:
        declaration = assignment.declaration
        reject tensor backing or explicit allocation group
        if assignment.logicalIdentity is already in regions:
            require identical declaration type and capacity
            append declaration to the existing region
        else:
            pages = physical pages per block
            pageBytes = encoded bytes per page
            payloadBytes = pages * pageBytes * blockCount
            require sizes to fit the compiler/runtime integer representation
            insert region with allocationBytes = roundUp(payloadBytes, alignment)
    return regions
```

## Storage Interference

The allocator uses `DFBPhysicalConflictModel::buildStorage` in
[DFBPhysicalAllocationPlan.cpp](../../lib/Dialect/TTL/Transforms/DFBPhysicalAllocationPlan.cpp).
It retains completion, lifetime, launch-domain, and static-ownership restrictions
while allowing sharing across formats, descriptor configurations, transaction
counts, and pointer owners. Pointer ownership denotes the processor advancing a
read or write cursor.

```text
buildInterference(lifetimeAnalysis):
    requirements = {
        requireExactDescriptor: false,
        requireMatchingElementType: false,
        requireMatchingTransactions: false,
        requireMatchingPointerOwners: false,
        requireStaticStorageOwnership: true
    }
    graph = empty symmetric graph
    for each pair of logical regions:
        evidence = analyze storage compatibility using requirements
        if evidence contains a conflict:
            add interference edge and retain diagnostic evidence
    return graph
```

Placement never removes an interference edge to satisfy a budget. The planner
rejects state-resetting boundaries before constructing the graph, preserving
static control-state ownership independently of the shared analysis.

## Aligned Placement

Each region retains a distinct 16-byte control record. Payloads may alias when
there is no interference edge. Placement uses decreasing extent with declaration
order as the tie-breaker.

```text
placeRegions(regions, interference, alignment, reuseEnabled, budget):
    controlBytes = roundUp(length(regions) * 16, alignment)
    order = stable sort of regions by decreasing allocationBytes
    placed = empty list
    for region in order:
        region.stateOffset = region.ordinal * 16
        blockers = placed regions that interfere with region
        if reuseEnabled is false:
            blockers = all placed regions
        sort blockers by (payloadOffset, ordinal)
        offset = controlBytes
        for blocker in blockers:
            if offset + region.allocationBytes <= blocker.payloadOffset:
                break
            offset = max(offset, roundUp(blocker.end, alignment))
        require offset + region.allocationBytes <= budget
        region.payloadOffset = offset
        append region to placed
    arenaBytes = maximum region.end, or zero for no regions
    return immutable plan(regions, arenaBytes)
```

Every candidate starts after the control prefix. Assuming previously placed
interfering intervals are disjoint, the next placement either fits before a
blocker or advances beyond it. It therefore preserves disjointness by induction.
Overlap with nonblockers is permitted by the lifetime analysis. Placement does
not change execution order.

After graph construction, placement costs `O(N^2 log N)` time and `O(N)` plan
entries for `N` regions. Graph adjacency uses `O(N^2)` bits; per-node diagnostic
evidence and lifetime analysis have additional costs.

Decreasing-size first-fit is deterministic but not optimal. Budget failure means
this placement failed, not that no allocation can fit. The implementation does
not search alternative orders or compute an optimality bound.

Control records impose a fixed `roundUp(N * 16, alignment)` cost regardless of
payload reuse. For 96 one-page BF16 regions, simultaneous lifetimes require
196608 payload bytes plus 1536 control bytes. Sequential lifetimes use 2048
payload bytes plus the same control prefix: control state is approximately 43%
of that arena. Many small, noninterfering regions can therefore be dominated by
control storage even with optimal payload placement.

## Runtime Binding

The allocation metadata uses `l1_offset` for the control record,
`l1_payload_offset` for the payload, and `l1_allocation_bytes` for its aligned
extent. `ttl.l1_arena_bytes` records total bytes per core.
[Kernel runner](../../python/ttl/kernel_runner.py) binds the plan:

```text
launch(plan, tensors, kernels, grid):
    if plan has no regions:
        launch without an arena
        return
    validate metadata and launch contract
    arena = host-zero-initialized, height-sharded L1 tensor(grid, plan.arenaBytes)
    for kernel in kernels:
        append arena.baseAddress to kernel.commonArguments
        prefix kernel.compileArguments with that common-argument index
    launch generic_op with tensors and arena, using zero Metal DFB descriptors
```

Generated code adds constant region offsets to the arena base. Page sizes and
formats come from type metadata, so neither runtime argument count nor Metal
index usage grows with the number of regions. The standalone runner preserves
these metadata fields and uses the same binding implementation.

Each invocation receives a new arena retained through `generic_op` tensor
ownership. Uniform sharding reserves the complete arena on every participating
core, including sparsely active cores. Host initialization currently clears the
payload as well as the control prefix.

## Validation

[Device tests](../../test/python/test_compiler_l1.py) check results and allocation
metadata across buffer-count, size, lifetime-overlap, reuse, specialization,
multi-core, cached-launch, and counter-wrap scenarios.
[Compiler stress tests](../../test/ttlang/Dialect/TTL/Transforms/compiler_l1_stress.py)
check independent schedule-derived conflicts, aligned extents, live-byte lower
bounds, unknown effects, and deterministic repetition on both target descriptions.
The adjacent MLIR tests cover exact budget boundaries and rejected inputs.
Compiler-only packed-format tests do not establish device support. Execution
results and exact revisions are recorded in the PR.

## Future Work

Compute execution requires address-based unpack, pack, and format-configuration
adapters behind the common target interface. Their completion rules must preserve
storage lifetimes across the compute engines, and device tests must establish
correctness independently on each architecture.

Reconfiguration requires an explicit transition for control-state ownership,
initialization, and outstanding-access completion. That transition must be proven
before allowing storage reuse across configuration epochs; enabling the shared
analysis's epoch-sharing permission alone is insufficient.
