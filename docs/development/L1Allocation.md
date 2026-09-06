# Compiler-Managed L1 Allocation

## Overview

`--ttl-memory-model=compiler-l1` selects experimental byte-addressed storage for
TT-Lang logical dataflow buffers. Python declarations and acquire/release
semantics remain unchanged. The compiler assigns offsets within an allocator-owned
L1 arena. Generated transfer kernels access addresses and ordinary L1 control
words instead of TT-Metal dataflow-buffer descriptors, format tables, or indexed
publication counters. `metal-cb` remains the default memory model.

This document describes the implemented allocation and transfer subset. Compute
integration through address-based low-level kernels (LLKs), PipeNet transport,
and fused-model qualification remain separate work. Selecting `compiler-l1` does
not permit fallback to a Metal descriptor when an operation lacks a lowering.

```text
Logical DFB declarations and acquire/release operations
                         |
                         v
          Logical identity and lifetime analysis
                         |
                         v
       Storage interference graph and byte sizes
                         |
                         v
           Immutable aligned allocation plan
                         |
             +-----------+-----------+
             |                       |
             v                       v
   Runtime allocation metadata    Address-based kernel code
             |                       |
             +-----------+-----------+
                         |
                         v
             One allocator-owned L1 arena
```

Related documents describe [DFB ownership and synchronization](DFBManagement.md),
[compute configuration](ComputeKernelConfiguration.md), and
[static execution analysis](StaticExecutionAnalysis.md). Public option syntax is
specified in the [compiler option reference](../sphinx/reference/compiler-options.md).

## Definitions

The intermediate representation (IR) is the compiler's typed representation of a
program. Static single assignment (SSA) gives each IR value one definition; it
expresses value dependencies but does not by itself describe device completion.

| Term | Definition |
|---|---|
| NoC | Network-on-chip hardware used by transfer processors to read and write device memory. |
| RISC | A reduced-instruction-set controller on a worker core. Transfer processors and compute-engine controllers execute separate instruction streams. |
| BF16 / FP32 | BFloat16 and 32-bit floating-point element encodings, respectively. Their standard 32-by-32 tiles occupy 2048 and 4096 bytes. |
| L1 | Worker-local static random-access memory, also called SRAM. Different worker cores have separate L1 address spaces. |
| Logical DFB | A source-level dataflow buffer with a module-wide identity, element type, pages per block, and block count. It specifies communication semantics rather than a Metal descriptor slot. |
| Region ordinal | A dense compiler identifier for a logical DFB. Existing intermediate representation fields named `cb_index` carry this identifier in the experimental backend. It is not passed to a Metal DFB API. |
| Page | The contiguous storage unit used by a transfer. For tiled storage, its byte size includes the physical tile encoding, including exponent metadata for packed formats. |
| Capacity | The number of pages held by a logical DFB: pages per block multiplied by block count. |
| Payload | The bytes that hold a logical DFB's data, excluding synchronization state and alignment padding. |
| Control record | Four 32-bit L1 words owned by one logical DFB: published page count, consumed page count, write position, and read position. |
| Launch node | A compiler execution-grid coordinate identifying a participating worker core. Its domain is the set of nodes on which a logical DFB may be accessed. |
| Acquisition lifetime | The period during which an acquired producer or consumer view can access storage, including outstanding asynchronous accesses. |
| Interference | A pairwise restriction that forbids two payload regions from overlapping in L1. |
| Arena | A runtime-owned L1 tensor containing all control records and compiler-assigned payload ranges. |
| High-water mark | The largest exclusive end offset occupied in the arena. It determines the required bytes per shard. |
| Alignment quantum | The target-defined byte multiple required for region placement and backing allocation. The shared target query supplies this value. |

A half-open byte interval `[start, end)` includes `start` and excludes `end`.
Intervals with one end equal to the other's start do not overlap. All offsets in
the compiler/runtime allocation interface use bytes, not LLK address units.

## Assumptions and Supported Subset

The implementation relies on the following contracts:

1. Each logical DFB has one producer and one consumer on each active launch node.
   Existing SPSC verification remains enabled. SPSC means single producer and
   single consumer; it does not require those participants to be the same RISC.
2. Declarations of the same logical identity have identical types and capacities.
   Tensor-backed regions and explicit allocation groups are rejected by the byte
   planner. Storage reconfiguration is not implemented by this backend.
3. The compiler's existing lifetime analysis supplies completion-aware ordering.
   Source adjacency, region ordinals, and ordinary SSA use order are not evidence
   that an asynchronous access has completed.
4. Unknown ordering is conservative: the affected payloads interfere. Unknown
   and unreachable execution are distinct analysis results.
5. The runtime launches one device program over the supplied core grid. The POC
   rejects device-domain placement and external runtime resources. Uniform
   height-sharded allocation reserves the same arena size on each participating
   core; this can waste bytes on sparsely active cores.
6. Each invocation receives a new, host-initialized arena. The runtime associates
   that arena with the program by passing it as an I/O tensor to `generic_op`.
   Trace capture/replay and independently scheduled command queues are not yet
   qualified by this POC.
7. Every reserve, wait, push, and pop processes one complete statically sized
   block. The converter rejects other page counts. Capacity is a whole number of
   such blocks, so an acquisition cannot cross the ring end. Runtime assertions
   also check bounds. Partial or wrapped acquisitions require a more general view
   and cursor proof and are not implemented by this subset.
8. Capacity is positive and less than `2^31` pages. The outstanding published
   count never exceeds capacity. Publication and consumption counters use
   unsigned 32-bit arithmetic; ring positions are stored separately.
9. The executable operation subset consists of tiled tensor transfers through
   local storage, their synchronization operations, and required address/accessor
   operations. The conversion pass maintains an explicit supported-operation
   check before rewriting. Pre-lowered C++ is also rejected because its effects
   have not been checked against the address-based storage contract. This includes
   instrumentation that requires such code. Compute operations, PipeNet, legacy
   external kernels, and interface reset/reconfiguration are rejected.
10. The transfer tests qualify BF16 and FP32 tiled tensors in interleaved DRAM and L1. Other
    formats, layouts, and memory configurations are not qualified by those tests.
    Wormhole and Blackhole target helper code does not by itself establish device
    qualification on both architectures.

The available arena budget comes from the existing target/runtime L1 budget
contract or an explicit compiler override. The allocator cannot infer arbitrary
allocations made outside that contract. The runtime allocator remains responsible
for fitting the arena alongside live tensors and its own reservations. An explicit
budget override is a caller-supplied limit, not additional device memory.

## Analysis and Planning

### Logical Regions and Sizes

The planner consumes
[`DFBLogicalIdentityAnalysis`](../../include/ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h)
and
[`DFBConcurrentKernelLivenessAnalysis`](../../lib/Dialect/TTL/Transforms/DFBConcurrentKernelLivenessAnalysis.h).
It validates all declarations and arithmetic before changing IR. The allocation
plan stores logical identities, types, byte sizes, offsets, and exact declarations
to update. Application executes these decisions; it does not rerun placement.

```text
collectRegions(module, identityAnalysis):
    regions = empty insertion-ordered map from logical identity to region

    for assignment in identityAnalysis, in IR order:
        declaration = assignment.declaration
        identity = assignment.logicalIdentity
        reject tensor backing or explicit allocation group

        if identity already has a region:
            require declaration.type == regions[identity].type
            append declaration to regions[identity].declarations
            continue

        pagesPerBlock = physical pages in declaration's block type
        pageBytes = physical encoded bytes per page
        capacity = pagesPerBlock * blockCount
        payloadBytes = capacity * pageBytes
        require all sizes to fit the compiler/runtime representation
        allocationBytes = roundUp(payloadBytes, targetAlignment)

        insert region(identity, type, pagesPerBlock, pageBytes,
                      allocationBytes, declarations=[declaration])

    return regions in insertion order
```

The existing type utilities compute page geometry and encoded size. Allocation
must not substitute logical element count times scalar width for a packed tile's
physical size. See
[`getDFBPageSizeBytes`](../../include/ttlang/Dialect/TTL/IR/TTLOpsUtils.h) and
[`getDFBAllocationSizeBytes`](../../lib/Dialect/TTL/Transforms/DFBAllocationLimits.cpp).

### Storage Interference

The byte planner reuses `DFBPhysicalConflictModel::buildStorage` from
[`DFBPhysicalAllocationPlan.cpp`](../../lib/Dialect/TTL/Transforms/DFBPhysicalAllocationPlan.cpp).
This analysis already separates storage legality from Metal descriptor sharing.
It does not require matching formats, exact descriptors, transaction counts, or
pointer owners. It retains completion, lifetime, launch-domain, and static storage
ownership restrictions.

The following pseudocode defines the planner's use of that analysis. The referenced
analysis implements the detailed control-flow and completion proofs.

```text
buildInterference(lifetimeAnalysis):
    requirements = storage compatibility requirements:
        exact descriptor matching = false
        matching element types = false
        matching transaction counts = false
        matching pointer owners = false
        static storage ownership = true
    graph = empty symmetric interference graph

    for every distinct pair of logical regions:
        evidence = existing completion-aware pair analysis(
            lifetimeAnalysis, firstRegion, secondRegion, requirements)
        if evidence contains a storage conflict:
            add an interference edge and retain its diagnostic evidence

    return graph, indexed by lifetimeAnalysis logical identities
```

The graph represents conservative restrictions, not a schedule. Removing an edge
requires a proof; placement cannot remove an edge to satisfy a memory budget.
A pointer owner is the processor that advances a read or write cursor.
Control records are excluded from payload sharing and retain distinct addresses
for the entire invocation.

### Aligned Byte Placement

The alignment query is defined in
[`TargetInfo.h`](../../include/ttlang/Target/TargetInfo.h). Architecture-specific
values remain in that target interface. The allocator contains no architecture
switch.

For `regionCount` logical regions, the permanent control prefix occupies
`roundUp(regionCount * 16, alignment)` bytes. Region ordinals select consecutive
16-byte records within this prefix.

```text
placeRegions(regions, interference, alignment, reuseEnabled, budget):
    controlBytes = roundUp(length(regions) * 16, alignment)
    order = stable sort of region ordinals by decreasing allocationBytes
    placed = empty list

    for regionOrdinal in order:
        region = regions[regionOrdinal]
        region.stateOffset = regionOrdinal * 16
        blockers = previously placed regions that interfere with region
        if reuseEnabled is false:
            blockers = every previously placed region
        sort blockers by (payloadOffset, regionOrdinal)

        candidateOffset = controlBytes
        for blocker in blockers:
            if candidateOffset + region.allocationBytes <= blocker.payloadOffset:
                break
            if candidateOffset < blocker.payloadOffset + blocker.allocationBytes:
                candidateOffset = roundUp(
                    blocker.payloadOffset + blocker.allocationBytes, alignment)

        require candidateOffset + region.allocationBytes <= budget
        region.payloadOffset = candidateOffset
        append regionOrdinal to placed

    arenaBytes = maximum(region.payloadOffset + region.allocationBytes)
    return immutable plan(regions, controlBytes, arenaBytes)
```

With `regionCount` equal to zero, the arena size is zero. The implementation checks
bounds before offset materialization. A failed placement emits a diagnostic before
any declaration is rewritten.

For `regionCount = N`, gathering and sorting blockers costs at most
`O(N^2 log N)` time after graph construction. The graph's dense adjacency matrix
uses `O(N^2)` bits. Diagnostic evidence additionally grows with the number of
reported per-node conflicts; lifetime-analysis costs are separate. The region plan
and placement lists use `O(N)` entries. These are compiler-time costs, not per-core
runtime metadata.

### Correctness Argument

The control prefix and payload intervals are disjoint because every candidate
starts at or after the rounded control-prefix end. Each control record has a
unique ordinal, so control records are mutually disjoint.

Assume all previously placed interfering pairs are disjoint. The next placement
examines every previously placed blocker in increasing offset order. It either
finds a sufficiently large gap before a blocker or advances past that blocker's
end. The resulting interval overlaps no blocker. Therefore placement preserves
pairwise disjointness by induction. A region may overlap nonblockers because the
analysis has already proved that those regions need not retain different bytes
on a common active core.

The memory planner does not change execution order. Producer/consumer blocking
comes from the existing FIFO semantics, not from artificial serialization added
to make the allocation fit. The current completion helper conservatively waits
for all outstanding NoC work from its processor. It can delay publication behind
unrelated transfers; completion restricted to the acquired storage requires an
additional ownership proof and remains a performance improvement.

### Allocation Quality and Limits

This algorithm uses variable-sized byte intervals rather than equal-sized graph
colors. It can reuse an earlier hole and can share storage across different
formats. It is deterministic, but decreasing-size first-fit is not optimal. A
budget failure reports failure of this placement, not proof that the program
cannot fit.

The current implementation does not compute an exact peak-live-byte lower bound,
try multiple placement orders, or perform exhaustive search. Such extensions must
preserve the same interference relation. A lower bound derived from that relation
must be labeled as a lower bound; a conservative graph need not describe one
simultaneously realizable set of live buffers.

For example, 96 independent one-page BF16 buffers with 2048-byte pages require
196608 payload bytes when simultaneously live. Their control records require
1536 bytes. If the compiler proves that the 96 lifetimes are sequential, payload
placement can use 2048 bytes plus the same 1536-byte control prefix. The corresponding
FP32 payload sizes are 393216 bytes without reuse and 4096 bytes with reuse. These
calculations describe the allocation contract; device tests verify both storage
assignments and numerical results.

## Runtime Binding and Emission

Allocation metadata includes `l1_offset` for the control record,
`l1_payload_offset` for the payload, and `l1_allocation_bytes` for the aligned
payload extent. `ttl.l1_arena_bytes` records the complete per-core allocation.
Existing DFB metadata retains page size, pages per block, block count, and format.

```text
launchWithCompilerL1(plan, tensors, kernelSpecifications, executionGrid):
    validate complete compiler-l1 metadata and supported runtime features
    arena = allocate host-zero-initialized height-sharded L1 tensor(
        executionGrid, plan.arenaBytes)

    for kernel in kernelSpecifications:
        commonArguments = existing tensor and kernel arguments
        arenaArgumentIndex = length(commonArguments)
        append arena.baseAddress to commonArguments
        compileArguments = [arenaArgumentIndex] + existing tensor accessor arguments
        create kernel descriptor from those arguments

    create program descriptor with zero Metal DFB descriptors
    launch generic_op with tensors and arena as its I/O tensor resources
```

Host initialization currently writes the complete arena, although only the
control prefix requires zero initialization. Avoiding that payload initialization
traffic requires a partial initialization mechanism and is not implemented here.

Generated code obtains the runtime arena base through one common argument.
A logical region's control address is `arenaBase + stateOffset`. The storage
object's compile-time parameters contain page bytes, capacity, and the displacement
from control address to payload address. Runtime argument count therefore does not
grow with the number of logical regions.

The emitter folds page-size and format queries from type metadata. It must not
recover those values from Metal DFB tables. The emitted standalone Python runner
preserves the allocation fields and uses the same runtime construction logic.

## Synchronization Algorithms

The storage interface is implemented in
[`compiler_l1.h`](../../include/ttlang/Target/TTKernel/LLKs/compiler_l1.h).
Target visibility and completion operations are isolated in
[`compiler_l1_target.h`](../../include/ttlang/Target/TTKernel/LLKs/compiler_l1_target.h).
The current transfer subset uses NoC completion. Compute integration must add
explicit unpack and pack completion through this interface before enabling those
operations.

### Visibility and Completion

`loadVisible` reads a shared L1 control word using the target's visibility
sequence. `storeVisible` completes the target's store/readback sequence.
`completeAccesses` waits for earlier accesses by the publishing or releasing
processor. A CPU fence alone does not complete NoC, unpack, or pack work.

```text
loadVisible(address):
    execute target visibility fence
    load word from address
    execute target dependent-load completion sequence
    return loaded word

storeVisible(address, value):
    store word to address
    load back from address
    execute target dependent-load completion sequence

completeAccesses():
    data-movement processor: wait for outstanding NoC work
```

### Producer Reservation and Publication

All counter subtraction and addition below use unsigned 32-bit arithmetic.
`writePosition` and `readPosition` are page offsets within `[0, capacity)`.
The producer exclusively updates `published` and `writePosition`.

```text
reserve(pageCount):
    require pageCount <= capacity
    while capacity - (loadVisible(published) - loadVisible(consumed)) < pageCount:
        poll
    require loadVisible(writePosition) + pageCount <= capacity
    return payloadAddress + loadVisible(writePosition) * pageBytes

publish(pageCount):
    completeAccesses()
    nextPosition = (loadVisible(writePosition) + pageCount) modulo capacity
    storeVisible(writePosition, nextPosition)
    storeVisible(published, loadVisible(published) + pageCount)
```

Publication makes a completed payload available. The producer cannot overwrite
unconsumed pages because reservation waits for sufficient capacity.

### Consumer Acquisition and Release

The consumer exclusively updates `consumed` and `readPosition`.

```text
wait(pageCount):
    require pageCount <= capacity
    while loadVisible(published) - loadVisible(consumed) < pageCount:
        poll
    require loadVisible(readPosition) + pageCount <= capacity
    return payloadAddress + loadVisible(readPosition) * pageBytes

release(pageCount):
    completeAccesses()
    nextPosition = (loadVisible(readPosition) + pageCount) modulo capacity
    storeVisible(readPosition, nextPosition)
    storeVisible(consumed, loadVisible(consumed) + pageCount)
```

Release permits reuse only after the consumer's accesses complete. Keeping ring
positions separate from sequence counters is necessary when capacity does not
divide `2^32`: wrapping a sequence counter must not change the ring position.

## Validation

[`test_compiler_l1.py`](../../test/python/test_compiler_l1.py) covers the executable
transfer subset. Tests compare exact BF16/FP32 copies, both memory models, DRAM/L1
tensor storage, repeated invocation with distinct arena addresses, non-power-of-two
ring capacity, 96 live
regions, core specialization enabled and disabled, and sequential payload reuse. Additional cases exercise independent
producer/consumer processors and sequence-counter overflow. Device execution and
inspection of compiler offsets are both required: a correct result alone does
not prove reuse, and small allocation metadata alone does not prove correctness.

The device stress matrix additionally uses four seeded schedules with eight
regions, one to five pages per block, and one to three blocks per region. It
covers one-core and four-core grids, both element types, and reuse enabled and
disabled. Each core transfers distinct tensor slices. The test independently
reconstructs overlapping lifetimes from the generated event sequence and checks
that their byte intervals do not overlap. It also checks payload sizes, control
record separation, alignment, and arena bounds.

[`compiler_l1_stress.py`](../../test/ttlang/Dialect/TTL/Transforms/compiler_l1_stress.py)
checks all 90 valid orderings of three producer/consumer pairs and 32 seeded
four-region schedules. Mixed packed subtiles and full tiles exercise alignment
padding and different allocation sizes. Both architecture descriptions and both
reuse settings produce 492 allocation checks, including unknown-access cases.
A repeated compilation must produce identical output. Expected conflicts come
from the event sequence rather than the compiler's interference graph; peak live
aligned bytes provide an independent lower bound. These checks prove neither
optimal placement nor device support for the packed formats.

MLIR tests also cover allocation metadata, both option values, and unsupported
forms. Exact-budget tests accept the complete arena size and reject one byte
less, including a separate post-allocation budget override. Negative tests are
separate files and verify diagnostics before partial lowering. Hardware
qualification, generated-program inspection, and remaining work are reported in
the POC PR. Host-only tests do not qualify a device architecture.
