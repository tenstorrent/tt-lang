# Compiler-Managed L1 Allocation

## Purpose

TT-Lang normally assigns each logical dataflow buffer (DFB) a TT-Metal DFB descriptor. Wormhole B0 provides 32 descriptor indices and Blackhole provides 64. A program can therefore exhaust descriptor indices while sufficient L1 storage remains.

`--ttl-memory-model=compiler-l1` replaces descriptor-indexed storage with compiler-assigned L1 byte ranges. The Python DFB API and its producer/consumer semantics remain unchanged. `metal-cb` remains the default.

| Property | `metal-cb` | `compiler-l1` |
| --- | --- | --- |
| Allocated identity | TT-Metal DFB index | Compiler allocation index used only in compiler metadata |
| Storage address | TT-Metal descriptor | Arena base plus compiler-assigned byte offset |
| Capacity limit | L1 capacity and 32 or 64 descriptor indices | L1 capacity, control records, and alignment |
| Payload reuse | Requires the Metal descriptor and backing-storage contracts | Requires noninterfering completed lifetimes |
| Producer/consumer state | TT-Metal DFB interface state | Two 32-bit sequence counters per logical DFB |
| Reset and reconfiguration in this PR | Blackhole TT-Metal interface reset and runtime descriptor reconfiguration | Blackhole address-based state reset with fixed geometry |
| External C++ DFB access | Numeric index or descriptor metadata | Typed address-bearing descriptor |

Shared terminology is defined in the [TT-Lang specification glossary](../sphinx/specs/TTLangSpecification.md#appendix-a-glossary). The DFB protocol and lifecycle rules are defined in [DFB Management](DFBManagement.md).

## Allocation Model

One compiler-managed arena exists on each participating worker core for each execution of a compiled Python `ttl.operation`. Every arena uses the same relative layout. Kernels receive the core-local arena base as one common runtime argument, so the argument count does not depend on the number of logical DFBs.

The arena has two sections:

```text
0                                                   arenaBytes
+----------------------+----------------------------------+
| 8-byte DFB records   | aligned, reusable payload ranges |
+----------------------+----------------------------------+
```

Each logical DFB owns one 8-byte record for the duration of the operation execution. The first 32-bit word is the published-block sequence and the second is the consumed-block sequence. Separate words allow the producer and consumer to update state without an atomic read-modify-write operation.

Payload storage can overlap when the compiler proves that the corresponding lifetimes cannot be active concurrently. Control records do not overlap because payload completion does not prove that sequence state can change ownership.

For `N` logical DFBs and target alignment `A`, the payload section begins at:

```text
controlEnd = roundUp(8 * N, A)
```

For a DFB with page size `P`, pages per block `T`, and block count `B`, its payload extent is:

```text
extent = roundUp(P * T * B, A)
```

Packed-format metadata is included in `P`. The complete arena size is the maximum assigned payload end. Empty programs allocate no arena.

## Conflict Analysis

Allocation consumes the existing logical-identity and completion-aware lifetime analyses. The compiler builds the complete conflict relation before changing IR. Unknown launch domains, unproved completion, concurrent lifetimes, and incompatible storage ownership remain conflicts.

The shared storage conflict analysis accepts an explicit storage mode. Metal storage includes conflicts caused by runtime descriptor installation and Metal-managed backing changes. Compiler-managed storage excludes those conflicts because every logical DFB retains fixed compile-time geometry and an independent control record. This distinction permits byte reuse across a reconfiguration boundary after the prior lifecycle ends while preserving DFBs that remain live across the boundary.

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

Possible launch domains use the same rules as exact domains and remain conservative. The compiler authorizes overlap only when every common worker core has a proven completion order.

## Placement Algorithm

The allocator uses deterministic first-fit decreasing placement. Large regions are placed first because they have fewer usable gaps. First-fit selects the lowest aligned gap that does not overlap an already placed conflicting region. Declaration order resolves equal-size ties.

```text
allocate(regions, conflicts, alignment, budget, reuseEnabled):
    controlEnd = roundUp(8 * count(regions), alignment)
    placementOrder = stableSort(regions, decreasing extent)
    placed = empty list

    for region in placementOrder:
        if reuseEnabled:
            blockers = placed regions that conflict with region
        else:
            blockers = placed
        blockers = sort(blockers, increasing payload start)

        candidate = controlEnd
        for blocker in blockers:
            if candidate + region.extent <= blocker.start:
                break
            if candidate < blocker.end:
                candidate = roundUp(blocker.end, alignment)

        if candidate + region.extent > budget:
            fail before modifying IR

        region.stateOffset = 8 * region.allocationIndex
        region.payload = [candidate, candidate + region.extent)
        append region to placed

    arenaBytes = maximum payload end, or zero for an empty plan
    return all offsets and arenaBytes
```

For each new region, the scan either finds a sufficient gap or advances beyond every overlapping blocker. The assigned interval therefore overlaps no conflicting interval. Applying this argument in placement order proves disjoint storage for every conflict edge. All other overlap is authorized by the lifetime analysis.

Placement takes `O(N^2 log N)` time after conflict construction and uses `O(N)` placement storage. Conflict adjacency uses `O(N^2)` bits. The algorithm does not prove an optimal arena size. A budget failure reports that greedy placement failed; it does not claim that no feasible placement exists.

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

`ttl.dfb_descriptor(dfb)` lowers to a C++ template type containing page size, pages per block, block count, state offset, and the payload offset relative to the state record. Its `bind()` method obtains the arena base through the target interface and constructs the address-based buffer. External functions therefore require no Metal DFB index and no additional runtime argument per DFB.

```text
bind(descriptor):
    stateAddress = target.arenaBase() + descriptor.stateOffset
    return AddressDFB(stateAddress, descriptor.payloadOffset, descriptor.geometry)
```

External calls that access DFBs must provide explicit `DFBEffect` entries. These effects participate in lifetime and conflict analysis. Unknown DFB access and numeric `dfb_index` template arguments are rejected for compiler-managed storage.

## Target Interfaces

Common allocation and lowering contain no architecture branches. `compiler_l1_target.h` provides arena-base access, L1 loads and stores, producer/consumer completion, and processor ownership. `compiler_l1_compute_target.h` provides LLK address conversion, format configuration, and address-based compute operations. Wormhole and Blackhole differences remain inside these target interfaces.

## Runtime Arena

The runtime allocates the arena as a row-major, height-sharded TTNN L1 tensor with one equal-length row per participating worker core. Height sharding directly represents one arena row per core. Width sharding provides no capacity benefit, and block sharding introduces an unused partition dimension.

The arena is passed as an auxiliary `generic_op` input so TTNN retains it through device execution while preserving the user output position. Arena and synchronization scratch are zero-initialized. Runtime resource caching includes the allocation metadata and reset count, so incompatible layouts do not share resources.

Uniform allocation reserves the largest required arena on every participating core. This can waste capacity when activity is sparse. Per-core layouts require target-independent per-node allocation metadata and are follow-on work.

## Memory Utilization

Storage efficiency comes from four decisions:

1. Completion-aware conflicts permit payload overlap across sequential lifetimes, formats, and reconfiguration epochs.
2. First-fit searches aligned gaps instead of using a monotonic offset.
3. Payloads are ordered by decreasing size to reduce fragmentation from early small placements.
4. The arena uses one runtime argument, and each logical DFB adds only its fixed control record rather than a Metal descriptor.

The fixed control cost is `roundUp(8 * N, A)`. For 96 one-page BF16 DFBs, simultaneous lifetimes require 196,608 payload bytes and 768 control bytes before final arena alignment. If all 96 lifetimes are sequential, they reuse one 2,048-byte payload range and retain 768 control bytes. The control records are then 27% of the 2,816 bytes before final alignment. Reducing that cost would require shared state ownership transitions or packed atomic updates, both of which add synchronization and target requirements.

Monotonic allocation with explicit execution-phase overlays was considered. It cannot reuse an aligned gap between active allocations and requires explicit phase boundaries. TT-Lang instead uses its completion-aware conflict graph and searches reusable gaps, which permits overlap within a phase and across different extents. More expensive exact or bounded search is justified only if measurements show material first-fit fragmentation.

## Implemented Contract

- One device and one uniform worker-core arena.
- Compiler-owned static storage. Tensor-backed DFBs and allocation groups are rejected.
- Full-block transactions with positive capacity below `2^31` pages.
- Full 32x32 BF16 and FP32 tiles for address-based compute.
- Address-based tensor transfer, elementwise compute, matmul, reductions, broadcast, transpose, and selected activation operations covered by the implementation tests.
- Typed external C++ calls with explicit DFB effects.
- Blackhole selected reset, reset-all, and reconfiguration.
- Wormhole allocation, transfer, compute, and external descriptors without reset or reconfiguration.

PipeNet transfers, computed-address DFBs, device-domain placement, multi-device execution, and external runtime resources are outside this contract and are rejected before device execution. The compiler does not fall back to Metal descriptors.

## Validation

| Scenario | Evidence |
| --- | --- |
| Blackhole transfer and compute | Device correctness across BF16/FP32, DRAM/L1 tensors, repeated executions, counter wraparound, 96 live DFBs, arithmetic with 66 allocated DFBs, matmul, reductions, residual, MLP, attention, and expert merge |
| External calls and lifecycle boundaries | 20 Blackhole device cases across BF16/FP32 and DRAM/L1, including repeated selected reset, reset-all, reconfiguration, live state preservation, payload reuse, and reset of allocation index 65 |
| Allocation | 10,444 compile-only generated placements covering conflicts, alignment, reuse enabled and disabled, determinism, and exact budget boundaries |
| Wormhole | Compile-only allocation, typed external descriptor, and UNPACK/MATH/PACK target compilation; negative reset and reconfiguration diagnostics |
| Invalid contracts | Compiler diagnostics for malformed metadata, unsupported transactions and tile forms, unknown external effects, numeric external DFB indices, storage ownership, and budget overflow |

Relevant tests are [transfer and allocator device tests](../../test/python/test_compiler_l1.py), [compute device tests](../../test/python/test_compiler_l1_compute.py), [lifecycle and external-call device tests](../../test/python/test_compiler_l1_lifecycle.py), and [generated allocator stress tests](../../test/ttlang/Dialect/TTL/Transforms/compiler_l1_stress.py).

## Follow-on PRs

The intended dependency order after this POC is:

1. Add multi-device arena ownership, device-domain placement, mesh program placement, and external runtime-resource composition. Preserve one validated arena layout per participating device and core.
2. Add tensor-backed DFB and allocation-group ownership. Represent external byte ranges, aliasing, and synchronized ownership transitions in the immutable allocation plan.
3. Add PipeNet and computed-address transfers. Extend lifetime completion through remote transfer and destination consumption before permitting payload reuse.
4. Qualify representative external C++ kernels against the typed descriptor interface and add common adapters for required address, geometry, and completion operations.
5. Add sub-tile and row-major metadata, partial-block transactions, and the corresponding address and capacity rules.
6. Add Wormhole reset and reconfiguration after defining and device-qualifying a Wormhole synchronization protocol behind the existing target interface.
7. Qualify complete model layers, then measure device cycles, arena high-water usage, initialization cost, compile time, and generated code size against `metal-cb`.

Each extension must preserve the fail-before-mutation rule, architecture isolation, explicit ownership, and compiler-managed descriptor independence.
