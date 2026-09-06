# Compiler-Managed L1 Allocation

## Problem and Approach

A program can exhaust Metal DFB indices while its data still fits in L1. Reusing
indices helps sequential lifetimes, but cannot accommodate more simultaneously
live, distinct descriptors than the hardware supports.

`--ttl-memory-model=compiler-l1` removes that constraint by assigning byte ranges
rather than Metal DFB indices. Logical DFBs retain their Python acquisition and
release semantics. The compiler determines storage overlap; the runtime allocates
the memory and supplies its base address.

| Property | Normal allocator (`metal-cb`) | Byte allocator (`compiler-l1`) |
|---|---|---|
| Assigned resources | Metal DFB indices and backing-storage groups. | Payload byte ranges and per-region control records. |
| Limits | L1 capacity and 32 or 64 descriptor indices. | L1 capacity, including control records and alignment. |
| Reuse | Compatible lifetimes share indices; distinct descriptors can also share backing storage. | Noninterfering payload ranges overlap; control records remain distinct. |
| Runtime interface | Metal descriptor-indexed pointers, formats, and counters. | Arena-relative addresses, compile-time formats, and L1 control words. |
| Execution coverage | Existing compute and transfer backend. | BF16/FP32 full-tile tensor transfers, arithmetic, matmul, reductions, broadcast, and selected activations. |

Both allocators already use completion-aware lifetime analysis. Storage reuse is
not new; independence from the Metal descriptor interface is the change.
Here, a full-tile transfer copies complete encoded tiles between a TTNN tensor
and an arena-backed DFB on the same device.

## Design Decisions

### Separate storage lifetime from descriptor compatibility

Two payloads need different bytes only while their lifetimes interfere. Matching
formats, descriptor indices, or pointer owners is not necessary for byte reuse.
The allocator therefore reuses the existing storage-interference analysis,
without the stricter descriptor-sharing requirements of normal DFB allocation.

This preserves the established completion and launch-domain proofs rather than
introducing another lifetime model. Unknown ordering remains a conflict. Placement
cannot remove conflicts or serialize execution to make a program fit.

A **region** represents one logical DFB. Its extent is the physical encoded page
size times pages per block times block count, rounded to target alignment.
Physical sizing includes packed-format metadata. A region's lifetime includes
outstanding asynchronous accesses, not just the lexical use of its value.

### Plan offsets statically; allocate one arena at launch

Sizes and interference are known during compilation. Computing offsets then
avoids a device-side allocator, runtime fragmentation, and allocation work inside
kernels. The complete plan is validated before declarations are rewritten, so a
budget failure cannot leave partially assigned storage.

For each call to a compiled `ttl.operation`, the runtime represents the arena as
a row-major tensor with shape `(coreCount, arenaWords)` and height-shards it with
one `(1, arenaWords)` row per participating worker core. Height sharding is the
TTNN allocation mechanism for obtaining one equal-sized, contiguous L1 shard on
each core; the arena has no tensor-height semantics. Kernels receive the
core-local arena base address and use constant offsets, so argument count does
not grow with region count. Metal still allocates the tensor, so its ownership
and reservation rules remain authoritative. Passing the arena as a `generic_op`
input retains it for execution while leaving the user output in the final tensor
position. A fresh arena separates synchronization state across cached launches.

The tradeoff is uniform allocation: every participating core reserves the largest
required arena, including sparsely active cores. Initialization also currently
clears the complete arena rather than only its control records.

### Reuse payloads; keep synchronization state distinct

Each logical DFB retains an 8-byte control record for the entire invocation.
Payload lifetime completion proves that data bytes may be overwritten. It does
not prove that another DFB can inherit the same counters and ring positions.
Keeping control records distinct avoids requiring a state-reset and ownership
handoff whenever payloads share an address.

Two 32-bit words are the smallest representation that keeps producer and
consumer updates independent. Packing both sequences into one word would require
an atomic read-modify-write operation.

This imposes a fixed `roundUp(8 * regionCount, alignment)` cost. For 96 one-page
BF16 regions, simultaneous lifetimes require 196608 payload bytes plus 768
control bytes. Sequential lifetimes require only 2048 payload bytes, but retain
all 768 control bytes: approximately 27% of that smaller arena is control state.
Payload optimization therefore has diminishing benefit for many small regions.

The [DFB protocol](DFBManagement.md#compiler-managed-storage-protocol) defines
counter ownership, completion, acquisition/release alternation, and wraparound.
Those semantics are separate from choosing payload offsets.

### Configure compute from operand metadata and addresses

Compute operands carry formats and arena-relative addresses instead of descriptor
indices. The [target adapter](../../include/ttlang/Target/TTKernel/LLKs/compiler_l1_compute_target.h)
invokes existing address-based LLK primitives;
architecture-specific signatures and address units stay inside that adapter.
The allocator and Python DSL require no architecture-specific behavior.

A processor-local context initializes compute hardware once per invocation and
tracks subsequent format changes. Reinitializing DST synchronization for every
logical DFB would disrupt in-flight compute. Format specialization is separate
from storage identity, and shared instruction-emission helpers remain out of line
to bound kernel code size as the number of logical DFBs grows.

UNPACK owns input consumption; PACK owns output publication. MATH uses the existing
DST synchronization protocol and does not update DFB counters. The completion
rules are defined in [DFB Management](DFBManagement.md#compiler-managed-storage-protocol).
Packing retains explicit output tile indices. `pack_tile_block` has no output
tile-index operand, so replacing individual pack operations with it would
discard the compiler-assigned L1 offsets.

### Place large regions first and reuse compatible gaps

Payload sizes differ, so placement operates on byte intervals rather than
assigning equal-sized slots. Decreasing-size order places the hardest-to-fit
regions before smaller ones consume available gaps. First-fit chooses the lowest
aligned address that avoids already placed interfering regions. This reuses
holes as well as complete earlier allocations.

The algorithm is deterministic and has bounded compilation cost. It does not
prove optimality. Exact search could find a smaller arena, but would add search
cost without changing the required lifetime proofs. A failed greedy placement
therefore reports a placement failure, not that no feasible allocation exists.

## Placement Algorithm

Inputs are region extents, a symmetric interference graph, target alignment, and
a byte budget. Declaration order breaks equal-size ties. Control records occupy
a separate prefix. All intervals below are half-open byte ranges.

```text
allocate(regions, interference, alignment, budget, reuseEnabled):
    controlEnd = roundUp(8 * regionCount, alignment)
    placed = empty set

    for region in decreasing extent, with declaration-order ties:
        blockers = placed regions that interfere with region
        if reuseEnabled is false:
            blockers = all placed regions
        sort blockers by start address

        candidate = controlEnd
        for blocker in blockers:
            if candidate + region.extent <= blocker.start:
                break
            candidate = max(candidate, roundUp(blocker.end, alignment))

        reject if candidate + region.extent exceeds budget
        assign region.payload = [candidate, candidate + region.extent)
        assign region.control = [8 * region.ordinal, 8 * (region.ordinal + 1))
        add region to placed

    return maximum payload end, or zero when there are no regions
```

**Correctness.** Every payload starts after the control prefix, whose records are
mutually disjoint. For each placement, scanning blockers either finds a sufficient
gap or advances past their ends. Thus the new interval overlaps no previously
placed interfering region. Induction establishes disjointness for every conflict
edge; all other overlap is authorized by the lifetime analysis.

Placement takes `O(N^2 log N)` time after interference construction and `O(N)`
plan storage. Graph adjacency uses `O(N^2)` bits; lifetime analysis and diagnostic
evidence have separate costs.

## Contract and Limits

- Storage ownership is static. Declarations of one logical DFB must agree on
  type and capacity. Tensor backing, explicit allocation groups, resets, and
  reconfiguration are rejected before offset assignment.
- The backend accepts full-block acquisitions with positive capacity
  below `2^31` pages and the ownership/alternation contract in DFB Management.
  Operations without an address-based lowering are rejected, including opaque
  pre-lowered C++ effects; there is no Metal-descriptor fallback.
- The budget includes payloads, control records, and padding. It comes from the
  target/runtime usable-L1 contract or an override. The runtime must still fit the
  arena alongside live tensors; an override does not reserve additional memory.
- Launches use one device and a uniform arena. Device-domain placement and
  external runtime resources are rejected. With no regions, no arena is allocated.

Compute currently accepts 32x32 BF16/FP32 tiles. Device tests on Blackhole cover
FPU/SFPU addition and multiplication with DRAM/L1 tensors, matmul, row/column
sum and maximum, broadcast, transpose, exponential, reciprocal, reciprocal square
root, sigmoid, tanh, and compute-produced intermediate storage. Both DRAM and L1
tensors are covered. The existing [compute configuration contract](ComputeKernelConfiguration.md)
requires a compatible unpack mode for every use of a logical DFB. Attention
publishes distinct reduction and SFPU operands to satisfy that FP32 constraint.
Compiler alignment tests cover Wormhole and Blackhole. Device correctness
evidence currently covers Blackhole.

## Validation and Implementation References

Validation checks numerical results and allocation safety independently:
[transfer tests](../../test/python/test_compiler_l1.py) exercise 96 simultaneously
live DFBs, sequential reuse, mixed extents, multiple cores, repeated launches,
and protocol wraparound. [Compute tests](../../test/python/test_compiler_l1_compute.py)
compare both memory models and exercise arithmetic with 66 allocated DFBs,
retained residuals, a Kimi-derived SiTU MLP residual, dependent state updates,
attention, expert merge, and repeated invocations. [Compiler stress tests](../../test/ttlang/Dialect/TTL/Transforms/compiler_l1_stress.py)
derive conflicts and live-byte lower bounds from generated schedules, then check
placement, target alignment, determinism, and budget boundaries. These checks do
not assume the greedy result is optimal.

The [allocator](../../lib/Dialect/TTL/Transforms/CompilerL1Allocation.cpp) consumes
[logical identities](../../include/ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h)
and the [storage conflict model](../../lib/Dialect/TTL/Transforms/DFBPhysicalAllocationPlan.cpp).
[TargetInfo](../../include/ttlang/Target/TargetInfo.h) supplies alignment through a
common interface. [Kernel runner](../../python/ttl/kernel_runner.py) owns arena
binding. Shared terminology is in the
[specification glossary](../sphinx/specs/TTLangSpecification.md#appendix-a-glossary).

## Future Work

### Sub-tile Compute

The compute lowering currently accepts only 32x32 BF16 and FP32 tiles. Supporting
smaller tile geometries requires the compiler to carry the selected height,
width, byte stride, and face layout into the address-based operand type. Target
adapters must translate that metadata into architecture-specific unpack, math,
and pack configuration. Allocation itself already operates on byte sizes and does
not require a new placement algorithm.

Device correctness coverage must include every supported geometry and dtype for
copy, element-wise operations, matmul, reductions, broadcast, transpose, and
format reconfiguration. Tests must also cover mixed geometries when the underlying
LLK operation permits them and require a compiler diagnostic otherwise.

### Other Backend Integrations

Broader layer sizes, mixed formats, and performance require additional
qualification. Consumer-owned in-place replacement requires a PACK-to-UNPACK
completion handoff before the consumer releases storage; producer publication
alone does not provide that ordering. Partial-block transactions require a
protocol extension that preserves capacity accounting and contiguous access
across wraparound.

Explicit DFB resets require a synchronized transition that restores counters and
positions after outstanding accesses complete. Supporting that semantic operation
is distinct from reconfiguring Metal descriptors, which this allocator eliminates.

Tensor-backed DFBs, allocation groups, PipeNet transfers, and external kernels
need explicit address ownership and completion contracts before they can use the
arena. Row-major transfer, tilize/untilize, packed-format execution, Wormhole
compute, trace replay, and independent command queues require separate device
qualification.
