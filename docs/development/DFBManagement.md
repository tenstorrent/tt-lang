# Dataflow Buffer Management

This document describes how the tt-lang compiler manages dataflow buffers (DFBs) -- the L1-resident circular buffers that transfer data between compute and data movement threads on Tenstorrent hardware.

## Overview

DFBs originate from two sources. User-declared DFBs are created explicitly in the DSL via `make_dataflow_buffer_like` and correspond to the programmer's data movement plan. Compiler-allocated DFBs are inserted automatically when the compiler needs concrete storage for a tensor SSA value: a tensor-level operation requires a CB-attached operand, a fused expression must be preserved before a source DFB release, or a computed value is stored by operations in multiple MLIR basic blocks.

Blackhole supports 64 physical DFB indices per node (0--63). Wormhole B0 and
Quasar support 32 (0--31). Compilation without target metadata uses the
conservative 32-index capacity. User and compiler-allocated DFBs share this
index space. Passes operating on individual kernels assign compiler DFBs
kernel-local provisional indices. The module-level finalization pass assigns
module-wide physical indices after the last user-declared DFB and applies
lifetime-based index reuse.

`ttl.bind_cb` separates logical and physical identity. `dfb_id` identifies one
logical DFB across kernel functions, while `cb_index` names its assigned
hardware slot. Keeping both identities allows non-overlapping logical DFBs to
share one physical index without merging their producer/consumer protocols.
Every user declaration carries `dfb_id`. Compiler-created declarations may
omit it until module finalization assigns a unique identity.

User declarations may also carry `allocation_group`. Equal typed group
identities require those logical DFBs to use one physical index. The identity
does not merge logical protocols, add synchronization, or reset DFB state. The
allocator validates the group contract before contracting its members into one
allocation vertex.

## Tensor-backed storage

`ttl.make_tensor_backed_dfb` binds a DFB's complete capacity to a byte range
of an operation tensor's node-local L1 allocation:

```python
input_dfb = ttl.make_tensor_backed_dfb(
    input_tensor, shape=(1, tile_count), block_count=1
)

@ttl.datamovement()
def publish_input():
    input_dfb.publish()
```

The tensor must use TILE layout, height-sharded L1 storage, and BF16 or FP32.
The optional `byte_offset` must be page-aligned. The bound byte size is
`product(shape) * block_count * page_size`; allocation padding does not change
the DFB capacity. The current contract supports one device and one or more
launch nodes whose tensor shards use the same local DFB specification.

The host creates or receives the device tensor before launching the operation.
The tt-lang launcher binds the DFB descriptor to the current invocation's
tensor with `ttnn.cb_descriptor_from_sharded_tensor`. Before binding, it uses
`ttnn.get_optimal_worker_cores_for_sharded_tensor` to require shard data on
every selected launch node. It does not copy tensor bytes while constructing
the program. TTNN orders a host-originated upload before the operation
dispatch.

Descriptor binding does not make input pages readable. `publish()` emits one
reserve/push pair for the complete DFB capacity. These protocol operations
advance the DFB producer state without moving bytes. Compute then uses the
normal wait/pop protocol. Consumers acquire and pop every block in FIFO order;
the DFB can be reused after the complete published capacity has been popped. A
tensor-backed output uses reserve/store/push so the packer writes directly into
the output tensor allocation.

`ttl.bind_cb` records the optional `#ttl.tensor_backing` identity as an
operation tensor index, byte offset, and byte size. Finalization emits exact
launch-node storage segments in `ttl.dfb_allocations`. Tensor-backed segments
do not contribute static DFB L1 bytes because the tensor allocator already
accounts for their storage. Scratch and tensor-backed segments may use the
same physical DFB index only on disjoint launch nodes.

Allocation groups do not weaken tensor storage identity. Tensor-backed group
members must have the same DFB capacity descriptor, and their backing ranges
must satisfy the normal per-node storage compatibility rules. The scratch-only
capacity-envelope rule described below does not apply to tensor-backed members.

The simulator exposes the same constructor signature but rejects it because
simulated tensor-backed storage is not implemented.

## Pipeline

The DFB-related passes in `ttl-to-ttkernel-pipeline` execute in this order:

```
ttl-form-accumulation-scopes       (FuncOp) Form tensor accumulation scopes
ttl-lower-accumulation-scopes      (FuncOp) Select tensor accumulation storage
ttl-materialize-loop-state         (FuncOp) Remove ranked-tensor scf.for iter_args
ttl-insert-copy-wait               (FuncOp) Insert missing ttl.wait ops
ttl-auto-sync                      (FuncOp) Insert/coalesce DFB synchronization
ttl-insert-accumulation-scopes     (FuncOp) Form DFB accumulation scopes
ttl-lower-accumulation-scopes      (FuncOp) Lower DFB accumulation metadata
ttl-create-producer-compute        (FuncOp) Create producer ttl.compute ops
ttl-insert-intermediate-dfbs       (FuncOp) Materialize compiler-allocated DFBs
convert-ttl-to-compute             (FuncOp) Lower remaining tensor ops
ttl-insert-cb-sync                 (FuncOp) Insert remaining DFB synchronization
ttl-verify-pipenet-guards          (Module) Verify PipeNet launch-node domains
ttl-verify-pipenet-schedule        (Module) Verify PipeNet event ordering
ttl-form-pipe-transports           (Module) Form PipeNet transport DFBs
ttl-coalesce-dfb-acquires          (FuncOp) Coalesce adjacent DFB acquisitions
ttl-finalize-dfb-indices           (Module) Finalize identities and allocations
ttl-set-compute-kernel-config      (Module) Resolve per-kernel configuration
  ... DST assignment, loop lowering, scheduling ...
ttl-annotate-cb-associations       (FuncOp) Copy DFB indices to tile ops
ttl-verify-dfb-spsc                (Module) Verify producer/consumer uniqueness
ttl-erase-pipenet-scopes           (Module) Remove verified PipeNet markers
ttl-validate-cb-budget             (Module) Validate DFB/reset/reconfig L1 use
convert-ttl-to-ttkernel            (Module) Lower to TTKernel dialect
ttkernel-insert-inits              (Module) Insert hardware init calls
ttkernel-specialize-cores          (Module) Clone coordinate-dependent kernels
canonicalize, cse                  (Module) Remove untaken coordinate branches
ttkernel-annotate-dfb-use          (Module) Record surviving physical DFB uses
```

The final three entries run only when per-core specialization is enabled.
Annotation follows canonicalization so an eliminated branch cannot keep an
otherwise-unused DFB live on that clone's launch node.

`ttl-finalize-dfb-indices` must precede
`ttl-set-compute-kernel-config` and `ttl-annotate-cb-associations`.
Compute configuration copies selected indices into
`ttl.unpack_to_dest_fp32`; association annotation copies `cb_index` onto tile
operations (`bcast`, `reduce`, `transpose`). Running either pass first would
leave stale attributes after finalization rewrites the indices.

The PipeNet verifiers run before DFB acquire coalescing and physical index
finalization. Guard verification uses the read-only logical identity analysis,
which assigns temporary IDs to compiler-created DFBs without modifying IR.
Schedule verification uses exact transfer provenance and does not depend on DFB
allocation.
Transport formation records a conservative PipeNet L1 reservation before
physical index assignment. Finalization may use that reservation to select a
lower-byte valid coloring. The budget pass validates finalized DFB and
synchronized-reset and reconfiguration allocations. Conversion validates the
exact combined allocation after PipeNet resource planning.

### Per-core descriptor allocation

Physical allocation records the exact union of launch nodes that access each
physical DFB index. The runtime restricts descriptors to this domain without
requiring kernel specialization. An exact empty domain installs no descriptor;
an unknown domain retains conservative whole-grid allocation.

Per-core kernel specialization can remove different DFB operations on
different launch nodes. Each final TTKernel function records the physical
indices still referenced by its compile-time arguments in
`ttl.used_dfb_indices`. Direct helper calls contribute their transitive uses.
An unresolved call or missing annotation is conservative and keeps every DFB
available on the affected function's launch nodes.

The runtime unions these sets for every kernel dispatched to a logical core,
then intersects the result with the physical allocation domain and finalized
storage segments. Static storage, tensor-backed ranges, and PipeNet
computed-address backing therefore use only cores where a surviving kernel can
access that physical index. A computed-address DFB with an empty effective
domain retains one hidden backing shard because the PipeNet ABI still requires
a receiver address, but no launch core installs its descriptor.

Descriptor construction creates one descriptor for each physical DFB storage
source and restricts it to the exact launch nodes using that source. Sparse
domains allocate one tensor shard per selected core rather than the area of
their bounding rectangle.

TT-Metal allocates static descriptor storage in descriptor order. It maintains
one allocation frontier per core, and a descriptor shared by several cores
starts at the greatest frontier among those cores. The runtime simulates these
frontiers with TT-Metal's address alignment and the remaining L1 on each core.
It preserves the physical-index order when that order fits. Otherwise, it
evaluates deterministic node-count orders and improving pairwise exchanges. If
those orders do not fit, a bounded exact search prunes states whose per-core
frontiers are dominated or whose remaining minimum allocation exceeds L1. Only
an order whose simulated allocation fits every selected core is emitted. Search
exhaustion proves that no order fits; reaching the state limit reports a
conservative failure and the best candidate's overflow.

The usable interval for each core begins at the configured DFB allocator base
and ends at the lowest live L1 tensor page. Subtracting only allocated page
sizes would ignore allocator gaps and could overestimate the available range.
Tensor-backed and already allocated computed-address storage do not advance the
static frontiers. For a multi-device mesh, tensor and runtime-resource
allocations use common L1 addresses, while harvested worker mappings can
differ. The runtime therefore applies the reference allocator's global minimum
remaining interval to every logical core.
The correctness invariant is that every surviving DFB access has one compatible
descriptor on its launch core; conservative metadata preserves the
whole-program descriptor behavior when this cannot be proved.

`ttl-verify-dfb-spsc` must run after `ttl-finalize-dfb-indices` so every
`bind_cb` carries its final `cb_index` and module-wide logical `dfb_id`. The
pass requires the `ttl.dfb_allocations` module attribute emitted by successful
finalization, then verifies that every declaration and lifecycle operand has a
resolved logical ID.

## Synchronized reset epochs

Completing one DFB transaction leaves its hardware read and write pointers at
their advanced ring positions. A later logical lifecycle cannot assume that a
reused physical index starts at its descriptor base, even when the earlier
lifecycle has zero occupancy. This prevents the compiler from assigning the
same physical index to otherwise disjoint lifecycles that require canonical
interface state.

`DFBReset` identifies one worker-local synchronization boundary and its logical
kernel participants. The built-in operations select either explicit DFBs or
every DFB allocated by the program:

```python
def make_reset_operation():
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset_boundary = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def reset_operation(input_tensor):
        scratch = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reset_dfbs(reset_boundary, dfbs=[scratch])

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            ttl.reset_dfbs(reset_boundary, dfbs=[scratch])

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.reset_dfbs(reset_boundary, dfbs=[scratch])

    return reset_operation


reset_operation = make_reset_operation()
```

The same `DFBReset` value identifies the three occurrences as one dynamic
boundary. `ttl.reset_all_dfbs(reset_boundary)` provides the same boundary for
every allocated physical DFB index. A declaration contains exactly one compute
kernel and two data movement kernels. It executes once per dispatch and launch
node, or once per iteration of the same immutable sequential loop nest in all
participants. Conditional occurrences must use equivalent structured conditions
on all participants and cannot form a repeated reset run.

The compiler treats the interval before the first reset, each interval between
resets, and the interval after the last reset as separate allocation epochs.
An epoch is an analysis interval; the compiler does not emit an epoch object.
The liveness analysis proves that every selected DFB has either a balanced
protocol lifecycle or bounded producer-only occupancy before the boundary,
that no pre-reset payload is consumed afterward, and that every participant
selects the same DFB set. The reset discards producer-only occupancy and
terminates the old lifecycle at canonical empty state. A later lifecycle can
then reuse the physical index when its launch-node domain, storage, element
type, and other allocation constraints are compatible. Missing participants,
nonuniform or mismatched repeated sequences, mismatched conditions or target
sets, incomplete transactions, and unordered boundaries are compilation errors.

On Blackhole, `convert-ttl-to-ttkernel` lowers each occurrence to
`experimental::reset_dfb_interfaces(state_address, low_mask, high_mask)` from
[`experimental_dfb_reset.h`](../../include/ttlang/Target/TTKernel/LLKs/experimental_dfb_reset.h).
The compiler reserves one 16-byte synchronization record per declaration after
PipeNet scratch storage. The combined allocation is rounded to the runtime L1
allocation quantum, included in transport selection and DFB budget validation,
and initialized to zero by the host. DM1 coordinates DM0, UNPACK, and PACK
through distinct L1 state words. Each participating data movement RISC drains
its own outstanding NoC commands before publishing arrival. UNPACK and PACK
wait for their previously issued interface commands to retire. After entry
synchronization, the selected interface owners reset their read pointer, write
pointer, packer write-tile pointer, initialization state, and stream occupancy
counters. An exit synchronization completes before any owner returns. MATH
executes a no-op because it does not own DFB interface state.

The operation does not clear payload bytes, change descriptor configuration,
or complete NoC commands issued by another core or a non-participating RISC.
Every producer must issue its required transfers before its local boundary
occurrence; the participating data movement RISC then completes its own
outstanding commands. Runtime lowering is currently restricted to Blackhole.

## Synchronized reconfiguration epochs

A `DFBReconfiguration` declares one compute kernel and two data-movement
kernels that execute one worker-local configuration boundary. Every boundary
site executes zero or one dynamic instance per dispatch and launch node. All
sites in one module declare the same participant set, and every active
participant executes the same boundary instances in the same dynamic order.
Structured conditional execution is supported only when the participant
conditions are equivalent. Runtime execution is restricted to Blackhole.

The declaration captures the participating logical kernels. Each participant
calls `ttl.reconfigure_dfbs` at the corresponding point between two DFB
lifecycles:

```python
reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
boundary = ttl.DFBReconfiguration(
    participants=(ttl.KernelKind.COMPUTE, reader_kernel, writer_kernel)
)


@ttl.operation(grid=(1, 1))
def two_stage_copy(first_input, first_output, second_input, second_output):
    first_source = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
    first_result = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
    second_source = ttl.make_dfb("bf16", shape=(1, 2), block_count=2)
    second_result = ttl.make_dfb("bf16", shape=(1, 2), block_count=2)

    @ttl.compute()
    def compute():
        with first_source.wait() as source:
            with first_result.reserve() as result:
                result.store(source)
        ttl.reconfigure_dfbs(boundary)
        with second_source.wait() as source:
            with second_result.reserve() as result:
                result.store(source)

    @ttl.datamovement(kernel=reader_kernel)
    def read():
        with first_source.reserve() as destination:
            ttl.copy(first_input[0, 0], destination).wait()
        ttl.reconfigure_dfbs(boundary)
        with second_source.reserve() as destination:
            ttl.copy(second_input[0:1, 0:2], destination).wait()

    @ttl.datamovement(kernel=writer_kernel)
    def write():
        with first_result.wait() as source:
            ttl.copy(source, first_output[0, 0]).wait()
        ttl.reconfigure_dfbs(boundary)
        with second_result.wait() as source:
            ttl.copy(source, second_output[0:1, 0:2]).wait()
```

The first and second DFBs may receive the same physical indices because their
lifecycles are ordered by the boundary. The compiler derives and installs the
second descriptors; the declaration does not supply descriptor values.

Concurrent-kernel liveness builds a happens-before graph for each launch node
and orders reconfiguration boundaries independently of their numeric ordinals.
An ordinal identifies a boundary; it does not define execution order. Every
boundary pair that co-occurs on a launch node must have a strict local order,
and the union of those local orders must be acyclic. Disjoint boundary domains
need not contain the same boundary sequence. Unknown execution or ordering
remains conservative.

A complete reserve/push/wait/pop lifecycle can end before a boundary and a new
lifecycle can begin afterward. An incomplete transaction, unread payload, or
other live protocol state may cross only when the logical DFB retains the same
physical index, storage, and interface configuration. Such a lifecycle remains
active in every crossed allocation epoch. A lifecycle beginning under a
conditional boundary must use the same condition so an inactive boundary
cannot leave a stale descriptor for unconditional following work.

The allocation conflict graph permits two lifecycle epochs to share a physical
index only when their per-node active epochs are disjoint and their static
element type and tile descriptor are compatible. Reconfiguration can change
outer DFB geometry, block count, and storage. Tensor-backed ranges are checked
against the complete set of descriptors installed initially and after each
boundary, in proven execution order.

Finalization emits the initial descriptor for every physical index and one
entry configuration for every lifecycle that begins at a boundary. Live
continuations have no entry update, so their FIFO pointers, occupancy, and
payload are preserved. The runtime plan records boundary order explicitly and
does not infer it by sorting ordinals.

Each boundary owns a per-core configuration tensor containing 64 four-word DFB
interface records, two update masks, synchronization state, and padding. The
runtime supplies its address to every participating kernel. DM1 coordinates
DM0, UNPACK, and PACK through separate shared-L1 state words. UNPACK and PACK
publish entry only after a hardware completion marker proves their prior engine
work retired. MATH does not access DFB interfaces and does not wait in shared
L1; normal compute dependencies order it against UNPACK and PACK. The exit
handshake prevents any interface owner from beginning following DFB work until
all masked updates complete. Independent math and SFPU work may overlap the
boundary.

Caller-defined per-core runtime arguments precede the compiler-owned
configuration addresses. The compiler reserves one compile-time argument for
the caller argument count, so generated kernels locate the configuration
addresses without changing caller-visible indices. When caller argument counts
differ by core, the runtime emits descriptors for disjoint core sets with one
count per descriptor.

Configuration tensors, PipeNet scratch, computed-address backing, and
GlobalSemaphore objects remain owned by the operation's serialized
runtime-resource cache. Compatible calls reuse one generation. Incompatible
replacement and owner destruction synchronize the device before releasing it;
failed synchronization retains ownership.

Per-core L1 accounting uses target allocation quanta rather than logical byte
counts. It includes one aligned maximum allocation per non-tensor-backed
physical DFB index, one aligned configuration tensor per boundary, aligned
PipeNet scratch, and one allocation quantum per GlobalSemaphore. Transport
formation uses a conservative upper bound that includes scalar, grouped,
residual, and record-selected callback resources. Finalization minimizes
weighted physical DFB allocation when authoritative capacity requires it and
may also do so for the conservative PipeNet reservation. The budget pass checks
finalized DFB plus configuration state; conversion performs the authoritative
combined check from the exact PipeNet plan. Every pass uses the same resolved
target budget or `l1-budget-override`.

## DFB Lifecycle

A DFB has two lifecycle halves: the producer (write) side driven by
`cb_reserve`/`cb_push`, and the consumer (read) side driven by
`cb_wait`/`cb_pop`. For user-declared DFBs these halves span different
kernels: a data movement kernel writes to the DFB, a compute kernel reads from
it, and both kernels reference the same DFB index. For compiler-allocated
intermediate DFBs, both halves are in the same compute kernel.

```
|
v time
          Producer (write)          Consumer (read)
          ----------------          ---------------
bind_cb   cb_reserve                cb_wait              L1 buffer held
          store                     attach_cb              |
          cb_push ------ slot ----> ... consumer ops       |
          (slot returned) <-------- cb_pop               L1 buffer free
```

`cb_reserve` claims a buffer slot for the packer; `cb_push` releases that slot
to the unpacker. `cb_wait` blocks until the slot is available; `cb_pop`
releases it back to the packer. `bind_cb` identifies the hardware binding
shared by both sides. The launcher provisions either static scratch storage or
the finalized tensor-backed storage for that binding.

The hardware-visible occupancy of one DFB is the difference between published
producer slots and released consumer slots. `cb_push` increases that occupancy
by publishing a filled slot, and `cb_pop` decreases it by acknowledging that
the consumer is finished with the slot. `cb_reserve` and `cb_wait` are blocking
guards over that state: reserve requires an unoccupied slot, while wait
requires an occupied slot. `attach_cb` has no protocol effect; it only turns
the waited tensor into an SSA value that carries the DFB association for later
tile-level lowering.

For a compiler-created intermediate, the compiler must construct one balanced
logical lifecycle for each materialized SSA value:

```
bind_cb {ttl.compiler_allocated}     // storage, no occupancy change
cb_reserve                           // producer may claim a free slot
ttl.compute {
  tile_store ..., %reserved_slot      // pack writes the tile
}
cb_push                              // occupancy: 0 -> 1
cb_wait                              // consumer may read the occupied slot
attach_cb                            // tensor SSA view of the waited slot
... consumers of attached tensor ...
cb_pop                               // occupancy: 1 -> 0
```

The producer side is ordered so the slot is reserved before the compute that
writes it and published only after the compute has packed the materialized
tile. The consumer side is ordered so every rewritten operand is dominated by
the attached tensor value, and the pop is placed after the final use of that
attached value. With `blockCount=1`, this is also the complete capacity proof:
there is never a second compiler-created push while the previous pushed slot is
still outstanding.

Every control-flow trace that executes a compiler-created `cb_push` must also
execute the matching `cb_pop`, and no trace may execute the pop without the
push. An unconditional push followed by a conditional wait/pop is therefore
invalid: on traces that skip the condition, the DFB remains occupied and a
later reserve can block. Until an explicit DFB occupancy dataflow analysis can
prove more general structured-control-flow placements, compiler-created
compute-result waits and attaches are emitted in the same straight-line
sequence immediately after the producer push. This construction gives all
traces the same occupancy transition and makes the attach dominate all original
users dominated by the producer result.

Producer `ComputeOp` creation replaces a tensor-level producer and its output
stores with one `ttl.compute`. It preserves the same write-before-push order for
user DFBs.
When `ttl-create-producer-compute` or `convert-ttl-to-compute` absorbs a
`ttl.store` into a `ttl.compute`, any `ttl.cb_push` that would otherwise precede
the new compute is relocated after the compute. This keeps
the generated DFB lifecycle in write-then-publish order: `cb_reserve`,
`ttl.compute` with `tile_store`, then `cb_push`. Both passes use the shared,
read-only compute-op-creation analysis described below before modifying IR.

A wait-backed `ttl.store` represents replacement of consumer-owned pages. The
initial contract accepts one complete block from a one-block DFB, with the wait
as the compute kernel's first access to that DFB. Analysis requires the
replacement computation to read the original generation, requires values
derived from the original contents to remain within that computation, excludes
overlapping DFB access, and requires every replacement-generation read to
precede the matching pop.
Lowering emits `ttkernel.pack_waited_tile`, then converts it to the same
`pack_tile` runtime call used for producer stores. It emits no reserve or push;
the operation changes neither occupancy nor either DFB pointer. Full-ring
acquisition and the absence of earlier producer-pointer access prove that the
consumer read window and pack destination identify the same pages.

After `cb_pop`, the producer may overwrite the released slot because its prior
contents are no longer live. The DFB's backing storage remains statically
allocated. Index reuse uses this release to prove that non-overlapping logical
DFBs can use the same storage. Two logical DFBs may share a physical index only
when their read- and write-pointer effects execute on the same hardware
processors on every shared launched node. A happens-before relation proves zero
occupancy but does not transfer ring-pointer state between processors.

## Single-producer Single-consumer Semantics

### Contract

Each DFB has at most one producer thread and at most one consumer thread on
each launched node. A *thread* here is a `func.func` carrying the
`ttl.kernel_thread` attribute (compute, noc, ethernet); ops in untagged
functions are outside the contract.

Multiple producer or consumer threads may reference the same DFB index when the
compiler can prove that their launch-node domains are disjoint. For example, a
DFB may be consumed by a compute thread on PipeNet destination nodes and by a
data-movement thread on PipeNet source nodes, provided no launched node belongs
to both consumer domains.

The rule is inherited from tt-metal: its CB protocol is not multi-writer safe on either side. Each CB has two shared counters in `dataflow_api.h`:

- `pages_received`, incremented by `cb_push_back` (producer side),
- `pages_acked`, incremented by `cb_pop_front` (consumer side).

`cb_reserve_back` blocks until `pages_received - pages_acked < block_count`.
`cb_wait_front` blocks until `pages_received > pages_acked`. The protocol is
correct only when exactly one thread on a physical node writes each counter;
the counters are not atomic with respect to multiple writers and carry no
per-thread identity.

### Violation

A two-consumer DFB inside a stripe loop:

```python
buf = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

@ttl.compute()
def compute():
    for _ in range(num_stripes):
        with buf.reserve() as b:
            b.store(...)
        with buf.wait() as b:       # consumer A: compute
            ...

@ttl.datamovement()
def dm_read():
    for _ in range(num_stripes):
        with buf.wait() as b:       # consumer B: dm_read
            ...
```

Per iteration, the producer pushes once (`pages_received += 1`) and each consumer pops once (`pages_acked += 2`). After iteration 0, the producer's `cb_reserve_back` on iteration 1 sees two free slots when only one has actually been consumed; it writes slot 0 while the late consumer is still reading slot 0's old data. The symmetric failure occurs with two producers: each `cb_push_back` advances the shared write pointer, and a consumer reads a partially-written slot.

A single-iteration test masks this — exactly one push and two over-pops do not corrupt data when the producer never refills — so the rule must be enforced statically rather than left to test coverage.

### Correct form

When two consumers or producers can execute on the same launched node, allocate
one DFB per consumer thread (and symmetrically per producer thread). The
producer writes the value into each DFB; each consumer reads its own. The
sketch below is illustrative (no `@ttl.operation` wrapper, no tensor shape);
for a runnable example see
`test/python/test_store_patterns.py::store_then_forward_kernel`:

```python
buf_for_compute = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
buf_for_dm     = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

@ttl.compute()
def compute():
    for _ in range(num_stripes):
        val = ...
        with buf_for_compute.reserve() as b: b.store(val)
        with buf_for_dm.reserve()     as b: b.store(val)
        with buf_for_compute.wait()   as b: ...

@ttl.datamovement()
def dm_read():
    for _ in range(num_stripes):
        with buf_for_dm.wait() as b: ...
```

Each `pages_received`/`pages_acked` pair is now driven by a single thread on
each launched node.

When the participating threads have disjoint launch-node domains, the same DFB
index can be shared without duplicating storage. The verifier accepts this form
because every physical node still observes a single producer and a single
consumer for that DFB.

### Verification

The `ttl-verify-dfb-spsc` module-level pass runs after
`ttl-annotate-cb-associations`. It walks producer and consumer actions exposed
through `DFBAccessOpInterface`, groups them by logical `dfb_id` and enclosing
`ttl.kernel_thread`-tagged `func.func`, and tracks the launch-node domain for
each participant. Concrete reserve, push, wait, and pop operations and external
protocol summaries therefore use the same verification. Distinct logical DFBs
remain separate after physical allocation assigns them the same `cb_index`.

The pass rejects a DFB when two producer domains overlap or when two consumer
domains overlap. If multiple threads participate and a coordinate-dependent
predicate cannot be analyzed statically, the pass rejects the DFB rather than
assuming disjointness. The diagnostic identifies the logical `dfb_id`, the
role (producer or consumer), an overlapping launched node when available, the
participating operation sites, and the originating `ttl.bind_cb`.

Setting `TTL_RELAX_DFB_SPSC` skips only launch-node-domain proofs that require
the program to provide synchronization absent from IR. It skips overlapping
producer/consumer domain checks here and producer correspondence for DFB waits
in `ttl-verify-pipenet-guards`. Finalized DFB identity, physical-index, and
launch-grid preconditions remain mandatory. PipeNet endpoint guards, transfer
correspondence, and synchronization schedules also remain mandatory. Strict
verification is the default.

See `test/ttlang/Dialect/TTL/Transforms/verify_dfb_spsc_invalid.mlir` for the rejected patterns and `verify_dfb_spsc.mlir` for the accepted ones.

The compiler does not currently auto-split overlapping multi-consumer DFBs;
users must duplicate explicitly via `make_dataflow_buffer_like`. Tracked in
[tenstorrent/tt-lang#581](https://github.com/tenstorrent/tt-lang/issues/581).

## Compiler-Created Intermediate DFB Insertion

`TTLInsertIntermediateDFBs` walks all operations implementing
`DFBInputOpInterface`, including reduce, block broadcast, matmul, transpose,
and selected elementwise forms that require DFB-attached operands. For each
operand that the interface marks as requiring a DFB-attached value, the pass
checks whether the operand traces to an existing DFB via `getAttachedCB` and
whether that storage remains available before the consumer. An unattached or
possibly released operand is materialized through a fresh compiler-allocated
DFB marked with `ttl.compiler_allocated`. This pass creates DFBs for
intermediate tensor SSA values. It does not replace existing user DFB
declarations; the lifetime analyses described below apply to values backed by
both user and compiler-created DFBs.

The standard pipeline runs this pass after `ttl-create-producer-compute`.
Values produced by `ttl.compute` are materialized by the compiler-created
intermediate lifecycle described above: the compute gains extra DFB outputs,
and consumers receive attached tensor values instead of the original
non-attached compute results. The final `convert-ttl-to-compute` pass lowers
consumers that now receive DFB-attached operands. The following
`ttl-insert-cb-sync` pass inserts the consumer `cb_pop`.

### DFB Lifetime and `ComputeOp` Creation Planning

This section defines the DFB ownership, availability, and materialization facts
used by `ComputeOp` creation. [ComputeOpCreation.md](ComputeOpCreation.md) is the
authoritative design for candidate planning, fusion, output-store placement,
kernel-wide selection, and mechanical application.

`ttl-create-producer-compute`, `ttl-insert-intermediate-dfbs`, and
`convert-ttl-to-compute` build complete read-only plans before modifying a
kernel. A plan records input identities, iteration semantics, output
transactions, application order, and any required intermediate DFBs. The
producer and final conversion passes recompute the plan around intermediate
DFB insertion; no analysis result is reused after its kernel changes.

#### Acquire and Release Ownership

The lifecycle index records every producer acquisition (`cb_reserve`),
consumer acquisition (`cb_wait`), producer release (`cb_push`), and consumer
release (`cb_pop`). Producer and consumer pointers are independent and are
analyzed separately.

Straight-line transactions in the kernel entry block use the DFB FIFO protocol
and static `num_tiles` values to match releases to one or more acquisitions.
Other blocks may receive an outstanding transaction, so their releases remain
unresolved. Control flow that prevents an exact entry-block FIFO match also
retains conservative ownership. An entry-block release that exceeds all
preceding acquisitions, or a release with no same-kind acquisition anywhere in
the kernel, is malformed IR and is diagnosed before any rewrite.

Block order is causal: an acquisition nested after an entry-block release
cannot supply tiles to the earlier release, even when the nested region later
executes. Conversely, a release after nested control flow is unresolved when
the analysis cannot determine which dynamic acquisitions reach it. Unresolved
ownership is a conservative set of possible owners, not proof that the
transaction counts are balanced.

```
indexLifecycles(kernel):
  record every acquisition and release in kernel walk order

  for each DFB and producer-or-consumer kind in the kernel entry block:
    outstanding = FIFO queue of (acquisition, remainingTiles)
    for operation in block order:
      acquisition -> append its tile count
      release     -> consume its tile count from the queue
      nested lifecycle operation -> mark later releases unresolved

  diagnose an entry-block release that underflows its queue

  exact one owner       -> Exact
  exact several owners  -> Multiple
  release outside the proven entry-block sequence -> Unresolved with every
                                                       same-kind acquisition
                                                       on the DFB as a candidate
```

A DFB-backed tensor has an exact identity when it derives from one acquisition
through conversion casts, `ttl.attach_cb`, `tensor.extract_slice`, or
`tensor.extract`. These operations preserve the acquired storage identity.
An association without a local acquisition has only its DFB identity. It
represents storage present at kernel entry because `attach_cb` has no protocol
effect. Any release on that DFB may invalidate it, and another association does
not reacquire it.

The availability analysis is an MLIR dense forward dataflow analysis. It
tracks every static acquisition and association at each program point and
uses MLIR's CFG and region control-flow propagation.

```
entry state:
  exact acquisition identities are unavailable
  standalone association identities are available

transfer(acquisition): mark its exact identity available
transfer(association): no state change
transfer(release):
  if FIFO ownership is exact or spans several acquisitions:
    mark every recorded owner unavailable
  otherwise:
    mark every possible owner may be unavailable
  mark standalone associations on the released DFB may be unavailable

transfer(DFBAccessOpInterface operation):
  for each summarized push or pop:
    mark every identity on the effect's DFB may be unavailable
  if unknown DFB access:
    mark every user-managed DFB identity may be unavailable

join(predecessors):
  available only if every reachable predecessor is available

query(non-executable program point):
  available, because no runtime read occurs
```

Partial releases invalidate the complete tensor because the lattice does not
track tile ranges. Unresolved ownership also invalidates every same-kind
acquisition on the DFB. An external push or pop does not identify a concrete
FIFO owner, so it may invalidate every identity on its DFB. Unknown access has
the same effect on every user-managed DFB identity. These rules may require an
additional intermediate DFB, but they cannot classify released storage as
available. Dead code analysis excludes statically non-executable blocks from
this conservative fallback; dense analysis creates no lattice there, and
availability holds vacuously because the consumer cannot execute.

#### `ComputeOp` Creation

The creation planner consumes the availability result at the program location
where it proposes to create the `ComputeOp`. A direct or fused candidate is
legal only when every lifetime root is definitely available there. Planned
materializations remain compute roots but are excluded from lifetime roots
because their replacement DFB supplies new storage. If another unmaterialized
occurrence reads the same SSA value, it remains a lifetime root.

Output-store planning groups stores by their `cb_reserve` or `cb_wait`
acquisitions and prevents one compute from combining several transactions of
the same DFB. Reserve-backed transactions preserve producer-pointer order when
a push moves after the created `ttl.compute`. Wait-backed transactions require
the complete consumer-owned replacement proof described above. Kernel-wide
selection and application ordering are described in `ComputeOpCreation.md`.

The analysis is kernel-local because creation moves operations only within
one kernel. Producer and consumer pointer states are separate, and the
module-level `ttl-verify-dfb-spsc` pass verifies cross-kernel producer and
consumer domains. The implementation supports any number of kernels; the
current two data movement kernels and one compute kernel are not hard-coded.

The correctness argument relies on these pipeline assumptions:

- DFB operations pass their op verifiers, including static tile counts and
  result types consistent with each acquisition.
- Reserve/push and wait/pop follow the DFB FIFO producer and consumer
  protocols. A tensor derived through a recognized view operation continues
  to name its acquisition until the corresponding release.
- Each selected tile recipe defines the tensor operation's tile
  semantics. Fusion relocates signposts and tile-observing debug prints with
  recorded placement. If that relocation would cross a non-reorderable
  operation, materialization splits the tensor SSA frontier first.
- A plan is applied only to the unchanged kernel from which it was built.
  Application verifies recorded operands and uses before rewriting them.
- `ttl-insert-cb-sync` runs after final creation and inserts absent pushes and pops
  after the resulting final uses.

#### Compiler-Created Intermediate DFB Analysis and Materialization

Intermediate materialization follows the One-Shot Bufferize analysis model. A
whole-kernel analysis state records each required `OpOperand` and the evidence
for that decision. The requirement set reaches a fixed point before the pass
builds or applies a materialization plan. Operand identities remain valid while
the kernel is unchanged and are checked again before application.

```
requirements = DFBInputOpInterface operands that are unattached or may be
               released before their consumer

repeat until requirements does not grow:
  for each ttl.compute result named by an existing requirement:
    require every other surviving use of that result

  for each fusable expression operand:
    roots = trace inputs, stopping at existing requirements
    if any root may be released before the consumer:
      require that operand

  for each supported `ComputeOp` source operation:
    outputs = plan output transactions
    inputs = collect current-storage inputs, stopping at existing requirements
    if tracing stops at an operand produced by ttl.compute or by an operation
       with a standalone compute recipe:
      require that exact operand
      continue
    if creation would reorder instrumentation with another operation:
      require each tensor SSA consumer operand crossing that boundary
      continue
    if creation would not dominate a surviving result use:
      require every result use
    if one output DFB has several reserve transactions:
      require every result use
    else for each output store:
      if any input may be released before that store:
        require the stored-result operand

group requirements by source value
build every materialization record
verify the complete plan
apply standalone materializations
topologically order compute rebuilds by SSA dominance
apply compute rebuilds in that order
```

Each requirement selects one consumer operand for DFB materialization. Input
tracing treats that operand as a future DFB-backed value, so a later creation
does not read the original expression's inputs. The requirement set grows
monotonically over the kernel's finite operand set, which proves termination
and removes kernel walk order from the result.

When fusion reaches a producer with a complete standalone compute recipe, the
failed trace reports the exact consumer operand. Materializing that operand
creates an independent producer `ttl.compute` and makes its result a DFB input
to the consumer. This is a producer/consumer boundary rule; it does not
enumerate operation pairs.

The instrumentation-order query uses MLIR's `isPure` contract rather than a TTL
operation list. When movable instrumentation precedes an operation that MLIR
cannot reorder but the created `ttl.compute` would follow it, the query records
tensor SSA uses from producers before that operation to consumers after it.
Materialization replaces those uses, and the next fixed-point iteration creates
an independent `ttl.compute` on each side. Uninstrumented pure tensor recomputation remains
legal. Output acquisitions are part of the recorded output transaction and are
not ordering boundaries; the created `ttl.compute` necessarily executes after
them.

`ttl.compute` results preserve SSA dependencies, but the compute body publishes
data only through `ttl.tile_store`; `ttl.yield` does not carry tile values. If
one surviving use requires a compiler DFB, every surviving use of that result
must read the same pushed and waited materialization. The original user-DFB
store remains a `ttl.tile_store` in the compute body and is not a
surviving result use. This rule prevents a later creation from treating the
compute result as readable storage without a DFB acquisition.

Materializing a published result adds another formal output to the existing
compute rather than creating another producer. This is also valid for an
accumulating compute. Tile operations execute inside the reduction loops, and
all stores execute afterward while the accumulated DST values remain
available. Each store uses the indexing map associated with its formal output,
so one accumulated value can publish both to its original user DFB and to the
compiler DFB read by another consumer.

The current `ttl.tile_store` syntax does not contain its formal-output index.
Verification and lowering therefore identify the output by the DFB attached to
the store view and require formal outputs to use different DFBs. This is an IR
representation limitation, not a hardware restriction. Issue
[#797](https://github.com/tenstorrent/tt-lang/issues/797) tracks encoding the
association explicitly and removing DFB-based output discovery.

An existing `ttl.compute` is rebuilt once with all required additional DFB
outputs. Other tensor definitions are each materialized once. All consumers of
one source receive the same waited and attached value. When one result has
several reserve transactions on the same DFB, the result is materialized once
and each original store becomes a passthrough compute in its original
transaction.

Standalone materializations run before compute rebuilds because they preserve
their source and consumer operations. MLIR's region-aware topological sort
orders compute rebuilds by SSA dominance rather than region or block list
order. Valid SSA requires a producer to dominate each consumer, so the
producer rewrites a recorded consumer operand before that consumer is rebuilt;
no plan retains an operation after an earlier application erases it.

Requirements may rewrite some original output stores before final creation.
The standalone plan therefore selects the last output store that will remain,
not the last store in the analyzed IR. That store must dominate every rewritten
consumer, preserve every unrewritten use, and precede each recorded
instrumentation-order boundary. If no output store satisfies those conditions,
definition-site materialization is legal only when every original use is
rewritten. These checks ensure the inserted compiler DFB store remains before
its wait and before the operation whose order must be preserved.

The materialization plan is sufficient by induction over its fixed point. A
new requirement identifies a consumer operand that application replaces with
storage owned by a new DFB transaction. The next iteration analyzes creation
inputs while treating that operand as the future DFB-backed value. At
termination, every creation candidate either has available inputs and valid
output ordering or records the condition that prevents its application. Before
mutation, final conversion verifies that every `ttl.store` will either be
absorbed into a planned `ttl.compute` or converted to a DFB-to-DFB passthrough,
so an unsupported or unsafe source cannot survive as partially converted IR.

#### Relationship to Upstream MLIR

The implementation reuses upstream compiler infrastructure for general
program analysis and structured operation semantics:

- MLIR dense forward dataflow and dead-code analysis propagate availability
  only through executable CFG and `RegionBranchOpInterface` edges.
- `DominanceInfo` proves SSA availability, and MLIR's region-aware topological
  sort orders producer rebuilds before their consumers.
- `TilingInterface`, `DestinationStyleOpInterface`, and
  `IndexingMapOpInterface` define compute iteration and output mapping.
- memory-effect and speculation interfaces state the precondition for
  recomputing a producer across a region boundary.

The upstream One-Shot Bufferize implementation supplies the analysis-first
methodology but not the required DFB semantics. Bufferization reasons about
tensor and memory aliases; it does not model DFB FIFO acquisitions,
reserve/push and wait/pop ownership, DFB output transactions, or TTL tile
recipes. The TTL-specific implementation therefore adds:

- exact acquisition identities and conservative release-owner sets;
- static FIFO tile-count matching for entry-block transactions;
- three-state planner results that distinguish a valid plan, a legal candidate
  that must remain unchanged, and malformed IR;
- complete `ComputeOp` creation and output-store records;
- a monotone set of required consumer operands and one grouped
  materialization plan.

This division keeps control-flow, dominance, operation-effect, and scheduling
mechanisms in upstream infrastructure while retaining DFB protocol and TTL
lowering semantics in the dialect.

#### Disabled Mode and Resource Accounting

The default pipeline enables compiler-created DFB materialization. All normal
tests therefore exercise planning with materialization available. With
`--no-ttl-compiler-dfbs`, analysis still reaches the same fixed point, but the
pass diagnoses every required materialization instead of modifying IR. This
includes consumers reached after a DFB release and values published through
several reserve transactions. The option does not restore the earlier,
mutation-dependent creation behavior.

Each materialized result adds one block-count-one DFB and its L1 allocation.
Physical index reuse may later assign the same index to non-overlapping
compiler DFB lifetimes of identical type. `ttl-validate-cb-budget` runs after
index finalization and verifies the resulting static DFB storage against the
device-specific remaining L1 budget. Materialization that is semantically
required but exceeds either the selected target's DFB-index capacity or the L1
budget is rejected with a resource diagnostic.

`test/python/test_recurrence_multi_output_dfb.py` exercises this behavior on
hardware with several results materialized from the same compute, nested and
sibling elementwise consumers of one producer, an unstored reduction consumed
by an elementwise operation, and one result published through several reserve
transactions. `compute_op_creation_materialization.mlir` checks the corresponding
creation order and exact materialization decisions before mutation.

Materialization does not infer a store from a Python assignment. The original
`ttl.compute` already contains a `ttl.tile_store` for each explicit block
store. Rebuilding the compute preserves that store and replicates its tile into
the compiler DFB needed by a downstream DFB-only consumer.

For each producer `ttl.compute`, the pass rebuilds the compute exactly once
using this sequence:

1. Preserve all original results in their existing result order.
2. Append one compiler-allocated DFB output for each source result that needs
   materialization, ordered by source result number.
3. Clone the original compute body.
4. For each cloned `ttl.tile_store` that writes the original source DFB,
   replicate the tile into the appended compiler DFB output.
5. Replace uses of the original compute results with the corresponding results
   of the replacement compute.
6. Emit `cb_push`, `cb_wait`, and `attach_cb` for each appended compiler DFB
   after the replacement compute.
7. Rewrite all planned consumer operands for a source result to the same
   attached value.

This producer-centric plan makes materialization independent of consumer order
and of other results from the same producer. For example:

```python
a, b = ttl.compute(...)
ttl.compute(a)
ttl.compute(b)
ttl.compute(a)
```

The source results are `a = (producer, 0)` and `b = (producer, 1)`. The pass
creates two compiler DFB outputs, not three, and both consumers of `a` use the
same attached materialization. The producer compute is rebuilt once, so the
plan is not affected by transient SSA values created while rewriting another
result of the same producer.

`TTLMaterializeLoopState` uses the same compiler-DFB materialization helper
(`include/ttlang/Dialect/TTL/Transforms/DFBMaterialization.h`) to remove
ranked-tensor `scf.for` iter_args before compute lowering.
The helper chooses a provisional index by scanning only the enclosing kernel,
so passes operating on individual kernels do not inspect sibling kernels while
MLIR executes them concurrently.

Non-compute producers use standalone tensor materialization. The plan records
the MLIR operation after which the helper creates the reserve/store and
wait/attach; `ttl-insert-cb-sync` inserts the missing releases:

```
bind_cb {ttl.compiler_allocated}
cb_reserve
store
cb_push
cb_wait
attach_cb
... consumer ...
cb_pop
```

When the source already has a valid output-store plan, the compute is created
immediately before its final output store if that store properly dominates
every rewritten consumer. Final `ComputeOp` creation may relocate source
evaluation to that program location; placing the compiler reserve/store/wait
afterward therefore keeps evaluation before the wait. A source without an
applicable output-store plan remains at its definition. If a valid output store
does not dominate every
consumer, the fixed point must require every source use. Rewriting all uses
removes the original output stores, so the compiler DFB store inserted after
the definition becomes the source's only output store and therefore its compute
creation location. The planner verifies these conditions before modifying IR.

When multiple `DFBInputOpInterface` operations consume the same non-compute
value, they share one materialization. The operation after which the compiler
creates the materialization properly dominates every rewritten consumer, so
the attached replacement is valid for consumers in nested or incomparable
regions.

When one computed value has direct `ttl.store` users in multiple MLIR basic
blocks, one `ttl.compute` cannot absorb all of them. A compute operation is
located in one MLIR basic block, and all output stores in its body execute
whenever that operation executes. Moving stores from different MLIR basic
blocks into one compute could therefore change their conditional execution or
violate SSA dominance.
`ttl-insert-intermediate-dfbs` therefore materializes the value once and
rewrites each original store to read the attached compiler DFB value while
remaining in its original MLIR basic block. Final compute creation sees one
store for the producer result: the compiler DFB store inserted at the
materialization point.

For `ttl.compute` results, the attached value is created immediately after the
producer push, so consumers originally dominated by the compute result remain
dominated by the materialized value. This includes branch-local consumers when
the producer is outside the branch. General occupancy-balance proofs for
placing compiler-created waits and pops inside arbitrary structured control
flow remain tracked by
[#724](https://github.com/tenstorrent/tt-lang/issues/724).
`insideMutuallyExclusiveRegions` proves branch-exclusive store fanout, but does
not prove that DFB lifecycle operations balance within each branch.
`ExecutionCountAnalysis`, which PipeNet schedule verification uses for
structured protocol occurrences, is applicable to the remaining occupancy
proof.

Compiler-allocated intermediate DFBs are created with `blockCount=1`. A
compute kernel's Unpack, Math, and Pack stages are separate RISC-V cores that
run the same kernel program, compiled once per core and executed concurrently.
They synchronize through the circular buffer: Pack executes
`cb_reserve_back`/`cb_push_back`, and Unpack executes
`cb_wait_front`/`cb_pop_front` ([METALIUM_GUIDE, three-core compilation],
[annotated compute kernel]).

That handshake is correct at any depth >= 1. Pack's `cb_reserve_back` blocks
until a slot is free, and Unpack's `cb_wait_front` blocks until a tile is
available, so one slot never lets Pack overwrite an unread tile or Unpack read
an unwritten one. A second slot only lets Pack run ahead of Unpack to overlap
data movement with compute, which is a throughput optimization with
diminishing returns and an L1 cost ([METALIUM_GUIDE, buffer depth]).

`blockCount=1` is always correct and minimizes L1, so it is the current default
for compiler intermediates. Whether a deeper buffer would improve throughput
for a given intermediate is workload-dependent: it depends on how much
Pack/Unpack overlap the surrounding computation admits and on
`dst_full_sync_en`, since single-buffer DST has no overlap to exploit. Deeper
compiler-created DFBs should be selected per intermediate by benchmarking and
a cost model rather than a fixed constant ([#727]). User-declared DFBs, which
transfer between the reader/compute/writer kernels, keep their double
buffering.

[METALIUM_GUIDE, three-core compilation]: https://github.com/tenstorrent/tt-metal/blob/c04ae2758fc87f3a49ca19a7d339464db90e995d/METALIUM_GUIDE.md#L132
[annotated compute kernel]: https://github.com/tenstorrent/tt-metal/blob/c04ae2758fc87f3a49ca19a7d339464db90e995d/METALIUM_GUIDE.md#L154-L170
[METALIUM_GUIDE, buffer depth]: https://github.com/tenstorrent/tt-metal/blob/c04ae2758fc87f3a49ca19a7d339464db90e995d/METALIUM_GUIDE.md#L333
[#727]: https://github.com/tenstorrent/tt-lang/issues/727

## DFB Sync Insertion

`TTLInsertCBSync` inserts missing releases for DFB acquire operations. A
`cb_reserve` acquire requires a later `cb_push`; a `cb_wait` acquire requires a
later `cb_pop`. The pass is also responsible for hoisting releases that were
emitted inside structured regions to the acquire's block when that is the
correct DFB interval boundary.

Here a release closes pointer-side access to acquired slots. A push publishes
producer-written tiles; a pop returns consumer-read capacity to the producer.
Neither operation deallocates the DFB.

The pass treats every acquire as opening a DFB live interval. The interval
starts at `cb_reserve` or `cb_wait` and ends after the last operation that can
use the acquired slot.

DFB acquire kinds separate the producer side from the consumer side:
`cb_reserve`/`cb_push` form producer intervals, and `cb_wait`/`cb_pop` form
consumer intervals. Producer acquires bound other producer intervals; consumer
acquires bound other consumer intervals.

Uses inside descendant regions are projected to their ancestor operation in the
acquire's block. This conservatively places the release after the enclosing
structured op when the exact use is nested in an `scf.for` or `scf.if` body.

Conditionally yielded acquires have one additional rule. `scf.yield`
propagates the acquired tensor value, but it is not a storage access. When all
owned uses remain inside the acquiring then-region, the inserted release also
remains in that region after the last local use. When the yielded tensor is
used after the acquiring `scf.if`, each use must execute under the same
condition as the acquire, and the inserted release is emitted under that
condition after the last escaped use.

### Ownership

A use `U` is *owned by* `acquire` if `U` accesses the slot `acquire` acquired.
Two disjoint criteria establish ownership:

- **Tile-SSA ownership** -- `U` is reachable from `acquire`'s result through
  the def-use chain over `attach_cb`, `tensor.extract`,
  `tensor.extract_slice`, compute ops, and `ttl.store`. Per-tile SSA values
  uniquely identify their source acquire, so this criterion has no positional
  bound: a use of `cb_wait t1`'s tile is owned by `t1` regardless of where it
  appears, even past later acquires on the same DFB.

- **Direct-DFB ownership** -- `U` references the DFB directly and may access
  its physical storage. A `ttl.copy` is owned only on the operand side matching
  the acquire kind. A `DFBAccessOpInterface` operation matches the
  producer or consumer class when it exposes an effect on that side. A
  dependency occurrence without effects remains a possible read and write, and
  unknown access matches every user-managed DFB. Identity-only `ttl.attach_cb`
  and `ttl.get_dfb_id` operations do not consume an acquired slot. With no SSA
  tile handle, ownership is positional: `U` belongs to the latest acquire on
  `(dfb, acquire kind)` that precedes it in operation order. Equivalently, `U`
  is bounded between `acquire` and the next acquire of the same kind
  (`interval.kindBoundary` in the analysis).

The criteria are disjoint. DM-thread `ttl.copy` does not flow through
`attach_cb` (it takes the DFB directly). Compute-kernel uses always go through
`attach_cb` and never reference the DFB as a direct operand of a tile op.

#### Why two criteria

Compute threads work through SSA tile handles
(`cb_wait` result -> `attach_cb` -> `ttl.store` / compute ops), so tile-SSA
ownership applies and the next-acquire boundary is irrelevant -- SSA already
distinguishes which slot the use refers to. Data-movement kernels and external
calls can use direct DFB references where no tile handle exists, so direct-DFB
ownership uses the operation interval and its boundary to disambiguate
consecutive direct uses on the same DFB. Unifying would require changing every
direct storage-accessing operation to take the attached tensor instead of the
DFB.

### Invariants on the inserted release

For each acquire `A`, the inserted release `R_A` must satisfy:

1. **Causal dominance** -- every owned use of `A` precedes `R_A` in op order
   (after projecting nested uses to `A`'s block). The pass enforces this
   directly: the release is positioned after the last owned use returned by
   `findLastDFBAcquireOwnedUse`.

2. **FIFO monotonicity** -- for `A_0 < A_1 < ...` on the same `(dfb, acquire
   kind)`, the inserted releases satisfy `R_0 < R_1 < ...` in op order. The DFB
   front or back pointer advances monotonically; out-of-order pops would advance
   it past slots whose data is still needed.

(1) is enforced explicitly by the pass. (2) is enforced *implicitly* when
tile-SSA consumers appear in declaration order (`use(t1); use(t2); use(t3)`).
Reordered consumes (`use(t2); use(t1)`) would violate FIFO monotonicity on
their own, but in the current pipeline `TTLCoalesceDFBAcquires` runs
immediately after `TTLInsertCBSync` and rewrites N consecutive same-DFB
acquires into one multi-tile acquire plus per-block `tensor.extract_slice`
views and a single coalesced release with `num_tiles = N*k`. Per-tile
`src_idx` values fall out of `extract_slice` offsets, so consume order is
decoupled from release order and (2) is preserved by construction.

### Idempotency

When the pass runs twice on the same IR, the second run must observe the
releases inserted by the first as already-present and skip re-injection.
Because tile-SSA ownership can place a release past the next-acquire boundary
(when a tile is consumed later than the next acquire on the same DFB),
`findOwnedDFBReleases` extends its release-search upper bound to the acquire's
last owned use. Without this extension, the second run sees the inserted
release as past the boundary and treats the acquire as needing another
release.

### Slot State Model

The pass models producer and consumer acquires as separate slot lifetimes:

```
Producer side:

  free slot
      |
      | cb_reserve
      v
  reserved slot
      |
      | reserve-side writes
      v
  written slot
      |
      | cb_push
      v
  visible to consumer

Consumer side:

  visible to consumer
      |
      | cb_wait
      v
  acquired slot
      |
      | wait-side reads
      v
  consumed slot
      |
      | cb_pop
      v
  free slot
```

Each acquire owns exactly one interval. The release inserted for that interval
must follow the last owned use. For direct-DFB ownership, the release must also
precede the next acquire of the same kind because direct DFB uses are
position-based:

```
cb_wait A  ->  owned reads  ->  cb_pop A  ->  cb_wait B
                                  ^
                                  inserted release
```

Direct-DFB ownership is positional: a release after the next acquire in the
same acquire kind is owned by that next acquire, not the earlier one. Tile-SSA
ownership is unbounded: a release placed after a tile's last use can sit past
the next acquire and still belong to the earlier interval. The pass
distinguishes these two cases by use criterion, not by a single bound.

### Algorithm

```
planAndInsertMissingReleases(func):
  reserves = all cb_reserve ops in func
  waits = all cb_wait ops in func
  producerReleases = all operations with push effects
  consumerReleases = all operations with pop effects

  producerPlan = planReleases(reserves, producerReleases, cb_push)
  consumerPlan = planReleases(waits, consumerReleases, cb_pop)
  reject before mutation if either plan is invalid
  apply producerPlan, then consumerPlan

planReleases(acquires, releases, releaseOp):
  for acquire in acquires:
    dfb = acquire.cb
    boundary = next acquire of the same kind on the DFB, projected to acquire.block

    liveEnd = latest owned use:
      direct-DFB uses are bounded by boundary
      tensor-SSA uses ignore boundary

    matching = same-block operation with the required release effect
    nested = nested operations with the required release effect
    reject if matching precedes an owned use
    if matching:
      continue

    if acquire is conditionally yielded:
      reject if an owned use is not under the acquire condition
      if all owned uses are local to the acquiring then-region:
        keep an existing local release, or insert releaseOp after the local use
      else:
        reject if a local release is an external effect summary
        plan erasure of local concrete releases
        plan insertion of scf.if acquire.condition { releaseOp(dfb) }
        after liveEnd
      continue

    reject if a nested release is an external effect summary
    plan erasure of nested concrete releases
    plan insertion of releaseOp(dfb) after liveEnd
```

The same-block release check makes the pass idempotent. A summarized external
push or pop can satisfy this check, but it cannot move out of a nested region;
the pass rejects that case because only concrete release operations can be
recreated at the interval boundary. Producer and consumer plans are both
validated before mutation. For direct-DFB ownership, a release after the next
acquire of the same kind on the DFB belongs to that later interval and does not
satisfy the earlier acquire. For tile-SSA ownership, an existing release past
the boundary still satisfies the earlier acquire when it follows that
acquire's last owned tensor use. Transaction tile-count validation occurs in
the concurrent-liveness analysis.

## DFB Acquire Coalescing

`TTLCoalesceDFBAcquires` runs immediately after `TTLInsertCBSync` and
rewrites a maximal run of consecutive same-DFB acquires (and their matched
releases) into a single multi-tile acquire plus per-block
`tensor.extract_slice` views, with the matched releases collapsed into one
release carrying `num_tiles = N*k`.

```
%t1 = ttl.cb_wait %cb            %g  = ttl.cb_wait %cb {num_tiles=N*k}
%t2 = ttl.cb_wait %cb            %t1 = extract_slice %g [0, 0]   [1,k]
...                              %t2 = extract_slice %g [0, k]   [1,k]
ttl.cb_pop %cb                   ...
ttl.cb_pop %cb                   ttl.cb_pop %cb {num_tiles=N*k}
```

This matches the canonical tt-metal "cumulative wait + indexed reads +
coalesced pop" pattern (eltwise_binary.cpp, bcast_h.cpp, the matmul
kernels). Without coalescing each acquire lowers to its own
non-cumulative `cb_wait_front(k)` / `cb_pop_front(k)`, which races
whenever consumes are deferred: the first pop advances the front before
the producer has pushed enough tiles to satisfy the next read.

`addSliceOffset` (`include/ttlang/Dialect/Utils/ConversionUtils.h`) folds
each `extract_slice` offset into the per-tile `src_idx` / `dst_idx` at
lowering, so no lowering changes are required. The producer side
(`cb_reserve` / `cb_push`) uses the same templated helpers — per-block
`extract_slice`s become the views of downstream `ttl.tile_store` /
`ttl.store` ops, and `addSliceOffset` handles store-side dst indices the
same way.

### Correctness criterion

For a candidate group of acquires `G = {a_1, ..., a_N}` on DFB `c`, the
rewrite is correct iff every op `O` between consecutive group members
preserves the synchronization invariant of `c` under the coalesced
schedule. The coalesced acquire performs one `N*k`-slot acquire before
anything between original `a_i` and `a_{i+1}` runs: `cb_wait` requires
`N*k` tiles to be present, while `cb_reserve` requires `N*k` free slots.
The coalesced release runs only after the last group member's last use.

This holds iff no op between members causes a release on `c` (directly or
transitively): the original IR may have advanced the matching DFB pointer
between `a_i` and `a_{i+1}`, and the coalesced version delays all releases
until the end. Forbidding inter-member releases is therefore necessary for
correctness at low `block_count`, and sufficient when paired with the
coalesced release placement.

A locally-checkable (sound, conservative) version of that criterion: an
op `O` between members is safe to skip past iff none of:

1. `O` operates on `c` directly (`c` appears as an operand). Covers
   `cb_pop` / `cb_push` on `c` and any other op that reads or writes `c`.
2. `O` consumes the SSA result of any current group member. A consume can
   flow into a release on `c` somewhere downstream, and we don't perform
   transitive analysis.
3. `O` carries a region. Region bodies might contain a release on `c`;
   conservative cutoff.

Anything else — an acquire or release on a different DFB, `arith.constant`,
pure compute on other DFBs — cannot affect `c` and is safe. `ttl.attach_cb`
is explicitly excluded from rules (1)–(2): it is an SSA-only identity op
(the metal lowering erases it) that always references the group's results
and `cb` as operands, so the generic check would otherwise wrongly break
the group at every `attach_cb`.

#### Why this is sufficient

Suppose `O` between `a_i` and `a_{i+1}` satisfies all three negations
above. Then:

- `O` does not directly call any release on `c` (rule 1).
- `O`'s outputs do not consume any tile from `G` (rule 2 on operands; the
  outputs cannot make further data depend on `G`'s tiles).
- `O` has no inner region that could hide an indirect release on `c`
  (rule 3).

So the only way a release on `c` could appear before the coalesced
release is via a transitive use of some non-`G` value. Because rule 2
forbids `G`'s outputs from being inputs to `O`, no fresh dataflow path is
created from `G` into a `c` release. Any release on `c` reachable from
some unrelated value would have run in the original IR too, at exactly
the same op-order position, so the coalesced version is no worse.

#### Why this is necessary

If `O` is itself a release on `c` (e.g., a user-written `cb_pop` for consumer
acquires or `cb_push` for producer acquires), the original IR advances one
slot at `O`, but the coalesced acquire holds all `N*k` slots from the start.
With `block_count` only slightly larger than the working set, the matching
kernel cannot make progress until all members are released. Same argument for
transitive releases via group results.

### Detection algorithm

Per block, pre-collect all acquires of the kind under consideration
(`cb_wait` for the consumer pass; `cb_reserve` for the producer pass).
For each candidate leader (in op order):

```
if leader is already coalesced (num_tiles set) or already erased:
  continue

group = [leader]
for op = leader.nextOp; op != nullptr; op = op.nextOp:
  if op is a same-kind same-cb acquire with no num_tiles:
    group.push_back(op); continue
  if op is a same-kind acquire on a different DFB:
    continue  # benign: cannot touch our DFB or our group's results
  if mayReleaseDFB(op, cb=leader.cb, group):
    break
  # else: tolerate (different-DFB op, attach_cb, arith, ...)

if group.size() < 2: continue
match N releases on cb after the last group member, in op order
apply rewrite, mark group members as erased
```

Because the candidate set is fixed before any rewrite, acquires on a
different DFB that the inner loop skips past (e.g., the matmul-style
`a1, b1, a2, b2` interleave) still get a chance to lead their own group
on a later iteration of the outer loop.

### Idempotency

The coalesced acquire and release carry a `num_tiles` attribute, and
`detectGroup` skips acquires that already have one. A second run of the
pass therefore finds no candidate groups and is a no-op. The doubled-pass
lit invocation
(`--pass-pipeline='builtin.module(func.func(ttl-coalesce-dfb-acquires,
ttl-coalesce-dfb-acquires))'`) verifies this.

### Limitations

- Non-rank-2 acquire result types are not coalesced. The existing
  `num_tiles` convention (matching `TTLSubblockComputeForDST`) produces
  `tensor<1, num_tiles, elem>`; the pass conservatively bails on other
  ranks rather than picking an axis to scale.
- Acquires already carrying `num_tiles` (set by
  `TTLSubblockComputeForDST`) are not extended.
- Region-bearing ops between members terminate the group, so coalescing
  does not span control flow within an `scf.if` or `scf.for` (loop-body
  coalescing still works because the body is its own block).

`ExecutionCountAnalysis` supplies structured execution counts for PipeNet
schedules, including reducible block-CFG loops. Counts alone do not justify
acquire coalescing, which also requires proof that every grouped acquire
executes in one contiguous DFB interval.

## Index Reuse

`TTLFinalizeDFBIndices` reduces the physical DFB count by assigning the same
index to logical DFBs whose lifetimes cannot overlap. The default analysis
considers all kernel functions concurrently, including user-declared DFBs
shared across data-movement and compute kernels.

Ordinary reuse requires identical `CircularBufferType` values (shape, element
type, and block count). An explicit allocation group may combine scratch DFBs
with different block shapes or block counts when their element types and page
formats are identical. The physical descriptor then uses the largest total
capacity. Both mechanisms require each lifecycle to complete. Ordinary reuse
also requires matching write- and read-pointer runs unless a synchronized reset
establishes canonical state. The matched sequences must remain boundary-safe
when repeated from their terminal offsets. Allocation groups instead advance
independent write and read cursors through each ordered member. A terminal
synchronized reset establishes equal canonical offsets before the next
handoff; otherwise the offsets must already be equal. Any pointer movement that
crosses the shared physical envelope is rejected. These conditions retain one
page format, sufficient storage, and legal ring-pointer progression.
`CircularBufferType` is an MLIR-uniqued type, so exact ordinary compatibility
is a pointer comparison.

### Repeated synchronized reset intervals

A synchronized reset declaration in an immutable sequential `scf.for` or
`affine.for` loop denotes one collective reset instance per iteration. Every
participant must execute once in each iteration on the same launch node and
must have the same nested trip-count sequence. Unknown counts, conditional
iterations, non-sequential loops, participant-count differences, and target-set
differences are rejected because they cannot establish corresponding collective
instances.

The liveness analysis represents the first and last reset instances without
expanding the happens-before graph by the trip count. A DFB receives a repeated
interval lifetime only when every active access executes once per iteration in
the same local loop nest and completes before that iteration's reset. Protocol
validation then normalizes each selected access to one representative interval.
The reset supplies the terminal canonical pointer and occupancy state, and the
lifecycle epoch records the number of represented intervals.

This representative interval does not create a dispatch-wide allocation
boundary between unrelated DFB declarations. Accesses outside the matching loop
domain, consumer-only intervals that may block before the reset, and incomplete
or conditionally mismatched protocols remain conservative.

### Logical identity

The frontend assigns each user-declared DFB a module-wide logical `dfb_id` and
copies that ID to the declaration in every participating kernel.
Compiler-created declarations receive distinct logical identities after the
largest explicit ID. A user declaration without `dfb_id` is rejected before
physical allocation.

Compiler-created DFBs are currently kernel-local. A module transformation
that distributes one compiler-created DFB across multiple kernels
must assign the same explicit `dfb_id` to every declaration.
The allocation planner rejects any used compiler-created DFB without a
complete reserve, push, wait, and pop lifecycle. It does not infer a relation
between declarations when the IR provides no shared identity.

The finalizer records every resolved logical identity on `ttl.bind_cb` before
rewriting `cb_index`. Repeated finalization therefore cannot merge logical
DFBs that already share a physical slot. Allocation visits DFBs in immutable
declaration order. Changing logical ID values therefore does not affect
acceptance.

The frontend creates allocation groups with
`ttl.make_dfb_allocation_group()`. Composition, operation splitting, and
logical-kernel replication preserve the declaration identity as a module-local
`#ttl.dfb_allocation_group<ordinal>` attribute. The ordinal has no runtime
meaning. Every declaration of one logical DFB must carry the same group or no
group; partial propagation is rejected.

### Concurrent-kernel lifetime analysis

`DFBConcurrentKernelLivenessAnalysis` models the concurrently executing kernel
functions separately on each launched physical node. It uses
`LaunchNodeDomainAnalysis` to associate every DFB access with the nodes where
that access may execute. DFBs with disjoint known launch-node domains require
no global lifetime order. An unknown domain remains conservative.

Each top-level operation in a single-block kernel function receives an entry
event and a completion event:

```text
op.entry -> op.completion -> next.entry
```

Program-order edges connect consecutive operations within each kernel. When a
logical DFB has the same number of statically enumerated push and wait
occurrences, the blocking protocol pairs them by occurrence order. Each pair
with equal tile counts adds this cross-kernel edge:

```text
producer kernel:  ... -> cb_push.completion -----------------+
                                                             |
consumer kernel:  cb_wait.entry -> (blocked) -> cb_wait.completion -> ...
```

The edge targets wait completion, not wait entry. A consumer may enter
`cb_wait` before the producer publishes data, but it cannot complete that wait
before the matching push completes. Treating the push as preceding wait entry
would permit a later logical DFB to reuse the physical slot while its consumer
is already waiting and could consume the earlier DFB's data.

After adding all sound program-order and protocol edges, the analysis computes
transitive reachability. Cyclic events are not considered ordered:
`strictlyPrecedes(A, B)` requires reachability from A to B and no reachability
from B to A.

This construction follows Lamport's
[happened-before relation](https://lamport.azurewebsites.net/pubs/time-clocks.pdf):
per-process order and communication order generate a partial order over events.
The [LLVM concurrent memory model](https://llvm.org/docs/LangRef.html#memory-model-for-concurrent-operations)
uses the analogous construction from per-sequence program order and
`synchronizes-with` edges. DFB push-to-wait completion is the protocol-specific
communication edge in this analysis.

Every kernel function forms a separate event sequence. A logical `dfb_id` is
only an equivalence key attached to `ttl.bind_cb` declarations; it does not
encode hardware state ownership. The analysis associates each lifecycle
operation with the ID of the declaration reached from its DFB operand.
Protocol edges from all logical DFBs share one module graph, so transitive
order can pass through any number of intermediate kernels.

#### Lifetimes with statically known transactions

For one-to-one transaction matching, a logical DFB is bounded only when all of
these conditions hold:

- a positive, equal number of reserve, push, wait, and pop executions reference
  it;
- static execution analysis proves exact counts and iteration domains for each
  repeated run;
- the normalized executions pair by order into reserve/push/wait/pop
  transactions;
- within every transaction, reserve precedes push and wait precedes pop;
- adjacent transaction runs are ordered within one exact static iteration
  domain, or every execution of the earlier run precedes every execution of the
  later run;
- each concrete push follows all uses owned by its reserve, and each concrete
  pop follows all uses owned by its wait;
- all four actions in a transaction transfer the same tile count (`num_tiles`),
  and that count divides the physical DFB capacity;
- all reserve and push occurrences have one known write-pointer owner, and all
  wait and pop occurrences have one known read-pointer owner;
- the final pop follows every active access occurrence on the DFB.

Exact cumulative external effect sequences use the separate queue-state proof
described under [External calls](#external-calls).

Lifecycle operations inside a statically selected `scf.if`, `affine.if`,
`ttl.if_src`, or `ttl.if_dst` region, or inside an exact static loop, may satisfy
these conditions. An at-most-once region inside a static loop also qualifies
when exact counts prove that the region executes in every loop iteration.
Dynamic trip counts and runtime-selected repeated regions remain unproven.

An access with an unknown launch-node domain is refined to an exact domain when
execution-count analysis proves a count on every base launch node. Nodes with
positive counts belong to the domain; nodes with zero counts do not. The proof
applies per access before their domains are combined. One unresolved count
preserves the unknown result. A logical DFB whose access-domain union is empty
has no runtime storage access and may share a type-compatible scratch descriptor
without a lifetime-order proof. Tensor-backed empty domains retain the issue
#813 allocation diagnostic.

The pass runs after `ttl-insert-copy-wait`. A transfer into or out of a DFB
completes at its `ttl.wait`, whose transfer-handle operand does not identify the
DFB. Inserting that wait before the corresponding push or pop ensures the
lifecycle release follows transfer completion.

The acquire/release ownership analysis described in
[DFB Sync Insertion](#dfb-sync-insertion) supplies the owned-use checks.
Failure to prove any condition leaves the DFB unbounded.

#### External calls

Every DFB accessed by a custom function or transitive helper must appear in the
value sequence returned by `DFBAccessOpInterface::getDFBDependencyOperands()`.
For `ttl.opaque_call`, the sequence contains DFB function arguments, descriptor
template arguments, then dependency-only operands. An effect's dependency index
identifies one element of this sequence; it does not describe execution order.
When a `ttl.get_dfb_id` result reaches external C++, finalization verifies that
the same logical DFB is also a dependency and rejects the call before mutation
otherwise. The compiler cannot inspect custom C++ for hidden constants or
global state, so validity of the declared access contract remains an
external-code requirement.

Each effect identifies one dependency occurrence, one reserve, push, wait, or
pop action, and a positive static tile count no greater than that DFB's physical
capacity. The effect list is a single call-wide execution sequence, including
actions on different DFBs. The event graph preserves these cross-DFB relations
and the order of statically expanded transactions. Effects are synchronous
facts about actions completed inside the external call; they do not emit
lifecycle operations.

An occurrence with neither a protocol effect nor a non-transactional access
remains a possible read or write beginning at call entry. Its access contract
is incomplete unless a matching synchronized reset proves lifecycle completion,
so allocation cannot prove bounded reuse with another logical DFB on a shared
launch node before that boundary. Exact disjoint launch-node domains may still
share because they never use the physical allocation on the same node. If
operand adaptation aliases several occurrences to one DFB, every occurrence
requires an explicit contract or the same reset-completion proof. A partial
summary supplies its listed events but cannot establish the complete
reserve/push/wait/pop lifecycle for that DFB. A bounded external lifecycle
requires balanced, ordered transactions with equal tile counts, known pointer
owners, supported execution counts, and no access after the terminal pop.

Native `ttl.copy` is not an external access. Its surrounding acquire and release
operations define slot ownership, so the ordinary lifecycle proof determines
whether reuse is valid.

`dfb_accesses` describes typed synchronous accesses without queue transactions.
`ttl.DFBAccess.inspect(dfb)` states that the callee may read the selected DFB's
descriptor or contents but does not publish, consume, or leave that DFB changed.
The call remains a storage access, so reuse still requires strict lifetime
order; the summary establishes an identity queue-state transition. One
dependency occurrence cannot declare both a protocol effect and a
non-transactional access.

`unknown_dfb_access` represents access to user-managed DFBs absent from the
declared dependencies. For allocation, liveness analysis conservatively adds
the call as an opaque occurrence on every user-managed logical DFB, including
listed DFBs, over the call's launch-node domain. Listed effects remain available
to other verification. Unknown access applies only to user-managed DFBs;
compiler-created DFB accesses require listed operands.

Every declared effect action must complete before the callee returns. Associated
interface work may remain active while the declared protocol retains ownership;
it must complete before the terminal consumer release or a synchronized reset.
For a named dependency with no effect summary, a synchronized reset ordered
after the call may terminate the opaque access and canonicalize protocol state.
The reset must complete earlier interface work before publishing arrival. This
does not validate the callee's internal protocol. Unlisted access declared by
`unknown_dfb_access` remains unbounded. The frontend and IR representation are
described in
[External Function Interop Lowering](ExternalFuncInteropLowering.md).

An exact static `dfb_effects` sequence may describe cumulative queue state
rather than one-to-one transactions. Reserve and wait effects establish
readiness thresholds relative to the current write or read cursor. Push and pop
effects advance those cursors. One readiness check may therefore authorize
several smaller pointer movements, and one publication may satisfy several
smaller waits. Repeated readiness checks retain the greatest still-valid
threshold.

The cumulative proof requires every protocol effect to execute exactly once in
one ordered external-call sequence. Producer and consumer cursor movement must
have equal positive totals, each side must have one constant pointer owner, and
every non-protocol DFB access must remain inside an acquire/release interval.
The analysis relates each wait completion to the first push that publishes its
required cumulative position. When a reservation exceeds currently available
capacity, it also relates that reserve completion to the first pop that returns
the required credit. Repeated effects within one opaque call are not matched as
independent operation-completion transactions. When one producer call and one
consumer call contain the complete ordered protocol, the analysis simulates
both effect sequences against the physical DFB capacity. A feasible cycle that
exists only within the candidate edge batch remains unproved at operation
granularity rather than becoming contradictory evidence. A protocol that
cannot progress, or a relation that contradicts the existing happens-before
graph, remains a hard contradiction. Unknown order, dynamic counts, mixed
native and external cumulative sequences, condition mismatch, and incomplete
terminal consumption remain conservative.

Reports preserve the normalized transaction boundaries and the raw
`write_cursor_runs` and `read_cursor_runs`. Allocation compatibility uses the
raw cursor movements; the normalized sequence is diagnostic evidence and does
not erase different pointer advancement.

`num_tiles` counts tiles of the DFB's `TileType`. TT-Lang configures each
tiled CB page from the byte size of that tile. Two 16x32 bf16 tiles therefore
consume the same bytes as one 32x32 bf16 tile. Tile dimensions remain part of
the `CircularBufferType`, so DFBs with different tile dimensions cannot share
a physical index.

TT-Metal advances each ring pointer by
[`num_pages * fifo_page_size`](https://github.com/tenstorrent/tt-metal/blob/e908c31332b60860ed0d4186452dc880cdd5a81d/tt_metal/hw/inc/api/dataflow/dataflow_api.h#L208-L214).
The pointer wraps only when it reaches the end of the physical DFB. Logical
DFBs sharing one physical index therefore preserve their exact write and read
cursor runs. Each run must remain within the physical capacity from its current
offset; a movement that crosses the allocation end is rejected. Producer and
consumer cursors must reach the same offset at a lifecycle handoff.

For a bounded DFB, every storage-accessing operation with a direct DFB operand
is projected to a top-level function operation. `attach_cb` and `get_dfb_id`
carry DFB identity without accessing physical storage and are excluded. The
owned-use check still rejects an attachment whose tensor use extends beyond
release. Unrelated operations are contracted from each kernel sequence because
this preserves reachability among all events queried by the lifetime proof. The
analysis records:

- `earliestEvents`: the minimal use-entry events under happens-before;
- `terminalEvents`: the `cb_pop` completion event.

`earliestEvents` can contain operations from several kernels. It is an
antichain: no recorded event strictly precedes another. Requiring the terminal
event of DFB A to precede every earliest event of DFB B proves that A is dead
before any kernel can begin using B.

```text
isOrderedBefore(A, B):
  return A is bounded
     and B is bounded
     and every A.terminalEvent strictly precedes
         every B.earliestEvent
```

For example, a second data movement function can relay completion from the
compute function back to the first data movement function:

```text
DM0 producer:  push A --------------------------- wait ack2 -> reserve B
                  |                                   ^
Compute:       wait A -> pop A -> push ack1           |        wait B
                                      |               |
DM1 relay:                       wait ack1 -> push ack2
```

Program order and the two acknowledgment DFBs establish:

```text
A.pop[Compute]
  -> ack1.push[Compute]
  -> ack1.wait.completion[DM1]
  -> ack2.push[DM1]
  -> ack2.wait.completion[DM0]
  -> B.reserve.entry[DM0]
```

Compute program order separately establishes
`A.pop[Compute] -> B.wait.entry[Compute]`. Therefore A's terminal pop precedes
both producer-side and consumer-side events in B's earliest-event frontier. An
unrelated third kernel adds no cross-kernel edge. A `B.wait` entered before
`A.pop` also remains unordered and prevents reuse.

The write- and read-pointer owners are also part of the allocation state. The
owner is the launched node plus NOC0, NOC1, Pack, or Unpack and the pointer
direction. Distinct kernel function symbols may share when they execute on the
same hardware pointer processors. A completed lifecycle does not transfer
pointer state between different processors.

#### Relation to prior allocation models

The concurrent lifetime model is closest to Suhendra, Roychoudhury, and
Mitra's
[scratchpad allocation for concurrent embedded software](https://www.comp.nus.edu.sg/~abhik/pdf/codes08-spm.pdf).
Their message-sequence model forms a partial order from process-local order and
inter-process communication, then overlays scratchpad allocations at proven
boundaries. DFB reuse requires additional proofs: FIFO occupancy must be zero,
all protocol accesses must be complete, and the physical index must retain
compatible kernel-local pointer and counter state.

Bhattacharyya and Lee's
[dataflow memory-management model](https://ptolemy.berkeley.edu/publications/papers/94/buffering/)
and Murthy and Bhattacharyya's
[shared-memory implementation model](https://citeseerx.ist.psu.edu/document?doi=6d061c87abf14b211d1f7c64c3527d14bc6b984b&repid=rep1&type=pdf)
combine static schedules, buffer lifetimes, storage sharing, and external-memory
tradeoffs for synchronous dataflow. These models apply to matched DFB protocol
occurrences and future DRAM spilling. Their static-rate assumptions do not
cover tt-lang structured control flow, external effects, or node-dependent
execution without further analysis.

Goens, Castrillon, Odendahl, and Leupers
[model logical-buffer placement on complex multicore systems](https://www.sciencedirect.com/science/article/pii/S1383762116300352)
using task access intervals, physical memories, topology, and bandwidth. This
is relevant when allocation differs by launched node or creates specialized
kernels. Interval non-overlap alone is insufficient for DFBs because one
physical index also contains protocol state owned by specific hardware
processors.

Repeated DFB protocols may produce periodic or disconnected lifetimes. Cyclic
register-allocation work models software-pipelined lifetimes with
[circular-arc graphs](https://www.sciencedirect.com/science/article/pii/S0166218X99001055).
Allocation algorithms that require one connected lifetime per value apply only
after the compiler proves that property for each physical DFB version. A
separate exact constraint model, following
[Unison](https://arxiv.org/abs/1804.02452), can validate small cases that
combine index assignment, per-node specialization, lifetime splitting,
scheduling, and spilling without requiring the production allocator to use a
general solver.

#### Analysis and allocation algorithms

`DFBLogicalIdentityAnalysis` and
`DFBConcurrentKernelLivenessAnalysis` are read-only pass-manager analyses.
The liveness analysis consumes the cached logical-identity result and exposes
logical lifecycles, lifetime frontiers, boundedness, and pairwise lifetime
order. Its operation events and program-order and protocol edges remain an
internal proof representation. It does not construct the physical-index
conflict relation or select indices.

```text
resolveLogicalIdentities(module):
  maxExplicitId = maximum dfb_id on all declarations
  reject any user declaration without dfb_id
  assign each compiler-created declaration a unique ID after maxExplicitId
  reject one logical ID with inconsistent CircularBufferType values
  return declaration -> logical dfb_id

analyzeConcurrentLifetimes(module, logicalIdentities):
  logicalDFBs = group bind_cb declarations by logical dfb_id
  collect every lifecycle operation and direct runtime use
  compute the launch-node domain of every access
  if every per-node execution count is exact for an otherwise unknown access:
    use the nodes with positive counts as its exact access domain

  for each launched physical node:
    graph = empty happens-before graph
    for each single-block kernel function:
      for each top-level operation active on the node in program order:
        add operation entry and completion events
        add entry -> completion
        add previous completion -> entry

    for each logical DFB active on the node:
      if statically counted reserve/push and wait/pop runs align within their
         respective exact iteration domains, producer and consumer counts
         match, and adjacent runs have proven order, or one reserve, push,
         wait, and pop share one at-most-once condition:
        DFB.nodeLifetime.transactionRuns = normalized counted transactions
        DFB.nodeLifetime.pointerOwners = read and write hardware processors
        add DFB.push.completion -> DFB.wait.completion

    compute transitive graph reachability

    for each logical DFB with a matched lifecycle on the node:
      uses = project every active runtime use to a top-level operation
      if every use completion precedes the final pop completion:
        DFB.nodeLifetime.earliestEvents = minimal entry events in uses
        DFB.nodeLifetime.terminalEvents = {final pop completion}
        DFB.nodeLifetime.completionProof = proven

    build a second graph with every unknown access domain treated as possible
    prove conditionally bounded lifetimes only for one complete conditional
      transaction on every possible node; separately evaluated conditions
      match across logical kernels only through typed dispatch-condition
      identities at one launch coordinate
    retain possible-domain order separately from exact-domain order

  return logical DFB lifecycles, per-node lifecycle completion, pointer owners,
         conditional boundedness, source evidence, and pairwise per-node
         exact and possible-domain lifetime order
```

`DFBPhysicalAllocationPlanner` consumes those immutable facts and constructs a
typed conflict model before selecting any assignment. Every edge retains the
logical DFB pair, optional launched node, source operations, and one of the
following reasons: descriptor mismatch, storage mismatch, unknown launch-node
domain, access-completion-not-proven, transaction mismatch, pointer-owner
mismatch, reset-domain-write, static-configuration mismatch, or concurrent
lifetime.

Before coloring, each allocation group is checked pairwise with the normal
conflict construction except for exact descriptor equality. The replacement
compatibility check requires identical element types, scratch storage for an
unequal capacity envelope, compatible static compute configuration, and no
lifecycle conflict. A successful group is contracted to one graph vertex. A
failed group request is a compilation error rather than permission to allocate
its members separately.

The optional unsafe allocation-group policy changes only explicit group
validation. It accepts missing launch-domain, access-completion,
pointer-handoff, and lifetime-order proofs as user-supplied runtime epoch
contracts. Each accepted group emits a warning and a
`ttl.assumed_dfb_allocation_groups` audit record. The allocator still rejects
incompatible page formats, tensor storage, compute-kernel configuration,
mutually reachable access events, selected-reset interface writes that overlap
another member, required cumulative synchronization that contradicts
established order, and transaction sequences that cross the selected ring
envelope when started at an assumed epoch boundary. Automatic reuse and every
target-capacity and L1-budget check remain proof-based. Strict validation is
the default.

#### Allocation diagnostics

An assertions-enabled build can print the allocation inputs and conflict
evidence without changing allocation behavior:

```bash
ttlang-opt input.mlir \
  --ttl-finalize-dfb-indices='reuse-user-dfbs=true' \
  -debug-only=ttl-finalize-dfb-indices \
  -o /dev/null
```

The report is emitted after liveness and conflict construction and before
capacity or L1-budget validation. Each logical DFB records its descriptor,
tensor backing, launch-node domain, boundedness, accesses, protocol effects,
exact execution counts, transaction sizes, pointer owners, and lifetime
frontiers. Frontier entries are access-occurrence indices that refer to the
numbered access records and their source locations. Each conflict records both
logical IDs, the typed reason, an applicable launch node, and source
operations.

For each validated allocation group, the report records its member logical
IDs, maximum byte capacity, `handoff=proven` or `handoff=assumed`, and every
compatible descriptor conflict replaced by the physical envelope. Logical DFB
and final-assignment rows also retain the typed group identity.

The report evaluates each base launch node with every unknown-domain access
treated as possible. Exact-zero execution excludes an access. These rows use
`domain_assumption=unknown-possible`. A row records
`conditional_execution=1` only when every unresolved active access shares one
structured at-most-once condition and forms one complete transaction. A
logical DFB becomes `conditionally_bounded=1` only when the proof succeeds on
every possible node. Two conditionally bounded unknown-domain DFBs may share
when their descriptors, transaction runs, pointer owners, and possible-domain
lifetimes are compatible. Unknown/exact-domain pairs and all other unknown
pairs retain an `unknown-launch-node-domain` conflict. Nodes with identical
facts are grouped for deterministic bounded output.

If node-dependent IR has no launch grid, no base launch nodes are available
for the possible-domain evaluation. The report retains logical DFB and access
facts but omits per-node rows.

#### Dispatch-stable condition identity

An external predicate can be evaluated independently in multiple logical
kernels when the frontend records one typed condition declaration:

```python
def make_conditional_operation():
    active = ttl.DispatchCondition(ttl.ScalarType.I64)
    producer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    consumer_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)

    @ttl.operation(grid=(1, 1))
    def conditional_operation(input_tensor):
        producer_active = ttl.call_extern_func(
            "condition.hpp",
            "evaluate_for_producer",
            condition_result=active,
            kernel=producer_kernel,
        )
        consumer_active = ttl.call_extern_func(
            "condition.hpp",
            "evaluate_for_consumer",
            condition_result=active,
            kernel=consumer_kernel,
        )
```

`DispatchCondition` is immutable and must be captured from an enclosing
operation factory. Its declaration selects the i32 or i64 external result
carrier. Zero is false and nonzero is true. The declared truth value must be
stable for one dispatch and launch coordinate, and each evaluation must be
repeat-safe. A condition-result call cannot access a DFB, declare DFB protocol
effects, or declare unknown DFB access. DFB arguments, indices, and descriptors
are all invalid on the call.

The frontend assigns deterministic ordinals by declaration identity within the
module compiled for an operation. Composition, unified-operation splitting,
and explicit logical-kernel replication preserve those ordinals. Equal IR
attributes identify the same condition; distinct conditions in one module use
distinct ordinals. The attribute on `ttl.opaque_call` has no runtime effect and
disappears with the opaque call during lowering.

The liveness analysis proves equality from the typed identity and the actual
structured condition expression. It preserves branch polarity and ordered
nesting and supports exact `arith.andi`, `arith.ori`, and `arith.xori`
expression trees over i1 values. Cross-function proof also requires the same
launch coordinate and a typed expression for every unresolved conditional
frame. Distinct identities, missing identities, mixed typed and untyped
nesting, differing expressions, and unresolved loops remain conservative.
Callee names, source locations, generated C++, template arguments, and textual
equality do not establish condition identity.

The allocation graph uses one vertex per logical DFB and one edge per pair
that cannot share. Assigning a graph color means assigning a physical DFB
index; vertices joined by an edge must receive different indices. A clique is
a set of vertices in which every pair has an edge. Every DFB in a clique needs a
different physical index, so the clique size is a proved lower bound on the
number of required indices. The allocator finds a clique greedily. It may miss
a larger clique, but the one it finds still proves its lower bound.

First-fit processes DFBs in immutable declaration order and assigns the lowest
index not used by a conflict. This produces a valid assignment quickly, but a
different order can use fewer indices. Exact search uses deterministic DSATUR
ordering: it next selects the unassigned DFB constrained by the most different
indices among already assigned conflicts, then tries legal indices in numeric
order. Resolving the most constrained DFB first reduces failed alternatives.
DSATUR changes search order only; feasibility and infeasibility results come
from exhaustive backtracking within the configured state limit.

```text
buildPhysicalAllocationPlan(module, logicalIdentities, perNodeLifetimes):
  reject if a derived DFB-index attribute exists

  validate each used compiler-created logical DFB after aggregating all
      declarations and identity-preserving casts

  for each logical DFB pair A, B:
    add descriptor mismatch unless A.type == B.type, or both types have an
        identical page format and either:
          A and B belong to one scratch allocation group without opaque access
          A and B occupy disjoint synchronized configuration epochs without
              opaque access
    add unknown-domain conflict unless both unknown domains are conditionally
      bounded
    for each node where A and B both execute:
      if A and B share an allocation group and either lifetime has explicit
          lifecycle epochs:
        add access-completion-not-proven conflict unless every epoch completes
        add concurrent-lifetime conflict unless every cross-member epoch pair
            has exactly one proven order
      else:
        add access-completion-not-proven conflict unless both node lifetimes
            are proven
        if the node lifetimes occupy disjoint configuration epochs:
          continue
        add transaction conflict unless their transaction tile-count sequences
            match
        add pointer-owner conflict unless read and write owners match
        add concurrent-lifetime conflict unless A precedes B or B precedes A

  for each typed allocation group:
    validate every member pair with descriptor equality disabled unless either
        member has opaque external access
    reject incompatible element types or tensor-backed capacity envelopes
    reject any storage, static-configuration, protocol, owner, domain, or
      lifetime conflict
    compute the largest scratch byte capacity required by a member
    for each launch node:
      order every active member epoch by its proven event relation
      require each adjacent handoff to preserve pointer owners or follow a
          canonical reset
      advance read and write cursors through every epoch in that order
      reject unequal handoff offsets or a transaction that crosses the envelope

  if reuseUserDFBs:
    candidates = all logical DFBs in immutable declaration order
  else:
    compactedUserIndices = compactDistinctUserIndices(module)
    reject every conflicting user pair assigned the same provisional index
    compilerDFBs = logicalDFBs containing only compiler-created declarations
    candidates = compilerDFBs

  conflicts = typed conflict graph induced by candidates
  contract each validated allocation group to one allocation vertex
  assignment = deterministicFirstFit(conflicts)
  pairwiseConflictLowerBound = findCliqueLowerBound(conflicts)
  if assignment exceeds the physical-index limit:
    fitResult = exactFixedLimitSearch(conflicts, physicalIndexLimit,
                                      searchStateLimit)
    if fitResult is SearchLimitReached:
      reject with an inconclusive-search diagnostic
    if fitResult is Infeasible:
      reject with a proved capacity diagnostic
    assignment = fitResult.assignment
  reject a capacity result only after exact infeasibility or the
      pairwise-conflict lower bound exceeds the applicable limit
  verify every pair assigned one index against the typed conflict model

  aggregate L1 bytes once per unique physical index
  authoritativeDFBBudget = L1 limit - synchronized-reset scratch
      - reconfiguration state
  minimumWeightSearchTrigger = authoritativeDFBBudget
      - provisional conservative PipeNet reservation
  if allocationBytes exceeds minimumWeightSearchTrigger and
      is not known minimum:
    minimumResult = exactMinimumWeightSearch(
        conflicts, per-DFB allocation bytes, physicalIndexLimit,
        searchStateLimit)
    if minimumResult is SearchLimitReached and
        allocationBytes exceeds authoritativeDFBBudget:
      reject with an inconclusive-search diagnostic
    if minimumResult found an assignment:
      assignment = minimumResult.assignment
      allocationBytes = minimumResult.minimumBytes
    aggregate L1 bytes once per unique physical index
  reject if allocationBytes exceeds authoritativeDFBBudget
  build one initial runtime descriptor and its ordered epoch configurations
      for every physical index
  reject conflicting descriptors within one configuration epoch
  reject if the internal assignment is not a dense zero-based range
  record every existing kernel base-index attribute
  return immutable {
    logical-to-physical assignments,
    runtime descriptors,
    physical DFB count,
    kernel base indices
  }

applyPhysicalAllocationPlan(module, plan):
  write dfb_id and cb_index from plan.assignments
  write ttl.base_cta_index from plan.kernelBaseIndices
  write ttl.dfb_allocations from plan.runtimeDescriptors
```

First-fit is accepted whenever its valid assignment satisfies the physical
index limit and the provisional L1 search threshold; proving a smaller
assignment would not change compilation. Backtracking can grow exponentially,
so exact search is reserved for cases where first-fit prevents acceptance or
exceeds that provisional threshold. A physical-index failure asks one direct
question at the available index count instead of proving the minimum. A valid
assignment that exceeds the authoritative DFB-plus-fixed-state budget, or the
provisional threshold after the conservative PipeNet reservation, requires
weighted search because equal index counts can have different sums of the
maximum allocation assigned to each physical index. Each exact query examines
at most `exact-coloring-search-limit` deterministic states, which defaults to
1,000,000, to bound compile time. Reaching the limit reports that feasibility
was not proved and identifies the option that increases the limit; it never
reports a proved capacity failure. The planner completes every
diagnostic-producing validation before `TTLFinalizeDFBIndices` changes any
`dfb_id`, `cb_index`, kernel attribute, or module attribute. The finalizer only
materializes the validated plan.

Transport formation may record a conservative PipeNet L1 reservation before
finalization. That reservation lowers the threshold that triggers
weighted-allocation search, so finalization can select a fitting DFB assignment
before exact PipeNet planning. It is not an authoritative rejection condition:
finalization rejects only when DFB storage plus synchronized-reset and
reconfiguration state exceeds L1. Conversion then validates that allocation
against the exact PipeNet scratch and GlobalSemaphore requirements.

Finalization is idempotent on unchanged finalized IR. Reanalysis reconstructs
the same logical identities, typed conflicts, physical indices, descriptors,
and kernel base indices before reapplying the same values.

#### Correctness sketch

Every happens-before edge is a required execution order:

- program order within each kernel is preserved;
- the matched push must complete before the matched blocking wait can complete.

For a bounded DFB, matching lifecycle tile counts across every transaction imply
zero occupancy at the final `cb_pop` completion. The owned-use checks prove that
neither the producer nor the consumer accesses a slot after its corresponding
closing operation. Every runtime use is reachable from at least one event in
the earliest-event antichain and completes no later than the terminal pop.

Suppose A and B receive the same physical index, with A ordered before B.
The conflict predicate proves:

1. A and B have identical descriptors, use one validated scratch capacity
   envelope, or occupy disjoint synchronized configuration epochs with an
   identical page format.
2. Within one configuration epoch, ordinary reuse requires the same normalized
   transaction-run sequence. Allocation groups instead prove that the
   cumulative member sequence never crosses the shared physical envelope.
3. Within one configuration epoch, their write effects have the same hardware
   pointer owner and their read effects have the same hardware pointer owner on
   every shared launched node.
4. A's terminal pop completes before every earliest use of B, or their complete
   lifecycles occupy different configuration epochs separated by a synchronized
   boundary. Disjoint launch-node domains need no temporal relation.

Therefore A has zero occupancy and no remaining access before any producer or
consumer can begin B. An early B wait cannot consume A's data because its entry
is at or after one of B's earliest use events. The physical allocation is
sufficient for both logical DFBs.

Allocation places two DFBs at one physical index only when this relation holds
in one direction on every shared node. The final pairwise verification checks
the typed conflict model again for every shared index. If a lifetime or
ordering proof is missing, the DFB conflicts with every candidate that may
execute on the same node; this can increase the physical count but cannot
create unsafe reuse.

Two DFBs consumed by the same operation necessarily overlap: both acquires
precede the consumer and both pops follow it. Allocation therefore assigns
them different physical indices.

#### Representative example

`test/python/test_flash_chain_8node.py` composes a per-node flash-attention
atom with a three-level tree-reduction atom over eight nodes. The composed
operation contains 36 logical DFBs across the compute and data movement
kernels. Proven non-overlapping lifetimes reduce the allocation to 29 physical
indices, within every supported target capacity. The device test compares the
final result with PyTorch scaled dot-product attention.

`test/ttlang/Dialect/TTL/Transforms/dfb_concurrent_kernel_liveness.mlir`
isolates the cross-kernel ordering rules. The independent allocation-oracle
test joins 30 DFBs that all conflict pairwise with the confirmed four-DFB
order-sensitive case. First-fit uses 33 indices and the target-capacity exact
check finds a 32-index assignment. The invalid liveness test confirms that 33
mutually conflicting lifetimes are rejected without target metadata.

`test/python/test_user_dfb_reuse.py` recursively composes copy atoms into an
operation with 33 logical DFBs. Reuse reduces the physical allocation for all
targets. A focused Blackhole case disables reuse and executes all 33 distinct
indices. The same file checks a typed allocation group whose two differently
sized scratch DFBs execute in different logical kernels and share the largest
capacity envelope for BF16 and FP32 with DRAM and L1 tensors.

### Module attribute and runtime integration

The pass compacts distinct provisional user indices before placing
compiler-created DFBs after them. This removes gaps caused by DFBs that the
frontend created but no kernel captured, without changing logical identity or
existing physical-index sharing. Lifetime-based assignment may further reuse
physical indices when enabled. The planned physical DFB count is one greater
than the greatest assigned index. The pass compares this count with
`getTargetMaxDFBIndices()`, which returns 64 for Blackhole and 32 for Wormhole
B0, Quasar, and absent target metadata.

The allocation planner records the final `ttl.base_cta_index` for every kernel
that has the attribute. Compile-time arguments to each kernel reserve
`[0, physical_dfb_count)` for physical DFB indices. Reconfiguration reserves
the next argument for the caller runtime-argument count. `base_cta_index` is
the first tensor-accessor argument index after these compiler-defined entries.

The plan contains one `ttl.dfb_allocations` descriptor per physical index.
Each descriptor contains `dfb_index`, `num_tiles`, `element_type`, `page_size`,
and `block_count`. When the compiler proves the exact union of launch nodes
that access the physical index, the descriptor also contains
`allocation_nodes`. An empty array records an unreachable allocation. Omitting
the field retains conservative whole-grid allocation when the node domain is
unknown. This metadata controls storage residency independently from optional
per-core executable specialization. The planner computes `page_size` with
`ttcore::getElementSizeBytes()` on the finalized element type, so subtile
dimensions affect the physical allocation without requiring runtime device
initialization.

```text
buildRuntimeDescriptors(assignments, lifecycles, boundaryOrder):
  for physicalIndex in assignments grouped by index:
    allocationDomain = exactUnionOrUnknown(physicalIndex.launchDomains)
    for assignment in physicalIndex.assignments:
      for active lifecycle epoch, or the initial epoch when none is recorded:
        configuration = configurations[epoch.entryBoundary]
        if configuration is new:
          initialize it from assignment.type
        else if configuration differs from assignment.type:
          require both assignments belong to one scratch allocation group
          require one page format and no opaque external access
          retain the type with the largest byte capacity
        merge assignment.storage into configuration.activeDomain
    sort configurations by the proved boundary order
    copy the initial configuration into the physical descriptor fields
    if the initial configuration is tensor-backed:
      add scratch placeholder segments for cores first active in later epochs
    emit the physical descriptor with allocation_nodes = allocationDomain
    emit its ordered epoch configurations
```

Every finalized declaration contributes to the table. Exact-type reuse keeps
one unchanged descriptor. A validated allocation group selects the maximum
scratch capacity required within one epoch. A synchronized reconfiguration may
select a different geometry, block count, or storage segment in a later epoch.
For a tensor-backed initial configuration, static scratch placeholders define
the physical index on cores that first use it later without installing a future
tensor address before its lifecycle begins.
The page format remains identical across all epochs. Deriving every page size
from the same element type used by lowering keeps compiler and runtime formats
equal.

Reconfiguration supports BF16, FP32, BFP8_B, BFP4_B, U32, U16, U8, and I32
DFB formats. Pack and unpack reconfiguration is qualified for every listed
format except U8, whose compute passthrough is unsupported; U8 is qualified for
the NoC interfaces. IEEE FP16 is rejected because TTNN does not expose a native
FP16 tensor representation; its `float16` compatibility name resolves to BF16
and does not represent IEEE FP16 storage.

The runtime intersects `allocation_nodes` with final per-kernel DFB-use
metadata when both are available. The allocation domain restricts an
unspecialized grid-wide kernel, while specialized use metadata may remove
additional DFBs after coordinate folding. Tensor-backed storage segments must
cover the exact allocation nodes and retain their existing storage identity.

The Python runtime validates that the descriptors form a dense index range and
builds all `ttnn.CBDescriptor` objects from this final allocation table. It
does not use the frontend's logical DFB list after physical assignment. This
preserves compiler-computed page sizes, tile dimensions, and data formats.
Scalar element types omit the optional TTNN tile descriptor, so their page
size is not paired with an invented 32x32 tile. Standalone runner emission uses
the same physical configuration as direct execution.

Setting `reuse-user-dfbs=false` retains physical sharing already expressed by
equal provisional user indices but does not introduce new sharing between user
DFBs. It compacts the distinct provisional values, then assigns only
compiler-created DFBs. An allocation-group declaration is rejected in this
mode because its required sharing cannot be honored. Both modes use the same
concurrent lifetime proof and conflict relation for every DFB whose index they
select. Both modes emit the complete `ttl.dfb_allocations` table and assign
identities to compiler-created declarations.

```text
allocateWithoutUserReuse(module, lifetimes):
  compactedUserIndices = compactDistinctUserIndices(module)
  assign every user logical DFB its compacted provisional index

  compilerDFBs = logicalDFBs containing only compiler-created declarations
  compilerColors = colorConcurrentLifetimes(compilerDFBs)
  assign compiler color C to compactedUserIndices.size + C
```

The production pipeline emits reserve, push, wait, and pop operations for every
used compiler-created DFB. Both allocation strategies reject a DFB with only
part of that lifecycle because its bounded interval is not proven. A
declaration with no lifecycle operations is legal and conservatively remains
live through the end of its kernel. The presence check does not restrict the
number of transactions; `ttl-insert-cb-sync` is assumed to balance acquires
with releases before finalization.

## Limitations and Future Work

- **Compute stores in different MLIR basic blocks.** A `ttl.compute` operation
  is located in one MLIR basic block, and all output stores in its body execute
  whenever the operation executes. Moving stores from different MLIR basic
  blocks into one compute could change their conditional execution or violate
  SSA dominance. `ttl-insert-intermediate-dfbs` instead materializes the tensor
  result once so each original store remains in its original MLIR basic block.
  Avoiding that intermediate requires a per-region DFB occupancy proof; this is
  tracked by [#724](https://github.com/tenstorrent/tt-lang/issues/724).

- **Cross-region instrumentation.** Cross-region creation recomputes only
  side-effect-free producers. Relocatable signposts or tile-observing debug
  prints prevent creation when their observation order cannot be preserved.
  Structured profiling regions could permit a more precise containment proof.

- **Tile-range availability.** Releasing any part of an acquisition invalidates
  its complete tensor result. Tracking remaining tile ranges could prove more
  values available after partial `cb_push` or `cb_pop` operations.

- **Control-flow-dependent FIFO ownership.** Exact FIFO matching is restricted
  to the kernel entry block and is disabled when nested lifecycle operations
  make the queue control-flow-dependent. A range-aware transaction lattice
  could propagate tile counts across `RegionBranchOpInterface` edges.

- **Tensor identity.** Identity tracing accepts conversion casts, DFB
  associations, slices, and extracts. An unrecognized aliasing operation
  produces an unknown identity. Additional view operations require a semantic
  guarantee that their results alias the same acquired storage.

- **Compute recipes.** `ComputeOp` creation plans recognize only tile recipes
  and instrumentation with defined relocation semantics. An unrecognized
  operation prevents creation. Adding a recipe requires defining its input
  roles, iteration maps, instrumentation order, and output-store placement and
  execution semantics together.

- **Structured control flow and index reuse.** Lifecycle operations inside
  `scf.if`, `affine.if`, `ttl.if_src`, and `ttl.if_dst` can be bounded when
  launch-node and execution-count analysis prove one execution on the
  applicable node. Exact static loops can be bounded when matched runs share
  one iteration domain and their operations have structural order. This
  includes at-most-once regions proven to execute in every static iteration.
  Nested operations project to the enclosing kernel-body operation for
  inter-kernel ordering. Dynamic or runtime-selected repeated regions and
  multi-block functions remain conservative. More precise region-local
  ordering for those forms requires occurrence-level entry and completion
  events.
  MLIR's
  [One-Shot Bufferize](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp)
  applies the same conservative restriction when repeated regions invalidate a
  dominance-based `happensBefore` result.

- **Unresolved repeated protocols.** Static loops with matched structured
  iteration domains and exact external cumulative sequences are supported.
  Dynamic trip counts, runtime-selected iterations, unknown external
  effect order, and mixed native/external cumulative sequences remain
  conservative.

- **General credit-return ordering.** Exact cumulative external sequences add
  pop-to-reserve completion relations when capacity requires returned credit.
  Native and dynamically repeated protocols still require exact occurrence
  matching before the compiler may infer the same relation.

- **Assignment granularity.** Allocation currently selects one physical index
  per logical DFB over its complete launch-node domain. Per-node or hybrid
  assignments can reduce the maximum index count but require kernel
  specialization and per-node-range allocation metadata.

- **Pointer-owner changes.** Proving zero occupancy does not transfer ring
  pointer state between NOC0, NOC1, Pack, and Unpack. Reuse across different
  hardware pointer owners requires an explicit state transfer or reset.

- **Storage compatibility.** Automatic reuse within one configuration epoch
  requires exact `CircularBufferType` equality. Typed allocation groups support
  a scratch-only capacity envelope across different block shapes and block
  counts with one exact page format. Synchronized reconfiguration supports
  different outer geometry, block counts, and storage in disjoint epochs.
  Different element types, tile formats, or statically incompatible compute
  configurations remain invalid.

- **Pressure above the unspilled limits.** Deterministic first-fit is accepted
  when it fits because a smaller assignment would not change acceptance. One
  fixed-limit exhaustive query runs when first-fit exceeds the physical-index
  limit. Weighted-allocation search runs when a valid assignment exceeds the
  DFB-plus-fixed-state L1 budget or a provisional threshold that reserves
  conservative PipeNet resources. The authoritative finalizer budget includes
  synchronized-reset and reconfiguration state; exact combined PipeNet and
  GlobalSemaphore resources are validated during conversion. Each query is
  limited to 1,000,000 deterministic states by default so difficult graphs
  cannot make compile time unbounded. Limit exhaustion reports an inconclusive
  allocation only when acceptance requires the search result; proven
  infeasibility reports a capacity failure. DRAM spilling is tracked by
  [#809](https://github.com/tenstorrent/tt-lang/issues/809).

- **Reachability cost.** Each launch node runs one graph traversal from every
  modeled entry, completion, and external-effect event. For `V` events and `E`
  ordering edges, this costs `O(V * (V + E))`; unrelated operations are
  excluded. Launch nodes with identical active operations currently recompute
  the same ordering relation. Grouping those nodes could reduce analysis time
  for large grids.

## Scalar Element Access to DFBs

`ttl.raw_element_read`, `ttl.raw_element_write`, and `ttl.read_index` give data
movement (noc) threads per-element L1 access to DFB slots. The existing DFB
interface operates on whole blocks; these ops provide scalar access for uses
such as KV-cache updates, top-K selection, and dynamic tensor slice indices.

```python
val = ttl.raw_element_read(block, coord0, coord1, ...)
ttl.raw_element_write(block, coord0, coord1, ..., val)
index = ttl.read_index(block, coord0, coord1, ...)
```

```mlir
%v = ttl.raw_element_read %block[%i, %j] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
ttl.raw_element_write %block[%i, %j], %v : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
%index = ttl.read_index %block[%i, %j] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
```

Coordinates are flat scalar-element positions (one per tensor dimension).
For tiled blocks, lowering will decompose each coordinate into tile index +
intra-tile offset; for row-major blocks they map directly to memory offsets.
Blocks of any rank are supported.

The verifiers enforce:

1. Enclosing function is a noc kernel thread.
2. Read blocks come from `ttl.cb_wait`, and write blocks come from
   `ttl.cb_reserve`.
3. Coordinate count equals block tensor rank.
4. Scalar read and write types match the block element type.
5. Only `f32` and `bf16` block elements are accepted.

This is a scalar-access lowering restriction, not a DFB storage restriction.
The lowering maps the 32-bit and 16-bit IEEE-754 representations to TTKernel
L1 integer loads and stores. `ttl.read_index` decodes those representations
with integer operations because noc kernels do not support generic
floating-point-to-integer conversion.

`ttl.read_index` truncates fractional values toward zero and returns an MLIR
index. Its source must be finite, nonnegative, and no greater than INT32_MAX;
behavior is undefined otherwise. The element-access operations carry
appropriate `MemRead`/`MemWrite` side effects to prevent reordering across
acquire/release boundaries.

### Lowering

`convert-ttl-to-ttkernel` lowers element access ops to
`ttkernel.load_from_l1` / `ttkernel.store_to_l1` with computed L1 pointer
offsets. For tiled blocks, the offset includes face-order decomposition
(4x16x16 faces per 32x32 tile). For row-major blocks, coordinates
linearize directly. See `computeRawElementOffset` in
`ConvertTTLToTTKernel.cpp`.

### Supported Value Sources for Writes

`materializeIntBits` in `ConvertTTLToTTKernel.cpp` resolves the integer
bit pattern for `raw_element_write`. Three value sources are handled:

- Result of `raw_element_read` (via `unrealized_conversion_cast`)
- Float constants (materializes the IEEE-754 bit pattern as an integer)
- `arith.truncf` from f32 to bf16 (extracts upper 16 bits via
  shift+trunc)

Other SSA float values (e.g., from `arith.addf`) fail lowering because
the compiler cannot extract a compile-time or cast-based bit pattern.

### bf16 Implicit Truncation

When writing an f32 value to a bf16 block, the Python DSL auto-inserts
`arith.truncf`. The lowering extracts the upper 16 bits of the f32
IEEE-754 encoding, which matches the bf16 representation. This
truncation is lossy for values that are not exactly representable in
bf16.
