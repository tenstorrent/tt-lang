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

The simulator exposes the same constructor signature but rejects it because
simulated tensor-backed storage is not implemented.

## Pipeline

The DFB-related passes in `ttl-to-ttkernel-pipeline` execute in this order:

```
ttl-materialize-loop-state     (FuncOp)   Remove ranked-tensor scf.for iter_args
ttl-insert-copy-wait           (FuncOp)   Insert missing ttl.wait ops
ttl-annotate-l1-acc-loops      (FuncOp)   Mark user accumulation loops
ttl-create-producer-compute    (FuncOp)   Create producer ttl.compute ops
ttl-insert-intermediate-dfbs   (FuncOp)   Materialize compiler-allocated DFBs
convert-ttl-to-compute         (FuncOp)   Lower remaining tensor ops
ttl-insert-cb-sync             (FuncOp)   Insert remaining DFB synchronization
ttl-verify-pipenet-guards      (Module)   Verify PipeNet launch-node domains
ttl-verify-pipenet-schedule    (Module)   Verify PipeNet event ordering
ttl-coalesce-dfb-acquires      (FuncOp)   Coalesce adjacent DFB acquisitions
ttl-finalize-dfb-indices       (Module)   Finalize identities and allocations
ttl-set-compute-kernel-config  (FuncOp)   Resolve per-kernel configuration
  ... DST assignment, loop lowering, scheduling ...
ttl-annotate-cb-associations   (FuncOp)   Copy CB indices to tile ops
ttl-verify-dfb-spsc            (Module)   Reject DFBs shared across threads
convert-ttl-to-ttkernel        (Module)   Lower to TTKernel dialect
ttkernel-insert-inits          (Module)   Insert hardware init calls
```

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

`ttl-verify-dfb-spsc` must run after `ttl-finalize-dfb-indices` so every
`bind_cb` carries its final `cb_index` and module-wide logical `dfb_id`. The
pass requires the `ttl.dfb_allocations` module attribute emitted by successful
finalization, then verifies that every declaration and lifecycle operand has a
resolved logical ID.

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
`ttl-annotate-cb-associations`. It walks every `cb_reserve` and `cb_wait` op,
groups them by logical `dfb_id` and enclosing
`ttl.kernel_thread`-tagged `func.func`, and tracks the launch-node domain for
each producer or consumer. Distinct logical DFBs therefore remain separate
after physical allocation assigns them the same `cb_index`.

The pass rejects a DFB when two producer domains overlap or when two consumer
domains overlap. If multiple threads participate and a coordinate-dependent
predicate cannot be analyzed statically, the pass rejects the DFB rather than
assuming disjointness. The diagnostic identifies the logical `dfb_id`, the
role (producer or consumer), an overlapping launched node when available, the
participating operation sites, and the originating `ttl.bind_cb`.

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

join(predecessors):
  available only if every reachable predecessor is available

query(non-executable program point):
  available, because no runtime read occurs
```

Partial releases invalidate the complete tensor because the lattice does not
track tile ranges. Unresolved ownership also invalidates every same-kind
acquisition on the DFB. These rules may require an additional intermediate
DFB, but they cannot classify released storage as available. Dead code
analysis excludes statically non-executable blocks from this conservative
fallback; dense analysis creates no lattice there, and availability holds
vacuously because the consumer cannot execute.

#### `ComputeOp` Creation

The creation planner consumes the availability result at the program location
where it proposes to create the `ComputeOp`. A direct or fused candidate is
legal only when every lifetime root is definitely available there. Planned
materializations remain compute roots but are excluded from lifetime roots
because their replacement DFB supplies new storage. If another unmaterialized
occurrence reads the same SSA value, it remains a lifetime root.

Output-store planning groups stores by their `cb_reserve` operations and
prevents one compute from combining several reserve transactions of the same
DFB. This preserves producer-pointer order when a push moves after the created
`ttl.compute`. Kernel-wide selection and application ordering are described in
`ComputeOpCreation.md`.

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
legal. Output reserves are part of the recorded output transaction and
are not ordering boundaries; the created `ttl.compute` necessarily executes after
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

The pass treats every acquire as opening a DFB live interval. The interval
starts at `cb_reserve` or `cb_wait` and ends after the last operation that can
use the acquired slot.

DFB sync classes separate the producer side from the consumer side:
`cb_reserve`/`cb_push` form producer intervals, and `cb_wait`/`cb_pop` form
consumer intervals. Producer acquires bound other producer intervals; consumer
acquires bound other consumer intervals.

Uses inside descendant regions are projected to their ancestor operation in the
acquire's block. This conservatively places the release after the enclosing
structured op when the exact use is nested in an `scf.for` or `scf.if` body.

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
  the acquire's sync class. An opaque external call is a possible read and
  write, so it can extend either class. Identity-only `ttl.attach_cb` and
  `ttl.get_dfb_id` operations do not consume an acquired slot. With no SSA tile
  handle, ownership is positional: `U` belongs to the latest acquire on
  `(cb, sync class)` that precedes it in operation order. Equivalently, `U` is
  bounded between `acquire` and the next acquire on the same sync class
  (`interval.syncClassBoundary` in the pass).

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
   `findLastOwnedUse`.

2. **FIFO monotonicity** -- for `A_0 < A_1 < ...` on the same `(cb, sync
   class)`, the inserted releases satisfy `R_0 < R_1 < ...` in op order. The
   CB front (or back) pointer advances monotonically; out-of-order pops would
   advance it past slots whose data is still needed.

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
`findOwnedReleases` extends its release-search upper bound to the acquire's
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
precede the next acquire in the same DFB sync class because direct DFB uses are
position-based:

```
cb_wait A  ->  owned reads  ->  cb_pop A  ->  cb_wait B
                                  ^
                                  inserted release
```

Direct-CB ownership is positional: a release after the next acquire in the
same sync class is owned by that next acquire, not the earlier one. Tile-SSA
ownership is unbounded: a release placed after a tile's last use can sit past
the next acquire and still belong to the earlier interval. The pass
distinguishes these two cases by use criterion, not by a single bound.

### Algorithm

```
insertMissingReleases(func):
  reserves = all cb_reserve ops in func
  waits = all cb_wait ops in func
  pushes = all cb_push ops in func
  pops = all cb_pop ops in func

  insertReleases(reserves, pushes, cb_push)
  insertReleases(waits, pops, cb_pop)

insertReleases(acquires, releases, releaseOp):
  for acquire in acquires:
    dfb = acquire.cb
    boundary = next acquire in the same DFB sync class, projected to acquire.block

    liveEnd = latest owned use:
      direct-DFB uses are bounded by boundary
      tensor-SSA uses ignore boundary

    matching = same-block owned release on dfb
    nested = nested releases on dfb after acquire and before boundary
    if matching:
      continue

    erase nested releases
    insert releaseOp(dfb) after liveEnd
```

The same-block release check makes the pass idempotent. For direct-DFB
ownership, a release after the next acquire in the same DFB sync class belongs
to that later interval and does not satisfy the earlier acquire. For tile-SSA
ownership, an existing release past the boundary still satisfies the earlier
acquire when it follows that acquire's last owned tensor use.

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

Two DFBs may share an index only if they have identical `CircularBufferType`
(shape, element type, block count), equal transaction tile counts, and a
transaction count that divides the physical capacity. These conditions ensure
one physical allocation has one page size, capacity, data format, and legal
ring-pointer progression. `CircularBufferType` is an MLIR-uniqued type, so
exact type equality is a pointer comparison.

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
logical DFB has exactly one `cb_push` and one `cb_wait`, the blocking protocol
adds this cross-kernel edge:

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

#### Lifetimes with one reserve/push and wait/pop pair

A logical DFB is bounded only when all of these conditions hold:

- exactly one `cb_reserve`, `cb_push`, `cb_wait`, and `cb_pop` reference it;
- static execution analysis proves that each operation executes exactly once
  on the applicable launched node;
- reserve precedes push, and wait precedes pop;
- push follows all uses owned by the reserve;
- pop follows all uses owned by the wait;
- reserve, push, wait, and pop transfer the same tile count (`num_tiles`);
- the transaction tile count divides the physical DFB capacity;
- reserve and push have one known write-pointer owner, and wait and pop have
  one known read-pointer owner.

Lifecycle operations inside a statically selected `scf.if`, `affine.if`,
`ttl.if_src`, or `ttl.if_dst` region may satisfy these conditions. Repeated or
unknown execution remains unproven.

The pass runs after `ttl-insert-copy-wait`. A transfer into or out of a DFB
completes at its `ttl.wait`, whose transfer-handle operand does not identify the
DFB. Inserting that wait before the corresponding push or pop ensures the
lifecycle release follows transfer completion.

The acquire/release ownership analysis described in
[DFB Sync Insertion](#dfb-sync-insertion) supplies the owned-use checks.
Failure to prove any condition leaves the DFB unbounded.

#### External calls

Every DFB accessed by a custom function or transitive helper must appear as a
direct DFB operand of `ttl.opaque_call`. When a direct `ttl.get_dfb_id` result
is passed as an ordinary or template argument, the finalizer verifies that its
source DFB is also present as a dependency operand and rejects the call before
mutation otherwise. The compiler cannot inspect custom C++ for hidden
constants or global state, so validity of the remaining declared access set is
an external-code assumption.

Every direct DFB operand is a possible read or write from call entry through
call completion. The liveness proof does not need a read/write distinction:
either access requires the same physical allocation to remain available. The
callee must complete every synchronous and asynchronous DFB access before
returning. An external call before the terminal `cb_pop` can therefore remain
within a bounded lifetime; the same call after the pop makes the DFB unbounded.

Some external functions implement their reserve, push, wait, and pop operations
inside C++. Direct operands make their DFB access sets explicit, but the hidden
protocol cannot supply the exact visible lifecycle required by the reuse proof.
Those DFBs remain unbounded and conflict with every other allocation candidate.
The external call does not disable reuse among other DFBs whose visible
lifecycles remain bounded.

The current DSL cannot declare a dependency-only DFB, summarize hidden
protocol effects, or represent an unknown DFB access set. An external callee
with an unknown set is outside the valid-program assumption. A future explicit
unknown form must disable user DFB reuse for the complete module because an
unresolved raw index may name any physical allocation. Issue
[#806](https://github.com/tenstorrent/tt-lang/issues/806) tracks the required
DFB dependency and protocol-effect representation.

`num_tiles` counts tiles of the DFB's `TileType`. TT-Lang configures each
tiled CB page from the byte size of that tile. Two 16x32 bf16 tiles therefore
consume the same bytes as one 32x32 bf16 tile. Tile dimensions remain part of
the `CircularBufferType`, so DFBs with different tile dimensions cannot share
a physical index.

TT-Metal advances each ring pointer by
[`num_pages * fifo_page_size`](https://github.com/tenstorrent/tt-metal/blob/e908c31332b60860ed0d4186452dc880cdd5a81d/tt_metal/hw/inc/api/dataflow/dataflow_api.h#L208-L214).
The pointer wraps only when it reaches the end of the physical DFB. Logical
DFBs sharing one physical index therefore use the same transaction tile count,
and that count divides `block_count * elements_per_block`. This keeps every
reserve, push, wait, and pop within the allocation and places each pointer on a
legal wrap boundary.

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
same hardware pointer processors. A quiescent handoff does not transfer pointer
state between different processors.

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

  for each launched physical node:
    graph = empty happens-before graph
    for each single-block kernel function:
      for each top-level operation active on the node in program order:
        add operation entry and completion events
        add entry -> completion
        add previous completion -> entry

    for each logical DFB active on the node:
      if exactly one reserve, push, wait, and pop form a matched lifecycle:
        DFB.nodeLifetime.transactionTileCount = transactionTileCount
        DFB.nodeLifetime.pointerOwners = read and write hardware processors
        add DFB.push.completion -> DFB.wait.completion

    compute transitive graph reachability

    for each logical DFB with a matched lifecycle on the node:
      uses = project every active runtime use to a top-level operation
      if every use completion precedes DFB.pop.completion:
        DFB.nodeLifetime.earliestEvents = minimal entry events in uses
        DFB.nodeLifetime.terminalEvents = {DFB.pop.completion}
        DFB.nodeLifetime.quiescence = proven

  return logical DFB lifecycles, per-node quiescence, pointer owners,
         source evidence, and pairwise per-node lifetime order
```

`DFBPhysicalAllocationPlanner` consumes those immutable facts and constructs a
typed conflict model before selecting any assignment. Every edge retains the
logical DFB pair, optional launched node, source operations, and one of the
following reasons: descriptor mismatch, unknown launch-node domain, unproven
quiescence, transaction mismatch, pointer-owner mismatch, or concurrent
lifetime.

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
    add descriptor mismatch if A.type != B.type
    add unknown-domain conflict if either launch-node domain is unknown
    for each node where A and B both execute:
      add unproven-quiescence conflict unless both node lifetimes are proven
      add transaction conflict unless their transaction sizes match
      add pointer-owner conflict unless read and write owners match
      add concurrent-lifetime conflict unless A precedes B or B precedes A

  if reuseUserDFBs:
    candidates = all logical DFBs in immutable declaration order
  else:
    compactedUserIndices = compactDistinctUserIndices(module)
    reject every conflicting user pair assigned the same provisional index
    compilerDFBs = logicalDFBs containing only compiler-created declarations
    candidates = compilerDFBs

  conflicts = typed conflict graph induced by candidates
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
  if assignment exceeds the L1 limit and is not known minimum:
    minimumResult = exactMinimumIndexSearch(
        conflicts, pairwiseConflictLowerBound, searchStateLimit)
    if minimumResult is SearchLimitReached:
      reject with an inconclusive-search diagnostic
    assignment = minimumResult.assignment
    aggregate L1 bytes once per unique physical index
  reject if the assignment exceeds the L1 limit
  build one runtime descriptor for every physical index
  reject conflicting descriptors at one physical index
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
index and L1 limits; proving a smaller assignment would not change compilation.
Backtracking can grow exponentially, so exact search is reserved for cases
where first-fit prevents acceptance. A physical-index failure asks one direct
question at the available index count instead of proving the minimum. A valid
assignment that exceeds L1 requires a minimum physical-index-count search
because a different sharing assignment may use less physical storage. Each
exact query examines at most `exact-coloring-search-limit` deterministic states,
which defaults to 1,000,000, to bound compile time. Reaching the limit reports
that feasibility was not proved and identifies the option that increases the
limit; it never reports a proved capacity failure. The planner completes every
diagnostic-producing validation before `TTLFinalizeDFBIndices` changes any
`dfb_id`, `cb_index`, kernel attribute, or module attribute. The finalizer only
materializes the validated plan.

Finalization is idempotent on unchanged finalized IR. Reanalysis reconstructs
the same logical identities, typed conflicts, physical indices, descriptors,
and kernel base indices before reapplying the same values.

#### Correctness sketch

Every happens-before edge is a required execution order:

- program order within each kernel is preserved;
- the matched push must complete before the matched blocking wait can complete.

For a bounded DFB, matching lifecycle tile counts and one push/pop pair imply
zero occupancy at `cb_pop` completion. The owned-use checks prove that neither
the producer nor the consumer accesses the slot after its corresponding
release. Every runtime use is reachable from at least one event in the
earliest-event antichain and completes no later than the terminal pop.

Suppose A and B receive the same physical index, with A ordered before B.
The conflict predicate proves:

1. A and B have the same page shape, data format, and block count.
2. They use the same transaction tile count, and that count divides their
   physical capacity. Their ring pointers therefore advance by equal increments
   and wrap only at the allocation boundary.
3. On every shared launched node, their write effects have the same hardware
   pointer owner and their read effects have the same hardware pointer owner.
4. On every shared launched node, A's terminal pop completes before every
   earliest use of B. Disjoint launch-node domains need no temporal relation.

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
indices.

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
`[0, base_cta_index)` for physical DFB indices; `base_cta_index` is the first
non-DFB argument index.

The plan contains one `ttl.dfb_allocations` descriptor per physical index.
Each descriptor contains `dfb_index`, `num_tiles`, `element_type`, `page_size`,
and `block_count`. The planner computes `page_size` with
`ttcore::getElementSizeBytes()` on the finalized element type, so subtile
dimensions affect the physical allocation without requiring runtime device
initialization.

```text
buildRuntimeDescriptors(assignments):
  for assignment in assignments:
    reject assignment.physicalIndex if another assignment at that index
        has a different exact type
    allocationByIndex.insert(assignment.physicalIndex, assignment.type)

  for (index, type) in allocationByIndex sorted by index:
    emit {dfb_index = index,
          num_tiles = type.elementsPerBlock,
          element_type = type.elementType,
          page_size = byteSize(type.elementType),
          block_count = type.blockCount}
```

Every finalized declaration contributes to the table. Type equality makes
each deduplicated descriptor valid for every declaration at that physical
index, and deriving the page size from the same element type used by lowering
keeps compiler and runtime allocation sizes equal.

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
compiler-created DFBs. Both allocation modes use the same concurrent lifetime
proof and conflict relation for every DFB whose index they select. Both modes
emit the complete `ttl.dfb_allocations` table and assign identities to
compiler-created declarations.

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
  applicable node. Nested operations project to the enclosing kernel-body
  operation for inter-kernel ordering. Repeated regions and multi-block
  functions remain conservative. More precise region-local ordering requires
  occurrence-level entry and completion events.
  MLIR's
  [One-Shot Bufferize](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp)
  applies the same conservative restriction when repeated regions invalidate a
  dominance-based `happensBefore` result.

- **Repeated protocols.** The analysis accepts one reserve/push/wait/pop
  occurrence per logical DFB. Loops and multi-acquire protocols require
  symbolic occurrence matching so a push, wait, and pop from the same
  iteration are related without conflating different iterations. PipeNet
  schedule verification already pairs static protocol occurrences and uses
  `ExecutionCountAnalysis` to prove equal dynamic counts. DFB reuse requires
  the corresponding reserve/push/wait/pop occurrence matching.

- **Credit-return ordering.** Only push-to-wait completion is modeled across
  kernels. Proving additional pop-to-reserve ordering could shorten later
  producer frontiers, but requires exact protocol and occurrence matching.
  PipeNet schedule verification proves receiver post/send/wait correspondence,
  but does not establish a DFB pop-to-reserve credit-return edge.

- **Assignment granularity.** Allocation currently selects one physical index
  per logical DFB over its complete launch-node domain. Per-node or hybrid
  assignments can reduce the maximum index count but require kernel
  specialization and per-node-range allocation metadata.

- **Pointer-owner changes.** Proving zero occupancy does not transfer ring
  pointer state between NOC0, NOC1, Pack, and Unpack. Reuse across different
  hardware pointer owners requires an explicit state transfer or reset.

- **Storage compatibility.** Exact `CircularBufferType` equality forbids reuse
  across different block shapes, tile dimensions, element types, or block
  counts. A broader compatibility relation would need one physical descriptor
  that satisfies every logical DFB assigned to it, including page size,
  capacity, and data format.
  A max-capacity descriptor could support some unequal logical types, but it
  must also prove compatible transaction counts and address calculations.

- **Pressure above the unspilled limits.** Deterministic first-fit is accepted
  when it fits because a smaller assignment would not change acceptance. One
  fixed-limit exhaustive query runs when first-fit exceeds the physical-index
  limit. Minimum physical-index-count search runs only when a valid assignment
  exceeds the L1 budget. Each query is limited to 1,000,000 deterministic
  states by default so difficult graphs cannot make compile time unbounded.
  Limit exhaustion reports an inconclusive allocation; proven infeasibility
  reports a capacity failure. DRAM spilling is tracked by
  [#809](https://github.com/tenstorrent/tt-lang/issues/809).

- **Reachability cost.** Each launch node runs one graph traversal from every
  top-level DFB-accessing operation. For `V` operation events and `E` ordering
  edges, this costs `O(V * (V + E))`; unrelated operations are excluded. Launch
  nodes with identical active operations currently recompute the same ordering
  relation. Grouping those nodes could reduce analysis time for large grids.

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
