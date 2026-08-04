# Dataflow Buffer Management

This document describes how the tt-lang compiler manages dataflow buffers (DFBs) -- the L1-resident circular buffers that transfer data between compute and data movement threads on Tenstorrent hardware.

## Overview

DFBs originate from two sources. User-declared DFBs are created explicitly in the DSL via `make_dataflow_buffer_like` and correspond to the programmer's data movement plan. Compiler-allocated DFBs are inserted automatically when the compiler needs concrete storage for a tensor SSA value: a tensor-level operation requires a CB-attached operand, a fused expression must be preserved before a source DFB release, or a computed value is stored from multiple blocks.

The hardware supports at most 32 DFBs per node (indices 0--31). User and
compiler-allocated DFBs share this index space. Passes operating on individual
kernels assign compiler DFBs kernel-local provisional indices. The module-level
finalization pass assigns module-wide physical indices after the last
user-declared DFB and applies lifetime-based index reuse.

`ttl.bind_cb` separates logical and physical identity. `dfb_id` identifies one
logical DFB across kernel functions, while `cb_index` identifies its assigned
hardware slot. Keeping both identities allows non-overlapping logical DFBs to
share one physical index without merging their producer and consumer
protocols. Every user declaration carries `dfb_id`. Compiler-created
declarations may omit it until module finalization assigns a unique ID.

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
ttl-set-compute-kernel-config  (FuncOp)   Set per-kernel configuration
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
releases it back to the packer. `bind_cb` allocates the L1 backing storage and
is shared by both sides.

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
stores with one `ttl.compute`. It follows the same publication rule for user
DFBs.
When `ttl-create-producer-compute` or `convert-ttl-to-compute` absorbs a
block-level `ttl.store` into a `ttl.compute`, any publication that would
otherwise precede the new compute is relocated after the compute. This keeps
the generated DFB lifecycle in write-then-publish order: `cb_reserve`,
`ttl.compute` with `tile_store`, then `cb_push`. Both passes use the shared,
read-only compute-op-creation analysis described below before modifying IR.

After `cb_pop`, the producer may overwrite the released slot because its prior
contents are no longer live. The DFB's L1 backing storage remains statically
allocated. Compiler DFB index reuse uses the final pop as the end of the
contents' live interval, allowing non-overlapping DFBs of the same type to use
the same backing storage.

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
authoritative design for candidate planning, fusion, output publication,
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

The creation planner consumes the availability result at the planned
`ComputeOp` insertion point. A direct or fused candidate is legal only when
every lifetime root is definitely available there. Planned materializations remain compute roots
but are excluded from lifetime roots because their replacement DFB supplies
new storage. If another unmaterialized occurrence reads the same SSA value, it
remains a lifetime root.

Output publication planning groups stores by their `cb_reserve` operations and
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
legal. Output reserves are part of the recorded publication transaction and
are not ordering boundaries; the created `ttl.compute` necessarily executes after
them.

`ttl.compute` results preserve SSA dependencies, but the compute body publishes
data only through `ttl.tile_store`; `ttl.yield` does not carry tile values. If
one surviving use requires a compiler DFB, every surviving use of that result
must read the same pushed and waited materialization. The original user-DFB
publication remains a `ttl.tile_store` in the compute body and is not a
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
instrumentation-order boundary. If no publication satisfies those conditions,
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
reserve/push and wait/pop ownership, publication transactions, or TTL tile
recipes. The TTL-specific implementation therefore adds:

- exact acquisition identities and conservative release-owner sets;
- static FIFO tile-count matching for entry-block transactions;
- three-state planner results that distinguish a valid plan, a legal candidate
  that must remain unchanged, and malformed IR;
- complete `ComputeOp` creation and output-publication records;
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
required but exceeds either the 32-index limit or the L1 budget is rejected
with a resource diagnostic.

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

Non-compute producers use standalone tensor materialization. The helper emits
the reserve/store and wait/attach after an insertion operation recorded by the
plan, while `ttl-insert-cb-sync` inserts the missing releases:

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

When the source already has a valid output-publication plan, its final output
store is the insertion operation if it properly dominates every rewritten
consumer. Final `ComputeOp` creation may relocate source evaluation to that store;
placing the compiler reserve/store/wait afterward therefore keeps evaluation
before the wait. A source without an applicable output-publication plan remains
at its definition. If a valid publication store does not dominate every
consumer, the fixed point must require every source use. Rewriting all uses
removes the original publications, so the compiler DFB store inserted after
the definition becomes the source's only output store and therefore its compute
insertion position. The planner verifies these conditions before modifying IR.

When multiple `DFBInputOpInterface` operations consume the same non-compute
value, they share one materialization. The selected insertion operation
properly dominates every rewritten consumer, so the attached replacement is
valid for consumers in nested or incomparable regions.

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
flow are future work ([#724](https://github.com/tenstorrent/tt-lang/issues/724)).

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

- **Direct-DFB ownership** -- `U` references the DFB directly as a `ttl.copy`
  operand on the side matching the acquire's sync class (the DM-thread case,
  e.g. `ttl.copy %cb, %slice` for a writer). With no SSA tile handle,
  ownership is positional: `U` belongs to the latest acquire on
  `(cb, sync class)` that precedes it in op order. Equivalently, `U` is
  bounded between `acquire` and the next acquire on the same sync class
  (`interval.syncClassBoundary` in the pass).

The criteria are disjoint. DM-thread `ttl.copy` does not flow through
`attach_cb` (it takes the DFB directly). Compute-kernel uses always go through
`attach_cb` and never reference the DFB as a direct operand of a tile op.

#### Why two criteria

Compute threads work through SSA tile handles
(`cb_wait` result -> `attach_cb` -> `ttl.store` / compute ops), so tile-SSA
ownership applies and the next-acquire boundary is irrelevant -- SSA already
distinguishes which slot the use refers to. Data-movement kernels use direct DFB
references (`ttl.copy %cb, %slice`) where no tile handle exists, so direct-DFB
ownership uses the operation interval and its boundary to disambiguate
consecutive direct uses on the same DFB. Unifying would require changing
`ttl.copy` to take the attached tensor instead of the DFB, a dialect change
tracked as future work.

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

## Index Reuse

`TTLFinalizeDFBIndices` reduces the physical DFB count by assigning the same
index to compiler-allocated DFBs whose lifetimes do not overlap. The algorithm
runs per kernel. It ignores the kernel-local provisional indices and
assigns each kernel a disjoint physical index range after the highest
user-declared index in the module.

Two DFBs may share an index only if they have identical `CircularBufferType`
(shape, element type, block count). Since `CircularBufferType` is an MLIR
uniqued type, this is a pointer comparison. The algorithm partitions DFBs by
type and applies deterministic first-fit interval coloring within each
partition.

### Logical identity

The frontend assigns each user-declared DFB a module-wide logical `dfb_id` and
copies that ID to the declaration in every participating kernel.
Compiler-created declarations receive distinct logical IDs after the largest
explicit ID. A user declaration without `dfb_id` is rejected before physical
allocation.

Declarations with one `dfb_id` must have the same `CircularBufferType`.
Compiler-created DFBs are currently kernel-local and receive distinct logical
IDs.

Logical identity resolution is a read-only analysis. The finalizer
materializes its complete assignment on `ttl.bind_cb` only after the identity
and capacity checks succeed. Rewriting `cb_index` therefore does not merge
the SPSC or PipeNet protocol state of distinct logical DFBs.

```
resolveLogicalDFBIdentities(module):
  maxExplicitId = max(dfb_id for declarations that have dfb_id, default=-1)
  reject any user declaration without dfb_id
  compilerCount = count(compiler-created declarations without dfb_id)
  reject if maxExplicitId + compilerCount exceeds the index domain
  nextCompilerId = maxExplicitId + 1 if compilerCount > 0 else 0

  for declaration in module traversal order:
    logicalId = declaration.dfb_id
    if declaration has no dfb_id:
      logicalId = nextCompilerId++
    reject logicalId if another declaration with that ID has a different type
    assignments.append({declaration, logicalId})

  return assignments
```

Generated IDs are strictly greater than every explicit ID, and each
compiler-created declaration consumes the next ID exactly once. Generated and
explicit identities therefore cannot collide. Equal types for repeated
explicit IDs ensure that one logical identity denotes one DFB representation.

### Algorithm

```
identityAssignments = resolveLogicalDFBIdentities(module)
reject if logical identity validation fails

if any compiler-created DFB exists and a derived DFB-index attribute exists:
  reject the invalid pass order

for bindOp in compilerAllocatedBindCBOps:
  lifecycleOps = reserveOrWaitOrPushOrPopUsers(bindOp)
  if lifecycleOps is not empty and any operation kind is missing:
    reject the partial lifecycle

compact distinct user physical indices into a dense zero-based range
record the compacted index for every user declaration

nextCompilerIndex = number of distinct user physical indices
for kernel in module:
  slots = planPhysicalDFBIndices(
      kernel, compilerAllocatedBindCBOps[kernel], nextCompilerIndex, plan)
  nextCompilerIndex += slots

if nextCompilerIndex > 32:
  reject the allocation

for declaration in module:
  physicalIndex = planned user or compiler index
  reject if logicalId previously mapped to a different physicalIndex
  reject if another declaration at physicalIndex has a different exact type

apply every planned cb_index and dfb_id assignment
apply ttl.base_cta_index and ttl.dfb_allocations

planPhysicalDFBIndices(
    kernel, compilerAllocatedBindCBOps, firstPhysicalIndex, plan):
  // Assign sequential indices to kernel-body operations.
  for op in kernel.entryBlock:
    opIndex[op] = nextIdx++

  kernelEndIndex = nextIdx

  // Build half-open intervals from reserve/push/wait/pop operations.
  // Nested acquires and pops are projected to their kernel-body ancestor.
  for bindOp in compilerAllocatedBindCBOps:
    if bindOp has no lifecycle operations:
      intervals[bindOp.type].append(
          {opIndex[bindOp], kernelEndIndex, bindOp.result})
      continue
    start = min(getBodyIndex(acq) for acq in reserveOrWaitUsers(bindOp))
    end = max(getBodyIndex(pop) + 1 for pop in cbPopUsers(bindOp))
    intervals[bindOp.type].append({start, end, bindOp.result})

  // First-fit coloring per type partition. Each partition gets a contiguous
  // block of indices starting at firstPhysicalIndex + offset.
  offset = 0
  for (type, typeIntervals) in intervals:
    sort typeIntervals by start
    colors = []
    for interval in typeIntervals:
      color = first color in colors where
          interval overlaps no interval already assigned to color
      if no such color exists:
        color = append new color to colors
      colors[color].append(interval)

    // Record assignments without modifying IR.
    for (color, assignedIntervals) in colors:
      for interval in assignedIntervals:
        plan.append(bindOp[interval.value],
                    firstPhysicalIndex + offset + color)
    offset += colors.size()

  return offset
```

Compiler-created DFBs emitted by the production pipeline have reserve, push,
wait, and pop operations. The check requires the presence of all four operation
kinds but does not restrict the number of transactions. Auto-sync is assumed to
have balanced each acquire with its corresponding release before finalization.
A declaration with no lifecycle operations is legal but conservatively remains
live through the end of its kernel.

Planning separates all fallible work from mutation. Pass-order, lifecycle,
logical-identity, capacity, and metadata validation complete before any
`cb_index`, `dfb_id`, `ttl.base_cta_index`, or `ttl.dfb_allocations` update. A
failed pass therefore leaves the input IR unchanged. Type-partitioned allocation
guarantees that compiler-created DFBs sharing a planned index have one exact
type. Metadata validation applies the same requirement to every declaration at
each physical index.

Intervals are half-open. A pop at kernel-body ordinal `N` produces endpoint
`N + 1`, so the interval includes the release operation. A reserve in the next
kernel-body operation may reuse the index because it starts at `N + 1`. If a
nested pop and reserve project to the same enclosing operation, the pop ends at
`N + 1` while the reserve starts at `N`, so their intervals overlap. This
preserves correctness after nested-region ordering is discarded.

### Correctness with control flow

The algorithm assigns sequential indices to kernel-body operations only.
Structured operations (`scf.for`, `scf.if`, `ttl.compute`) occupy a single
index in this sequence; their contents are not individually numbered. Any
nested acquire or `cb_pop` is projected to its enclosing kernel-body
operation with `Block::findAncestorOpInBlock`.

Projection overestimates liveness because an interval covers the enclosing
structured operation rather than the exact nested operation. This can miss
reuse opportunities across loop bodies or mutually exclusive branches, but it
cannot assign one physical DFB index to lifetimes that may overlap at runtime.

Two DFBs consumed simultaneously by the same operation (e.g., both operands of
a matmul) necessarily have overlapping intervals because their acquires
precede the consumer and their pops follow it. First-fit coloring assigns them
different slots.

### Module attribute and runtime integration

The pass compacts distinct user physical indices before placing
compiler-created DFBs after them. This removes gaps caused by declarations that
the frontend created but no kernel captured, without changing logical identity
or existing physical-index sharing. The planned physical DFB count is one
greater than the greatest assigned index, and the pass verifies this does not
exceed `kMaxCircularBuffers` (32).

The pass then sets `ttl.base_cta_index` on every kernel function. Compile-time
arguments (CTAs) to each kernel are laid out as `[CB indices..., other args...]`.
`base_cta_index` is the starting index of the non-CB arguments -- equivalently,
one past the last CB index. CB indices occupy `[0, base_cta_index)`.

Finally, the pass builds the `ttl.dfb_allocations` module attribute with one
descriptor per physical index. Each descriptor contains `dfb_index`,
`num_tiles`, `element_type`, `page_size`, and `block_count`. The finalizer
computes `page_size` with `ttcore::getElementSizeBytes()` on the finalized
element type, so subtile dimensions affect the allocation without requiring
runtime device initialization.

```
emitDFBAllocations(declarations, plannedIndices):
  for declaration in declarations:
    index = plannedIndices[declaration]
    reject index if another declaration at that index has a different type
    allocationByIndex.insert(index, declaration.type)

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

## Limitations and Future Work

`ComputeOp` creation requires one block containing all stores of a tensor
result. Stores in different blocks have different execution conditions and
cannot be represented by one unconditional compute publication.
`ttl-insert-intermediate-dfbs` handles this by materializing the tensor result
before final compute creation, so each control-flow block stores from the same
attached compiler DFB value. A future region-aware creation plan could avoid
that intermediate when it proves per-region DFB occupancy balance; this is
tracked by [#724](https://github.com/tenstorrent/tt-lang/issues/724).

Cross-region fusion recomputes only side-effect-free producers. When the
producer's block contains relocatable signposts or tile-observing debug prints,
the planner rejects the creation because it has no cross-block placement
relation that proves the observation order is preserved. Representing profiling
scopes as structured region operations would permit a more precise containment
proof without weakening this correctness condition.

The availability lattice is not tile-range-sensitive. Releasing any part of an
acquisition invalidates its complete tensor result. Tracking remaining tile
ranges could prove more values available after partial `cb_push` or `cb_pop`
operations without changing creation semantics.

Exact FIFO matching is restricted to the kernel entry block and is disabled
for a DFB when nested lifecycle operations make that queue
control-flow-dependent. Interval ownership and dense dataflow propagation
remain conservative in these cases. A future range-aware transaction lattice
could propagate FIFO tile counts across `RegionBranchOpInterface` edges and
recover additional exact owners.

Exact tensor identity tracing accepts conversion casts, DFB associations,
slices, and extracts. An unrecognized aliasing operation produces an unknown
identity and therefore cannot prove availability. Extending the recognized
view interface can improve precision only when the operation guarantees that
its result aliases the same acquired storage.

`ComputeOp` creation plans recognize only tile recipes and instrumentation
whose relocation semantics are defined. An unrecognized operation prevents fusion rather than
assuming purity or moving an effect. Adding a recipe requires defining its
input roles, iteration maps, instrumentation order, and output publication
semantics together.

The interval model operates on a linear sequence of kernel-body operations. It
cannot distinguish between branches of an `scf.if`, so DFBs used in mutually
exclusive branches are treated as overlapping. This is conservative for
physical index reuse. More precise reuse across mutually exclusive regions
would need branch-sensitive liveness.

Index reuse is restricted to compiler-allocated DFBs. User-declared DFBs retain
their original indices because the same CB index is referenced by multiple
kernels (reader, compute, writer) to implement cross-kernel data flow. Reusing
a user index in one kernel would invalidate references in the others.

Liveness is computed at kernel-body granularity. If an acquire or `CBPopOp`
is inside a structured op, it is projected to its enclosing kernel-body
operation. This is used by loop-state materialization and by later lowering
passes that can place lifecycle ops in nested regions. The projection is safe
but may keep a physical DFB index live longer than necessary.

The type compatibility constraint prevents reuse across DFBs with different
shapes or element types, even when L1 footprints happen to match. A size-based
rather than type-based compatibility check could recover some reuse
opportunities.

## Scalar Element Access to DFBs

`ttl.raw_element_read` and `ttl.raw_element_write` give data movement (noc)
threads per-element L1 access to DFB slots. The existing DFB interface
operates on whole blocks; these ops fill the gap for use cases like KV-cache
updates, top-K, and element-level data manipulation in DM threads.

```python
val = ttl.raw_element_read(block, coord0, coord1, ...)
ttl.raw_element_write(block, coord0, coord1, ..., val)
```

```mlir
%v = ttl.raw_element_read %block[%i, %j] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
ttl.raw_element_write %block[%i, %j], %v : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
```

Coordinates are flat scalar-element positions (one per tensor dimension).
For tiled blocks, lowering will decompose each coordinate into tile index +
intra-tile offset; for row-major blocks they map directly to memory offsets.
Blocks of any rank are supported.

The verifier (`verifyRawElementOp` in `TTLOps.cpp`) enforces:

1. Enclosing function is a noc kernel thread.
2. Coordinate count equals block tensor rank.
3. Scalar type matches the block's element dtype (resolved through
   `TileType` for tiled blocks).
4. Only `f32` and `bf16` are accepted.

Both ops carry `MemRead`/`MemWrite` side effects to prevent reordering
across acquire/release boundaries.

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
