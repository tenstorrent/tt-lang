# Dataflow Buffer Management

This document describes how the tt-lang compiler manages dataflow buffers (DFBs) -- the L1-resident circular buffers that transfer data between compute and data movement threads on Tenstorrent hardware.

## Overview

DFBs originate from two sources. User-declared DFBs are created explicitly in the DSL via `make_dataflow_buffer_like` and correspond to the programmer's data movement plan. Compiler-allocated DFBs are inserted automatically at fusion split points where a tensor-level operation requires a CB-attached operand but receives the result of a fused expression chain.

The hardware supports at most 32 DFBs per node (indices 0--31). User and
compiler-allocated DFBs share this index space. DFB-creating function passes
assign compiler DFBs function-local provisional indices. The module-level
finalization pass assigns module-wide physical indices after the last
user-declared DFB and applies lifetime-based index reuse.

`ttl.bind_cb` separates logical and physical identity. `dfb_id` identifies one
logical DFB across kernel functions, while `cb_index` names its assigned
hardware slot. Keeping both identities allows non-overlapping logical DFBs to
share one physical index without merging their producer/consumer protocols.

## Pipeline

The DFB-related passes in `ttl-to-ttkernel-pipeline` execute in this order:

```
ttl-materialize-loop-state     (FuncOp)   Remove ranked-tensor scf.for iter_args
ttl-insert-copy-wait           (FuncOp)   Insert missing ttl.wait ops
ttl-annotate-l1-acc-loops      (FuncOp)   Mark user accumulation loops
ttl-form-producer-compute      (FuncOp)   Form producer compute regions
ttl-insert-intermediate-dfbs   (FuncOp)   Materialize compiler-allocated DFBs
convert-ttl-to-compute         (FuncOp)   Lower remaining tensor ops
ttl-auto-sync                  (FuncOp)   Insert/coalesce remaining DFB sync
ttl-finalize-dfb-indices       (Module)   Index reuse + limit check
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

`ttl-verify-dfb-spsc` must run after `ttl-finalize-dfb-indices` so every
`bind_cb` carries its final `cb_index`. Reuse finalization also preserves the
stable `dfb_id`; legacy IR falls back to its current `cb_index`. The pass
asserts on unresolvable identities.

## DFB Lifecycle

A DFB has two lifecycle halves: the producer (write) side driven by
`cb_reserve`/`cb_push`, and the consumer (read) side driven by
`cb_wait`/`cb_pop`. For user-declared DFBs these halves span different
threads: data movement writes to the CB, compute reads from it, and both
threads reference the same CB index. For compiler-allocated intermediate DFBs
both halves are in the same compute function.

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

Producer compute formation follows the same publication rule for user DFBs.
When `ttl-form-producer-compute` or `convert-ttl-to-compute` absorbs a
block-level `ttl.store` into a `ttl.compute`, any producer release that would
otherwise precede the new compute is replaced after the compute. This keeps the
generated DFB lifecycle in write-then-publish order: `cb_reserve`,
`ttl.compute` with `tile_store`, then `cb_push`.

A DFB's L1 memory is reclaimable after its last `cb_pop`. This defines the
interval used for index reuse.

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
each producer or consumer. Legacy IR without `dfb_id` uses `cb_index`.
Distinct logical DFBs therefore remain separate after physical allocation
assigns them the same `cb_index`.

The pass rejects a DFB when two producer domains overlap or when two consumer
domains overlap. If multiple threads participate and a coordinate-dependent
predicate cannot be analyzed statically, the pass rejects the DFB rather than
assuming disjointness. The diagnostic identifies the physical `cb_index`, the
role (producer or consumer), an overlapping launched node when available, the
participating operation sites, and the originating `ttl.bind_cb`.

See `test/ttlang/Dialect/TTL/Transforms/verify_dfb_spsc_invalid.mlir` for the rejected patterns and `verify_dfb_spsc.mlir` for the accepted ones.

The compiler does not currently auto-split overlapping multi-consumer DFBs;
users must duplicate explicitly via `make_dataflow_buffer_like`. Tracked in
[tenstorrent/tt-lang#581](https://github.com/tenstorrent/tt-lang/issues/581).

## Intermediate DFB Insertion

`TTLInsertIntermediateDFBs` walks all operations implementing
`DFBInputOpInterface`, including reduce, block broadcast, matmul, transpose,
and selected elementwise forms that require DFB-attached operands. For each
operand that the interface marks as requiring a DFB-attached value, the pass
checks whether the operand traces to an existing DFB via `getAttachedCB`. If
not, the pass materializes the value through a fresh compiler-allocated DFB
marked with `ttl.compiler_allocated`.

The standard pipeline runs this pass after `ttl-form-producer-compute`.
Values produced by `ttl.compute` are materialized by the compiler-created
intermediate lifecycle described above: the compute gains extra DFB outputs,
and consumers receive attached tensor values instead of the original
non-attached compute results. The final `convert-ttl-to-compute` pass lowers
consumers that now receive DFB-attached operands. The following `ttl-auto-sync`
run inserts the consumer `cb_pop`.

Compute-result materialization is planned before rewriting IR. The pass records
each required consumer operand under its original producer result:

```
source = (producer ttl.compute, result number)
use    = (consumer operation, operand number)
```

For each producer `ttl.compute`, the pass rebuilds the compute exactly once
using this sequence:

1. Preserve all original results in their existing result order.
2. Append one compiler-allocated DFB output for each source result that needs
   materialization, ordered by source result number.
3. Clone the original compute body.
4. For each cloned `ttl.tile_store` that writes the original source DFB, emit
   a matching `ttl.tile_store` to the appended compiler DFB output.
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
The helper chooses a provisional index by scanning only the enclosing function,
so DFB-creating function passes do not inspect sibling functions while MLIR
executes them concurrently.

Non-compute producers use the tensor-level fallback. The helper emits the
reserve/store and wait/attach at the tensor definition site, while
`ttl-auto-sync` inserts the missing releases:

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

When multiple `DFBInputOpInterface` operations consume the same non-compute
value, a materialization is shared only when its attached value dominates the
later consumer. Incomparable consumers receive separate compiler DFBs.

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

- **Direct-CB ownership** -- `U` references the CB directly as a `ttl.copy`
  operand on the side matching the acquire's sync class (the DM-thread case,
  e.g. `ttl.copy %cb, %slice` for a writer). With no SSA tile handle,
  ownership is positional: `U` belongs to the latest acquire on
  `(cb, sync class)` that precedes it in op order. Equivalently, `U` is
  bounded between `acquire` and the next acquire on the same sync class
  (`interval.syncClassBoundary` in the pass).

The criteria are disjoint. DM-thread `ttl.copy` does not flow through
`attach_cb` (it takes the CB directly). Compute-thread uses always go through
`attach_cb` and never reference the CB as a direct operand of a tile op.

#### Why two criteria

Compute threads work through SSA tile handles
(`cb_wait` result -> `attach_cb` -> `ttl.store` / compute ops), so tile-SSA
ownership applies and the next-acquire boundary is irrelevant -- SSA already
distinguishes which slot the use refers to. DM threads use direct CB
references (`ttl.copy %cb, %slice`) where no tile handle exists, so direct-CB
ownership is the fallback and the boundary is essential to disambiguate
consecutive direct uses on the same CB. Unifying would require changing
`ttl.copy` to take the attached tensor instead of the CB, a dialect change
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
thread cannot make progress until all members are released. Same argument for
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
index to logical DFBs whose lifetimes cannot overlap. The default analysis
considers all kernel functions concurrently, including user-declared DFBs
shared across data movement and compute threads.

Two DFBs may share an index only if they have identical `CircularBufferType`
(shape, element type, block count). This ensures one physical allocation has
one page size, capacity, and data format.

### Logical identity

The frontend assigns each user-declared DFB a stable `dfb_id` and copies it to
the declaration in every participating kernel function. Compiler-created
declarations receive distinct logical identities. Legacy IR without `dfb_id`
uses the provisional `cb_index`.

The finalizer records every resolved logical identity on `ttl.bind_cb` before
rewriting `cb_index`. Repeated finalization therefore cannot merge logical
DFBs that already share a physical slot.

### Multithreaded lifetime analysis

The analysis constructs one entry event and one completion event for each
function-level operation. Program order connects consecutive operations in
each kernel function. For a statically matched one-shot DFB lifecycle,
`cb_push` completion precedes the corresponding blocking `cb_wait`
completion. The existing acquire/release ownership analysis verifies that
`cb_push` and `cb_pop` follow all uses of their acquired tensor views.

A balanced one-shot DFB has an earliest-event antichain and a terminal
`cb_pop` completion. Logical DFB A precedes logical DFB B only when A's
terminal event happens before every event in B's earliest antichain. Otherwise
the DFBs interfere. Requiring the complete frontier prevents reuse when an
operation in another thread can begin before the synchronization ordering A
and B.

The analysis does not infer dynamic occurrence matching for loops,
multi-acquire protocols, conditional lifecycle operations, or credit-return
ordering from `cb_pop` to `cb_reserve`. Such DFBs are unbounded and interfere
with every same-type logical DFB. Nested non-lifecycle uses project to their
enclosing function-level operation.

The analysis partitions logical DFBs by exact type and greedily colors each
interference graph. Each color is one physical index. The allocator verifies
that no interfering pair received the same index and that the final count does
not exceed the hardware limit. The analysis object does not modify IR; the
finalizer applies the complete verified assignment afterward.

### Module attribute and runtime integration

The finalizer updates `ttl.base_cta_index` on every function and emits
`ttl.dfb_allocations` with one descriptor per physical index. Each descriptor
contains `dfb_index`, `num_tiles`, `element_type`, and `block_count`.
Compile-time arguments to each kernel reserve `[0, base_cta_index)` for these
physical DFB indices.

The Python runtime validates that the descriptors form a dense index range and
builds all `ttnn.CBDescriptor` objects from this final allocation table. It
does not use the frontend's logical DFB list after physical assignment. This
also makes standalone runner emission use the same physical sizes and data
formats as direct execution.

Setting `reuse-user-dfbs=false` selects the legacy compiler-only allocator. It
retains user indices, applies per-function linear-scan allocation to
compiler-created DFBs, and emits `ttl.compiler_allocated_dfbs`. The runtime
continues to support that attribute for compatibility.

## Limitations and Future Work

Liveness is computed at function-level granularity. Operations nested in
structured control flow project to the enclosing function-level operation.
This prevents unsafe reuse but may keep a physical index live longer than its
executed lifetime.

Loops, conditional lifecycles, and repeated acquire/release protocols remain
unbounded because the analysis does not match dynamic event occurrences. They
cannot reuse an index with another logical DFB of the same type.

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
