# Dataflow Buffer Management

This document describes how the tt-lang compiler manages dataflow buffers (DFBs) -- the L1-resident circular buffers that transfer data between compute and data movement threads on Tenstorrent hardware.

## Overview

DFBs originate from two sources. User-declared DFBs are created explicitly in the DSL via `make_dataflow_buffer_like` and correspond to the programmer's data movement plan. Compiler-allocated DFBs are inserted automatically at fusion split points where a tensor-level operation requires a CB-attached operand but receives the result of a fused expression chain.

The hardware supports at most 32 DFBs per node (indices 0--31). User and compiler-allocated DFBs share this index space. The compiler assigns indices sequentially during insertion (starting *after* the last user-declared DFB), then applies lifetime-based index reuse to reduce the physical DFB count.

## Pipeline

The DFB-related passes in `ttl-to-ttkernel-pipeline` execute in this order:

```
ttl-insert-intermediate-dfbs   (FuncOp)   Create compiler-allocated DFBs
ttl-insert-cb-sync             (FuncOp)   Insert cb_push / cb_pop
  ... compute lowering, DST assignment, loop lowering ...
ttl-finalize-dfb-indices       (Module)   Index reuse + limit check
ttl-annotate-cb-associations   (FuncOp)   Copy CB indices to tile ops
ttl-verify-dfb-spsc            (Module)   Reject DFBs shared across threads
convert-ttl-to-ttkernel        (Module)   Lower to TTKernel dialect
ttkernel-insert-inits          (Module)   Insert hardware init calls
```

`ttl-finalize-dfb-indices` must precede `ttl-annotate-cb-associations` because annotation copies the `cb_index` attribute from `BindCBOp` onto tile operations (`bcast`, `reduce`, `transpose`). If annotation runs before finalization, the copied indices become stale after reuse rewrites them.

`ttl-verify-dfb-spsc` must run after `ttl-finalize-dfb-indices` so every `bind_cb` carries its final `cb_index`. The pass asserts on unresolvable indices.

## DFB Lifecycle

A DFB has two lifecycle halves: the producer (write) side driven by `cb_reserve`/`cb_push`, and the consumer (read) side driven by `cb_wait`/`cb_pop`. For user-declared DFBs these halves span different threads: data movement writes to the CB, compute reads from it, and both threads reference the same CB index. For compiler-allocated intermediate DFBs both halves are in the same compute function.

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

`cb_reserve` claims a buffer slot for the packer; `cb_push` releases that slot to the unpacker. `cb_wait` blocks until the slot is available; `cb_pop` releases it back to the packer. `bind_cb` allocates the L1 backing storage and is shared by both sides.

For compiler-allocated DFBs, `InsertIntermediateDFBs` creates the full sequence from `bind_cb` through `attach_cb`. `InsertCBSync` adds `cb_push` (after the producer's last use) and `cb_pop` (after the consumer's last use).

A DFB's L1 memory is reclaimable after its last `cb_pop`. This defines the interval used for index reuse.

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
groups them by `cb_index` and enclosing `ttl.kernel_thread`-tagged
`func.func`, and tracks the launch-node domain for each producer or consumer.
The pass rejects a DFB when two producer domains overlap or when two consumer
domains overlap. If multiple threads participate and a coordinate-dependent
predicate cannot be analyzed statically, the pass rejects the DFB rather than
assuming disjointness. The diagnostic identifies the cb_index, the role
(producer or consumer), an overlapping launched node when available, the
participating operation sites, and the originating `ttl.bind_cb`.

See `test/ttlang/Dialect/TTL/Transforms/verify_dfb_spsc_invalid.mlir` for the rejected patterns and `verify_dfb_spsc.mlir` for the accepted ones.

The compiler does not currently auto-split overlapping multi-consumer DFBs;
users must duplicate explicitly via `make_dataflow_buffer_like`. Tracked in
[tenstorrent/tt-lang#581](https://github.com/tenstorrent/tt-lang/issues/581).

## Intermediate DFB Insertion

`TTLInsertIntermediateDFBs` walks all operations implementing `DFBInputOpInterface` (reduce, bcast, matmul, transpose). For each operand that the interface marks as requiring a CB-attached value, the pass checks whether the operand traces to an existing CB via `getAttachedCB`. If not, the pass materializes the value through a fresh DFB: `bind_cb`, `cb_reserve`, `store`, `cb_wait`, `attach_cb`. The new DFB receives the `ttl.compiler_allocated` marker attribute.

When multiple `DFBInputOpInterface` operations consume the same non-CB-attached value, the materialization is shared -- only one DFB is created and the second consumer's operand is rewritten to the existing attached value.

Each DFB is created with `blockCount=2` (double-buffering) so the packer and unpacker can operate on different halves simultaneously within the same thread.

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
consumers under criterion (a) appear in declaration order
(`use(t1); use(t2); use(t3)`). Reordered consumes (`use(t2); use(t1)`) would
violate FIFO monotonicity on their own, but in the current pipeline `TTLCoalesceDFBAcquires`
runs immediately after `TTLInsertCBSync` and rewrites N consecutive same-DFB
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
must follow the last owned use and precede the next acquire in the same DFB sync
class:

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

    matching = same-block release on dfb after acquire and before boundary
    nested = nested releases on dfb after acquire and before boundary
    if matching:
      continue

    erase nested releases
    liveEnd = last transitive tensor or direction-matched direct DFB use
              before boundary
    insert releaseOp(dfb) after liveEnd
```

The same-block release check makes the pass idempotent. A release after the
next acquire in the same DFB sync class belongs to that later interval and does
not satisfy the earlier acquire.

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
schedule. The coalesced acquire blocks until `N*k` tiles are present
*before* anything between original `a_i` and `a_{i+1}` runs; the
coalesced release runs only after the last group member's last use.

This holds iff no op between members causes a release on `c` (directly or
transitively): the original IR may have allowed the producer to recycle
slots between `a_i` and `a_{i+1}`, and the coalesced version forbids that
until the very end. Forbidding inter-member releases is therefore
necessary for correctness at low `block_count`, and sufficient when paired
with the coalesced release placement.

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

If `O` is itself a release on `c` (e.g., a user-written `cb_pop`), the
original IR lets the producer recycle one slot at `O`, but the coalesced
acquire holds all `N*k` slots from the start. With `block_count` only
slightly larger than the working set, the producer cannot push the next
batch and the consumer cannot release until all members are consumed —
deadlock. Same argument for transitive releases via group results.

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

`TTLFinalizeDFBIndices` is a module pass that reduces the physical DFB count by overlaying logical DFBs with disjoint lifetimes onto shared indices. It runs after `InsertCBSync` and `LowerToLoops`, when threads are separate `func.func`s and `cb_pop`s exist. A logical DFB is one `cb_index`; each kernel thread that touches it binds it once.

A physical index carries shared `pages_received`/`pages_acked` counters plus per-RISC read/write pointers. None of those are reset between lifetimes, so two logical DFBs may share an index only when:

- both have the same producer thread and the same consumer thread,
- their lifetimes are disjoint in both threads' program order,
- they have the same element type (one page size).

When thread identity matches, per-thread program order plus blocking reserve/wait make disjoint lifetimes sound at runtime; sharing across different producer or consumer threads would corrupt counters/pointers and would need a barrier-plus-reset edge, which does not exist. The element-type bar is the only shape constraint: capacity is a `>=` check, the slot is sized to the member needing the most pages, and every reserve/wait carries its own block size.

A DFB whose producer and consumer are the same thread is single-threaded (compiler-allocated intermediates always; user DFBs whenever the pass derives it), so the pair (thread, thread) acts as one reuse class regardless of how the buffer was declared.

Lifetimes are computed per thread: from the first reserve/wait (bind position when no acquire exists) to the last use, following wait -> readers -> attach_cb -> consumers and pops. Nested uses project onto their top-level ancestor, so a buffer used in a loop is live across the back-edge for the whole loop. Intervals are closed; sharing an endpoint counts as overlap.

The reuse condition is checked against actual blocking acquires; control-flow guards (e.g. mutually exclusive per-core roles) do not refine it. `ttl-verify-dfb-spsc` always runs afterwards and rejects shared indices with multiple producer or consumer threads on overlapping launch nodes; role-guarded accesses pass when launch-node domain analysis proves them disjoint.

### Pipe-attached DFBs never reuse

For a pipe destination, `block_count` is not queue depth: each sender writes its own block, addressed by sender ordinal, so `block_count` must equal the sender count. Pipe lowering lays sender slots out per `cb_index`; two destinations on one index would concatenate their slot ranges and change the addressing. Pipe transfers also move data over NOC with semaphore protocol, outside the CB counters that make program-order lifetimes sound, and a sender's stage buffer is read asynchronously after its wait. Any DFB whose user chain reaches a pipe op or a `ttl.copy` with a pipe operand therefore keeps its dedicated index and block count.

### Runtime integration

After rewriting indices, the pass recomputes `getNextAvailableDFBIndex` (max index + 1), checks `kMaxCircularBuffers` (32), and updates `ttl.base_cta_index` on every function. Two module attributes carry the result to the Python runtime: `ttl.compiler_allocated_dfbs` (one entry per physical index, sized to the largest member) and `ttl.dfb_index_map` (old -> new entries for moved user DFBs, applied to the CB config list before descriptors are built).

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
