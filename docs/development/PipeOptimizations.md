# Pipe Communications Optimization

## 1. Introduction

This document specifies proposed lowering rewrites for `ttl.create_pipe`
that select the NoC primitive based on the PipeNet pattern. The current
`convert-ttl-to-ttkernel` policy is one fixed primitive per Pipe shape
(`noc_async_write_multicast` for `slice` destinations,
`noc_async_write` for point destinations, slot-per-pipe receiver
dataflow buffers when multiple pipes target the same receiver). This
policy is independent of
the receiver compute and the destination geometry, and emits a
sub-optimal primitive for several common patterns. The
performance-tuned tt-metal kernels (`minimal_matmul`,
`llama_all_gather_matmul_async`, `reduce_scatter_minimal_async`)
select different primitives for these patterns by hand. Closing the
gap is a lowering-policy change, not a dialect change.

Goals:

1. Pick the lowering primitive for each PipeNet from the receiver
   compute and the destination geometry, not just from the Pipe shape.
2. Emit the primitives that the tuned tt-metal kernels use for each
   pattern (forwarding chain for 1→K broadcast, ring for N→1
   reductions), without changing the user-facing PipeNet language.
3. Remove the limit on overlapping multicast width. Today, when `N`
   pipes target the same receiver and share its dataflow buffer, the
   receiver's `block_count` must be at least `N` because each pipe
   gets its own dedicated slot in that buffer. With the tt-metal
   per-Tensix CB cap of 32 (`NUM_CIRCULAR_BUFFERS` in
   [`tt_metal/llrt/hal.hpp:409-411`][hal-num-cb]), this constrains
   overlapping multicast to `N <= 32`. The compiler should let one
   wide overlapping multicast lower without that constraint.

## 2. Background

Currently `convert-ttl-to-ttkernel` lowers each `ttl.create_pipe` to one fixed
primitive based on the Pipe shape. The mapping is:

| Pipe shape | Primitive | Receiver dataflow buffer |
| --- | --- | --- |
| `Pipe(src=p, dst=p')` (point) | `noc_async_write` + `noc_semaphore_inc` | `block_count = 2` |
| `Pipe(src=p, dst=slice(...))` (rectangular multicast) | `noc_async_write_multicast` + `noc_semaphore_inc_multicast` | `block_count = max gather slot + 1` |
| Loopback multicast (`src` in dst range) | `noc_async_write_multicast_loopback_src` + remote `inc_multicast` + local `noc_semaphore_inc` | same as above |

When several pipes target the same receiver and share its dataflow
buffer, the receiver-side slot allocation is handled by
`PipeGraph::assignGatherSlotIndices` (in
`lib/Dialect/TTL/Transforms/PipeGraph.h`). It greedy-colors the pipes
that share a `(receiver, cbIndex)` pair so each pipe gets a distinct
slot index in the receiver dataflow buffer. `verifyReceiverDFBBlockCounts`
then requires `block_count >= max_slot_idx + 1` per receiver. This
makes overlapping multicast unrepresentable when more than 32 pipes
target the same receiver: the tt-metal per-Tensix CB cap is 32
(`NUM_CIRCULAR_BUFFERS` in
[`tt_metal/llrt/hal.hpp:409-411`][hal-num-cb], enforced in tt-lang
at `python/ttl/dataflow_buffer.py:66`), and the slot table requires
the count to equal the number of pipes.

### Multicast handshake protocol

The sender and receiver in each multicast pipe coordinate via a
per-PipeNet receiver counter, allocated by
`allocatePipeNetCountersForMulticast` as a kernel-local
`memref<1xi32>`. The lowering in `lib/Dialect/TTL/Transforms/PipeLowering.cpp`
emits the following sequence (some arguments elided for brevity):

```
                // one per (receiver, PipeNet); kernel-local memref<1xi32>
                int32_t recv_counter[1] = {0};

sender:    noc_async_write_multicast(data, recv_slot_addr, num_dests)
           noc_async_write_barrier()
           noc_semaphore_inc_multicast(recv_sem, +1, num_dests)
           noc_async_atomic_barrier()                          // order: data, then inc
receiver:  ++recv_counter[0]
           experimental::semaphore_wait_min(recv_sem, recv_counter[0])
           consume tile
```

`inc_multicast` adds to the remote semaphore, so `N` senders each
calling `inc(+1)` once per round produce a monotonically increasing
arrival count at every receiver. The receiver maintains a local
expectation in `recv_counter[0]` (one entry per `(receiver, PipeNet)`
pair) and waits with `experimental::semaphore_wait_min` until the
remote semaphore reaches at least that count. Each sender writes its
data to a distinct slot in the receiver's CB (slot assignment from
`PipeGraph::assignGatherSlotIndices`), so the data writes themselves
also do not collide.

Loopback (`src` in `dst` range) skips the increment on the source
core itself: the sender's `if_src` callback has already deposited the
tile in the local CB, so the loopback receiver advances its counter
expectation without waiting on the remote semaphore. The compiler
emits `noc_async_write_multicast_loopback_src` for the data write to
keep the multicast topology uniform across all receivers including
the source core.

## 3. Optimization opportunities

This section presents two kinds of design moves. §3.1 lists
architectural alternatives that would replace or substantially
reshape the rest of §3; §3.2 lists static per-Pipe rewrites at the
TTL-to-TTKernel boundary. The chosen approach in this document is
§3.2, but the alternatives in §3.1 are listed so the choice is
explicit and can be revisited.

### 3.1 Architectural alternatives

**High-level CCL primitives in the DSL.** Add intent-level ops to
the user-facing language: `ttl.AllReduce(grid=..., op="sum")`,
`ttl.Broadcast(src=root, grid=...)`, `ttl.AllGather(grid=...)`. The
compiler picks ring / tree / multicast per primitive based on grid
size and operand shape. The user states intent directly instead of
having the compiler recover it from primitive Pipe patterns.
Precedent: NCCL, MPI, JAX `pmap`, PyTorch `torch.distributed`, and
tt-metal's own [`experimental::ccl::all_gather_async`](https://github.com/tenstorrent/tt-metal/tree/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async)
expose collectives at API level. Trade-off: requires language-level
changes (new dialect ops and a new Python surface) on top of any
lowering work; §3.2 avoids touching the language but must guess
intent from low-level Pipe shapes.

**Autotuning instead of a static cost model.** §4.1's cost model
assumes the compiler can predict which lowering wins. An
alternative: at first invocation of a kernel on a target, compile
each candidate lowering, run them once each, pick the fastest, and
cache the choice keyed by `(kernel_signature, target)`. Subsequent
invocations use the cached pick. Precedent: cuBLAS, cuDNN, and
tt-metal's `MatmulMultiCoreReuseMultiCast*ProgramConfig` selection
all autotune;
[OpenTuner](https://github.com/jansel/opentuner) is an
open-source framework for building program autotuners over a
parametric search space. Trade-off: first-invocation cost is
`N × compile + N × device runtime` where `N` is the candidate
count; caches amortize it across repeat runs. Avoids the JSON
cost-model fitting in §4.3 entirely.

**Whole-kernel rewrites instead of per-Pipe rewrites.** §3.2
operates at Pipe granularity. An alternative is a kernel-template
library: recognize whole-kernel patterns (matmul + row-mcast +
col-mcast, gather + reduce, etc.) and re-emit the entire kernel
from a hand-tuned skeleton. Precedent: tt-metal already organizes
its performance kernels this way —
`matmul_multicore_reuse_mcast_2d_program_factory`,
`minimal_matmul_program_factory`,
`llama_all_gather_matmul_async_program_factory` are kernel
templates, not pipe-level rewrites. The user's Python is matched
against a known kernel shape; the emitted code is the corresponding
template instantiation. Trade-off: each new kernel pattern requires
a new skeleton (no automatic generalization), and template-match
failures fall back to the per-Pipe lowering, doubling the
maintenance surface. Captures the actual tt-metal workflow but
moves the policy choice from compiler-internal to library-author.

**Protocol-level slot reuse, replacing 3.2.6.** §3.2.6 (waves) works
around `block_count >= N` by sequentially decomposing the wide
PipeNet. A simpler alternative is a protocol change: receivers track
multi-iteration arrivals via the cumulative `wait_min` already in
place, and senders write to `(iteration_id mod
block_count)` slots — `block_count` becomes the in-flight pipeline
depth, not the sender count. Each sender knows its global
`iteration_id` from the static PipeGraph. Trade-off: replaces a
§3.2 entry with one protocol change. Constraint: requires the
receiver compute to consume tiles in slot order, which the
slot-per-pipe-gather contract already provides, so this is a
plausible refactor rather than a behavior change.

**Use existing MLIR collective dialects.** MLIR's
[`shard`](https://mlir.llvm.org/docs/Dialects/ShardOps/) dialect
(historically `mesh`) provides distributed-tensor sharding ops.
tt-mlir's TTCore dialect has its own collective abstractions.
tt-lang could lower to one of these instead of pattern-matching
Pipes. Trade-off: pulls in a larger external dialect surface;
depends on whether the existing dialects' semantics fit tt-lang's
intra-chip PipeNet model (the `shard` dialect targets distributed
memory across SPMD nodes, which is closer to tt-lang's
grid-of-cores than at first glance but not identical, and would
need careful evaluation before adoption).

**Why §3.2 (the static per-Pipe approach) is chosen here.** It
requires no language-level changes, composes with the existing
PipeNet protocol incrementally, and the rewrites can be deployed
individually behind `PassOptions` flags.

The high-level CCL alternative (first entry in §3.1) is the
strongest contender to replace §3.2. The reason §3.2 is chosen over
it: tt-lang's PipeNet primitive is intentionally lower-level than
NCCL/MPI-style CCL ops. PipeNets express custom dataflow patterns —
forwarding chains, ring rotations of arbitrary shape, the multi-stage
`test_pipe_conv` chain — that do not map onto a fixed set of
collective primitives. Adding `ttl.AllReduce` etc. would add a
sibling abstraction the user could opt into for canonical
collectives, but the per-Pipe rewrites still need to exist for
patterns the CCL ops do not cover. The two are not mutually
exclusive; if CCL ops are added later, the §3.2 rewrites can keep
handling non-canonical patterns and §4.8's per-Pipe `lowering`
override mitigates the pattern-recovery brittleness in the
meantime. The other §3.1 alternatives (autotuning, whole-kernel
rewrites, protocol-level slot reuse, MLIR collective dialects)
remain available; any of them could supersede all or part of §3.2
if implementation experience shows it is the better approach.

### 3.2 Static per-Pipe rewrites

Each of the following is a separate pattern rewrite at the
TTL-to-TTKernel boundary. The rewrite pass operates on
`ttl.create_pipe` ops and the receiver compute body for each
PipeNet; the existing per-Pipe-shape lowering is the fallback when
no rewrite matches.

| § | Rewrite | Pattern | Difficulty | Priority |
| --- | --- | --- | --- | --- |
| [3.2.1](#321-forwarding-chain-lowering-for-1-k-rectangular-broadcast) | Forwarding-chain lowering for 1→K broadcast | Multicast Pipe with no receiver-side reduction | Medium–High | High |
| [3.2.2](#322-ring-decomposition-lowering-for-n1-reductions) | Ring-decomposition for N→1 reductions | Convergent PipeNet with associative-commutative receiver compute | High | High |
| [3.2.3](#323-pre-push-before-forward-peephole) | Pre-push-before-forward peephole | Node both receiver and sender of same data | Low | High |
| [3.2.4](#324-data-movement-kernel-thread-assignment-from-the-dataflow-graph) | Data-movement kernel-thread assignment from dataflow graph | Multi-DM-kernel handshake deadlock under overlapping multicast | Medium | High |
| [3.2.5](#325-receiver-dfb-sharing-with-the-next-data-movement-kernel) | Receiver-DFB sharing with the next data-movement kernel | Pipe receiver consumed by exactly one downstream data-movement kernel | Medium | Medium |
| [3.2.6](#326-wave-decomposition-for-wide-overlapping-multicast) | Wave decomposition for wide overlapping multicast | PipeNet whose slot-per-pipe `block_count` exceeds 32 | Medium | Medium |

Difficulty estimates assume the rewrite is implemented as a TTL pass
emitting existing TTKernel ops. Priority is set against the patterns
the tuned tt-metal kernels actually use: `minimal_matmul`'s in0/in1
broadcast is the canonical 1→K case (3.2.1);
`reduce_scatter_minimal_async` and `llama_all_gather_matmul_async`
are the canonical N→1 reduction case (3.2.2). Whether these patterns
account for the bulk of intended tt-lang workloads has not been
measured.

Other PipeNet shapes already in tt-lang's test corpus are not
addressed here and fall through to the existing per-Pipe-shape
lowering:

- N→N all-to-all (`test/python/pipe/test_pipe_patterns.py::test_scatter_gather_1d`,
  `test/python/pipe/test_pipenet_overlap.py`) — overlapping multicast
  with a reduction at every receiver. These lower correctly under the
  existing per-Pipe lowering and only invoke 3.2.6 (waves) when
  `N > 32`.
- Ring forward without reduction (`test_pipe_patterns.py::test_forward_ring`,
  `test_pipe_patterns.py::test_row_rings_full`) — `N` parallel
  unicast pipes with `block_count = 2`, no convergence at any
  receiver, so none of the rewrites here match.
- Pipe chain (`test/python/pipe/test_pipe_conv.py`) — multi-stage
  unicast forwarding the user wrote by hand, which is exactly what
  3.2.1 would emit but with a different source shape (no `slice`
  dst).

Whether to add rewrites for those shapes is open and depends on
profile data on representative workloads. §3.2 covers the matmul +
CCL patterns documented in tt-metal but does not claim coverage of
every PipeNet shape tt-lang accepts.

3.2.3 and 3.2.4 are low-risk prerequisites that several later
rewrites depend on. 3.2.2 is the largest correctness win at scale
because it removes the per-receiver `O(N)` L1 footprint of
slot-per-pipe gather. 3.2.6 is the only rewrite proposed here for
`block_count > 32`. §7 gives a concrete implementation order that
respects the dependencies between rewrites.

#### 3.2.1 Forwarding-chain lowering for 1->K rectangular broadcast

**Pattern.** A single multicast `Pipe(src=p, dst=slice_along_axis)`
where exactly one axis of the destination is a `slice` and the other
is a constant coordinate (the receivers form a contiguous 1-D line),
and the receiver compute does not reduce across senders (it consumes
each block once and forwards to compute). The chain-vs-multicast
selection is K-dependent (see *Threshold selection* below).

**Destination-shape constraint.** The rewrite matches only the 1-D
case: `dst=(slice(a, b), const_y)` or `dst=(const_x, slice(a, b))`.
A true 2-D rectangular destination (`dst=(slice(...), slice(...))`,
both axes sliced) falls through to the existing
`noc_async_write_multicast` lowering. tt-metal's `minimal_matmul`
itself does not chain a 2-D rectangle directly — it organizes the
broadcast as two separate 1-D chains
([dm_in0_sender.cpp][dm-in0-chain] runs the in0 chain along rows,
[dm_in1_sender_out.cpp][dm-in1-out] runs the in1 chain along
columns), one PipeNet per axis. A 2-D-aware rewrite that splits one
`Pipe` with both-axes-sliced `dst` into a row chain plus per-row
column chains is feasible as a follow-up but is not proposed here;
in tt-lang's current PipeNet usage every multicast `Pipe` is already
1-D (every example in `test/python/pipe/` uses one `slice` axis plus
one constant axis).

**Rewrite.** Replace the one multicast write with `K-1` unicast hops
along the axis. The injector core (the source `p`) reads from DRAM (or
its dataflow buffer producer), pushes the block to its own dataflow
buffer, then issues `noc_async_write` to its successor in the line and
`noc_semaphore_set_remote` on the successor's "valid" semaphore. Each
downstream core, on receiving the valid signal, pushes the block to
its own dataflow buffer, then forwards to its successor by the same
primitive.

**Reference.** The pattern is implemented in
[`minimal_matmul/device/kernels/dm_in0_sender.cpp`][dm-in0-chain]
(in0 broadcast) and
[`dm_in1_sender_out.cpp`][dm-in1-out] (in1 broadcast). Each receiver
core waits on a semaphore from its predecessor, pushes the block to
its local compute DFB, then unicasts to its successor and signals
the next semaphore. The host-side wiring of `next_core` /
`prev_core` per row and column is in
[`minimal_matmul_program_factory.cpp`][factory-in0-next]; the
[same file][factory-chain-comment] uses the term "forwarding chain"
incidentally in a deadlock-condition comment, but the term is in
use in the tt-metal source.

**Why minimal_matmul prefers a chain.** The pre-push-before-forward
comment at
[`dm_in0_sender.cpp:294-296`][dm-in0-prepush] names the overlap as
performance-critical: each chain core pushes to its local compute
DFB before forwarding to its successor, which "frees sender to
start next read earlier." The chain pipelines the next-block DRAM
read with the current-block forward on every core in the chain;
multicast does not have this structure.

Beyond the pre-push overlap, the chain-vs-multicast tradeoff on
Wormhole / Blackhole is not independently characterized in this
document. tt-metal's
[multicast docstring][noc-mcast-docstring] describes the primitive
as a single transaction with a destination-range encoding (the
source issues one packet; routing and replication happen in the
NoC fabric), so a simple "source port serializes `K` destinations"
model does not hold. The microbenchmark suite in §4.3 is the way
to settle the threshold empirically. Receiver DFB stays at
`block_count = 2` either way.

**Threshold selection.** The chain-vs-multicast crossover depends on
`K`, the payload size, and the per-hop NoC latency. Multicast wins
when the multicast setup cost plus a single source-port-serialized
egress is less than `K-1` per-hop latencies plus `K-1` semaphore
handshakes. The crossover is not a constant: it varies with payload
size (large payloads amortize multicast setup) and with whether
3.2.3 (pre-push) is enabled. With 3.2.3, each chain hop's outbound
`noc_async_write` runs in parallel with the local `cb_push_back`,
so the chain's effective per-hop cost is the maximum of the two
rather than their sum; this lowers the chain's threshold and
expands the regime where the chain wins.

The threshold is empirical and depends on calibration data. The
choice of cost-model form, calibration strategy, and accuracy
budget is an open empirical-performance-modeling problem and is
deferred to the issue tracked in §4.3 rather than specified here.

Until the cost model is in place, the rewrite is conditional on a
`PassOptions` flag that defaults to off, so tt-lang's existing
tests (`test_mcast_matmul`, `test_scatter_full`, etc.) see no
behavior change. Reference points from tt-metal:
`minimal_matmul`'s data-movement kernels always use the forwarding
chain (no multicast fallback), while
`matmul_multicore_reuse_mcast_*` (1D and 2D variants under
`ttnn/cpp/ttnn/operations/matmul/device/factory/`) always use
`noc_async_write_multicast` for the in0/in1 broadcast. Both are
tuned for their workload, so neither extreme is a sound default for
arbitrary user PipeNets.

#### 3.2.2 Ring-decomposition lowering for N→1 reductions

**Pattern.** A PipeNet where `N` pipes target the same receiver and
the receiver compute is an associative-commutative reduction (the
receiver compute body matches the existing gather-then-sum shape: an
initial `acc_cb.store(t)` followed by `N-1` iterations of
`acc_cb.store(prev + t)` over `recv_cb.wait()`).

**Rewrite.** Replace the `N` gather pipes with a ring of `N-1` unicast
pipes carrying partial sums between neighbors. The reduction compute
moves into the per-step receiver block: each core adds its local
contribution to the incoming partial sum and forwards. After `N-1`
steps, the total lands at the designated reducer.

**Numerical reordering.** Float `+` is not associative, and bf16 is
more sensitive to reordering than fp32. Slot-per-pipe gather sums `N`
tiles in a fixed order set by the gather slot indices at the receiver:
`((t0 + t1) + t2) + ... + tN-1`. Ring decomposition sums in a different
order set by the physical ring topology — each core inserts its local
contribution at its position in the ring. The two schemes are
mathematically equal but produce different bit patterns in
finite-precision accumulation. For an `N`-way bf16 reduction the
result can drift by a relative epsilon comparable to one accumulation
step's precision.

This matches the LLVM stance on FP reductions: a vectorizer does not
reassociate FP additions by default; reassociation requires an
explicit opt-in. In MLIR this is the `fastmath<reassoc>` flag on
`arith.addf`, defined in
[`ArithBase.td:117`][mlir-arith-fastmath-bit] and applied via the
`fastmath` operand of
[`arith.addf` in `ArithOps.td:74-89`][mlir-arith-addf-op]. The loop
vectorizer's
[`useOrderedReductions`][llvm-lv-ordered] check at
[`LoopVectorize.cpp:948`][llvm-lv-ordered] walks the recurrence
descriptor and forces an ordered reduction unless the fastmath flag
is set; a [`--force-ordered-reductions`][llvm-lv-force-ordered] CLI
flag at [line 344][llvm-lv-force-ordered] lets the user disable
reassociation globally.

**Tt-lang opt-in mechanism.** The rewrite applies only when reassociation
is allowed, in this preference order:

- Per-PipeNet, at the construction site:
  `ttl.PipeNet([...], reduce_reassoc=True)` (a new keyword argument
  that lowers to an attribute on `ttl.create_pipe_net`).
- Per-operation, as a shorthand for "all PipeNets in this operation":
  `@ttl.operation(reduce_reassoc=True)`.
- Pass-level, as a global override matching LLVM's `-ffast-math`:
  `--ttl-pipe-rewrite-allow-reassoc` `PassOption`.

The default is off so existing PipeNets retain slot-per-pipe gather
order. Without this opt-in the rewrite does not apply and 3.2.6
(waves) handles wide reductions instead.

**Reference.** [`reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp`][rs-ring-reduction]
and the sibling [`ring_reduce_scatter_minimal_async_reader.cpp`][rs-ring-reader]
and [`_writer.cpp`][rs-ring-writer] implement this pattern over fabric
for cross-chip reductions; the intra-chip analogue uses the same
dataflow shape over NoC unicast.

**Why this is faster than slot-per-pipe gather.** Per-receiver inbound
traffic drops from `N * chunk` (slot-per-pipe gather, every sender
delivers a full tile) to `(N-1) * chunk` distributed across `N-1` ring
steps with the reduction folded into transit. Receiver dataflow buffer
stays at `block_count = 2` instead of `block_count >= N`. Every NoC
link carries one chunk per step instead of all senders saturating one
receiver's inbound port.

**Testing implication.** Pytests covering the rewritten lowering must
compare against a torch reference using PCC tolerance (e.g.
`assert_pcc(expected, result)`), not `torch.equal`. The tolerance
regime should match the existing bf16 reduction tests (e.g.
`test_pipe_patterns.py::test_gather`). Lit tests can `CHECK:` for the
emitted `noc_async_write` ring sequence as a structural property
without depending on numerical equivalence.

#### 3.2.3 Pre-push-before-forward peephole

**Pattern.** Any node that is both a receiver of one pipe and a sender
of another pipe carrying the same data (forwarding-chain lowering above
or any user-written chain).

**Rewrite.** Hoist the local `cb_push_back` above the outbound
`noc_async_write` so compute on the local node starts in parallel with
the forward, instead of waiting for the forward to complete.

**Reference.** The comment in
[`minimal_matmul/device/kernels/dm_in0_sender.cpp:294-296`][dm-in0-prepush]
documents this as performance-critical: "Critical to performance for
sender to push data to compute before mcasting / This frees sender to
start next read earlier." Pure pattern rewrite at the TTKernel level,
no PipeNet-level analysis required.

#### 3.2.4 Data-movement kernel-thread assignment from the dataflow graph

**Pattern.** Today the user manually splits work between two
`@ttl.datamovement()` kernel functions (conventionally `dm_read` and
`dm_write`) to avoid the handshake deadlock that occurs when one
function holds both `if_src` and `if_dst` work for an overlapping
multicast PipeNet. The deadlock is a single-thread in-order-issue
problem: a data-movement kernel function runs as one in-order RISC
sequence, so if `if_src` is issued first it blocks on the sender
ready handshake (`senderSem`) waiting for receivers to ack, but the
only place those acks are issued is the same core's `if_dst` block,
which sits later in the same sequence and never runs. Splitting the
two roles across two kernel functions gives them separate concurrent
issue contexts on the two data-movement RISCs.

tt-lang's current model pairs RISC and NOC: each
`@ttl.datamovement()` function is assigned to one RISC (`BRISC` /
`NCRISC`, surfaced in tt-metal as
`DataMovementProcessor::RISCV_0` / `RISCV_1`) and one NOC channel
(`NOC_0` / `NOC_1`) together, distinguished by the `ttl.noc_index`
attribute set by declaration order in `python/ttl/ttl_api.py:1344`.
tt-lang enforces exactly two data-movement kernel functions per
operation (`ttl_api.py:702`) because there are only two
data-movement RISCs available per core. The deadlock fix is the
RISC-thread split; the NOC channel distinction is conventional
pairing, not what the deadlock fix is about.

**Rewrite.** A pre-emission pass partitions data-movement work
across the two `@ttl.datamovement()` kernel functions based on
PipeNet roles (source vs destination) and known deadlock patterns.
The user writes a single intent — e.g. one `@ttl.datamovement()`
that contains all `if_src` and `if_dst` blocks — and the compiler
splits it into two functions assigned to the two RISCs.

**Constraint: only two RISCs per core.** When a kernel has more than
two logical data-movement roles (e.g. A-broadcast, B-broadcast, and
output-write — see `make_balanced_kernel` in
`test/python/pipe/test_mcast_matmul.py`), the rewrite must merge
roles to fit the two-thread limit. Today the user does this
manually (in `make_balanced_kernel`, A-broadcast goes on `dm_read`,
B-broadcast plus output-write share `dm_write`). The rewrite would
apply the same merging using role-pair compatibility rules: roles
that do not deadlock against each other (one source plus one
destination of the same PipeNet, or two sources of different
PipeNets, etc.) may share a kernel function. Roles whose pairing
would re-introduce the deadlock get separate functions.

**Reference.** [`minimal_matmul_program_factory.cpp:229-247`][factory-noc-policy]
shows the tt-metal-side convention that pairs RISC and NOC:
small-input DM on `RISCV_1` / `NOC_1`, large-input DM on `RISCV_0`
/ `NOC_0`, with grid transpose to keep the assignment symmetric for
non-square outputs. The deadlock condition the rewrite eliminates
is documented in the comment block of `scatter_gather_1d_kernel` in
`test/python/pipe/test_pipe_patterns.py`.

#### 3.2.5 Receiver-DFB sharing with the next data-movement kernel

**Pattern.** A Pipe whose receiver DFB is consumed by exactly one
downstream data-movement kernel, which immediately forwards the
data to a compute kernel via a separate DFB. The canonical case: a
CCL receiver staging DFB feeding a matmul operand-reader DFB.

**Rewrite.** Eliminate the staging DFB. The Pipe's
`noc_async_write_multicast` lands directly in the L1 buffer that the
operand-reader DFB occupies; the operand-reader waits on a semaphore
(signaled by the multicast sender) instead of doing a
`cb_wait_front` on the staging DFB. Saves one CB index (one of the
32 per-Tensix tt-metal CB slots that DFBs lower to), one
`cb_push_back` / `cb_pop_front` pair, and the implicit L1 region
reservation for the staging DFB.

**Reference.** [`llama_all_gather_matmul_async/.../reader_bmm_tile_layout_in0_ring_all_gather.cpp`][llama-reader]
is the matmul A-operand reader; the gather sender writes directly
into the same L1 region the reader consumes from, so no staging
DFB exists between gather and matmul-A. The companion compute
kernel is
[`bmm_large_block_zm_fused_bias_activation_gathered.cpp`][llama-compute].

**Why this is faster.** Frees one of the 32 per-Tensix CB indices
that DFBs lower to and removes one push/pop pair per delivered
tile. For kernels approaching the index ceiling (e.g.
`make_balanced_relu_kernel` in `test_mcast_matmul.py` already uses
4-5 DFBs), the index headroom is the limiting factor.

**Constraints.** Requires the Pipe receiver and the operand-reader
to live on the same data-movement kernel function (so they can share
an L1 buffer assigned at allocation time). Cross-thread sharing
needs a separate analysis pass.

#### 3.2.6 Wave decomposition for wide overlapping multicast

**Pattern.** A single PipeNet whose slot-per-pipe `block_count`
exceeds the L1 budget or the tt-metal per-Tensix CB cap of 32.

**Rewrite.** The pass splits the PipeNet into `K` narrower PipeNets
executed sequentially, each with `block_count = ceil(N/K)` where `N`
is the original number of pipes targeting the same receiver. The user writes one wide
PipeNet; the compiler emits a sequential loop over the `K` waves.
Receiver compute runs once per wave with that wave's `block_count`,
and the accumulator dataflow buffer rolls across waves.

**Why this matters.** Decouples the PipeNet width `N` from the
`block_count` constraint, so the dataflow buffer depth remains
within the tt-metal CB cap regardless of `N`. Source IR is one
PipeNet; the rewrite is internal to lowering.

## 4. Implementation details

### 4.1 Cost models

Several of the rewrites in §3.2 are conditional: 3.2.1 selects chain
versus multicast based on `K` and payload size, and 3.2.2 selects
ring versus slot-per-pipe gather based on `N` and the
`reduce_reassoc` opt-in. 3.2.6 has no condition — it applies whenever
the slot-per-pipe `block_count` would exceed the dataflow buffer
depth cap.
Each conditional rewrite needs a cost estimate of the alternatives
that the compiler can compute statically (or with a small set of
fitted parameters).

The proposal is a `TTLPipeCostModel` interface, populated per target
(Wormhole, Blackhole), that provides the primitive costs each rewrite
queries. At minimum the interface exposes:

- `nocHopLatency(payloadBytes)`: cycles per single NoC hop as a
  function of payload size, fitted from the 4.3 microbenchmark.
- `multicastSetupCost(numDests, payloadBytes)`: fixed setup plus
  per-destination serialization at the source NoC port; depends on
  the destination count and payload size.
- `semaphoreHandshakeCost()`: round-trip cost of the ready/valid
  handshake used by the Pipe protocol (sender-side
  `noc_semaphore_set_remote` plus receiver-side ack).
- `l1BandwidthPerCycle()`: per-Tensix L1 bandwidth in bytes per
  cycle, used to bound the receiver-side staging cost subtracted by
  3.2.5.
- `dataflowBufferDepthCap()`: 32 on Wormhole and Blackhole, the
  static maximum `block_count` enforced by `python/ttl/circular_buffer.py`.
  3.2.6 applies only when the slot-per-pipe `block_count` would
  exceed this value.

Each rewrite then implements `costOfRewritten(pipeNet, ctx)` and
`costOfFallback(pipeNet, ctx)` returning a comparable scalar; the
rewrite applies when the difference exceeds a target-specific threshold.
This isolates target-specific calibration from the rewrite logic.

### 4.2 Upstream precedents

Three patterns from upstream LLVM/MLIR are directly relevant. Each
one solves a structurally similar problem (pick a code-generation
strategy from a parametric family using a target-specific cost model)
and provides an interface shape we can adapt rather than invent.

<a id="tti"></a>
**`llvm::TargetTransformInfo` (TTI).** TTI is LLVM's per-target hook
interface for backend-cost queries. It is consumed by the loop
vectorizer (vectorization factor selection), the SLP vectorizer
(superword-level packing decisions), and the inliner (threshold
adjustments via `getInliningThresholdMultiplier`,
`getInlinerVectorBonusPercent`). The class declaration is at
[`llvm/include/llvm/Analysis/TargetTransformInfo.h:271`][llvm-tti-class],
and the central per-instruction cost query is
[`getInstructionCost`][llvm-tti-getinst] at line 483. Targets supply
their own implementation of the abstract concept; transformation
passes consume the interface without referencing a specific backend.
The same shape — abstract per-target interface, supplied by a target
descriptor, consumed by a target-agnostic rewrite — is what
`TTLPipeCostModel` should follow.

**`LoopVectorizationCostModel`.** The loop vectorizer uses [TTI](#tti) to
compute the expected cost of vectorizing at each candidate `VF` and
picks the `VF` that minimizes it. The cost-model class is at
[`llvm/lib/Transforms/Vectorize/LoopVectorize.cpp:867`][llvm-lv-class].
The per-`VF` cost summation is in
[`LoopVectorizationCostModel::expectedCost(ElementCount VF)`][llvm-lv-expectedcost]
at line 4996.
[`computeMaxVF`][llvm-lv-computemaxvf] at line 3450 returns the upper
bound on `VF` (above which vectorization is unsafe);
[`LoopVectorizationPlanner::computeBestVF`][llvm-lv-computebestvf] at
line 6897 enumerates candidate factors below that bound and picks the
one with the lowest `expectedCost`. The parametric-family analogy is
direct: the loop vectorizer picks `VF` from `{1, 2, 4, 8, ...}`; the
chain rewrite picks chain-versus-multicast based on `K`; the wave
decomposition picks the wave count based on `N` and the dataflow
buffer depth cap.

**Inliner cost analyzer.** The inliner accumulates a per-instruction
cost as it walks the callee and compares against a threshold to
decide whether to inline. The walker is `CallAnalyzer` at
[`llvm/lib/Analysis/InlineCost.cpp:248`][llvm-inline-callanalyzer].
The relevant analogy is per-rewrite: walk the PipeNet body once,
accumulate the cost of the alternative lowering, compare against the
fallback. Same control flow, different cost atoms.

**Affine loop fusion profitability.** MLIR's affine loop fusion uses
a cost model to decide whether to fuse two nests. The decision
function is
[`isFusionProfitable`][mlir-affine-fusion] in
`mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp:500`. It computes
sliced and unsliced loop-nest costs, compares them, and applies a
tolerance threshold (`computeToleranceThreshold` argument). For
tt-lang the same pattern applies to 3.2.5 (receiver-DFB sharing):
fold the receiver staging buffer into the next data-movement
kernel's DFB only when the post-fold cost (one fewer DFB, one
fewer push/pop pair) beats the unfolded cost by some
target-specific tolerance.

### 4.3 Cost-model fitting and calibration (deferred to a separate issue)

How to obtain the primitive cost values that §4.1 declares — what
microbenchmarks to run, what functional form to fit to the
measurements, how to handle measurement noise, how to pick a
sample budget that trades accuracy against calibration cost — is
itself an empirical-performance-modeling problem. This document
does not specify a fitting approach; that decision should be made
by someone with empirical-performance-modeling expertise against
tt-lang's actual calibration data, and tracked as a separate work
item.

Relevant literature for that work item:

- Ritter, Naumann, Calotoiu, Rinke, Reimann, Hoefler, Wolf,
  *Cost-Effective Empirical Performance Modeling*, IEEE TPDS Vol.
  37 No. 2, Feb. 2026 — recent treatment of measurement-point
  selection under a budget. Uses Performance Model Normal Form
  (PMNF, sums of products of polynomial and logarithmic terms in
  the parameters: `f(x₁..x_i) = Σ_k c_k · Π_l x_l^α · log₂^β(x_l)`)
  as the hypothesis space, fits coefficients by linear regression,
  and picks the best hypothesis by cross-validation SMAPE. Proposes
  a Gaussian-process-regression-driven strategy for choosing which
  measurement points to acquire next, achieving 77.8% accuracy at
  10% of the naïve full-grid measurement cost.
- [Extra-P](https://github.com/extra-p/extrap) (Calotoiu et al.)
  is the underlying open-source tool implementing the PMNF
  modeling framework that Ritter et al. extend.

Until the cost model is in place, each conditional rewrite (3.2.1,
3.2.2) is conditional on a `PassOptions` flag that defaults to off,
so tt-lang's behavior does not depend on cost-model availability.

Existing tt-metal microbenchmarks usable as data-collection
starting points, regardless of which fitting approach is chosen:

- [`tests/tt_metal/microbenchmarks/noc/test_noc_unicast_vs_multicast_to_single_core_latency.py`][tt-metal-uvm-py]
  and its [C++ driver][tt-metal-uvm-cpp] map directly onto 3.2.1's
  chain-vs-multicast question; both use the device profiler
  (`TT_METAL_DEVICE_PROFILER=1`) to capture per-NoC-zone cycle
  counts.
- [`tests/tt_metal/tt_metal/perf_microbenchmark/2_noc_adjacent/test_noc_adjacent.cpp`][tt-metal-noc-adj]
  is an adjacent-core NoC sweep.
- [`tests/tt_metal/tt_metal/perf_microbenchmark/2_noc_rtor/test_noc_rtor.cpp`][tt-metal-noc-rtor]
  is a random-source-to-random-destination NoC sweep.
- The [tt-benchmarking][tt-benchmarking-repo] repository hosts
  op-level perf harnesses; if a chain-vs-multicast or a
  slot-per-pipe sweep already lives there, the tt-lang
  calibration pipeline should reuse the existing driver rather
  than duplicate it.

### 4.4 Pipeline placement of the rewrite pass

Each rewrite is a separate pattern matcher. They can be bundled into a
single TTL pass that runs after the active-set guard pass and before
`convert-ttl-to-ttkernel` (the conversion consumes
`ttl.create_pipe`, so the rewrites must run first). §8 covers the
active-set re-run that the chain (3.2.1) and the receiver-DFB
sharing (3.2.5) require.

The pass must be deterministic on identical inputs. tt-metal's JIT
cache key is content-addressed by the emitted kernel C++ source,
compile-time args, and `#define`s
([`Kernel::compute_hash`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tt_metal/impl/kernels/kernel.cpp#L374-L399)),
so different cost-model outputs naturally produce different emitted
source and land in distinct cache entries — the cache doesn't need
to know about the cost model. If the rewrite output ever depends on
non-deterministic input (hash randomization in PipeGraph
traversal, parallel-pass ordering, etc.) the same Python source
would produce different emitted C++ across compiles and miss the
cache every time.

### 4.5 L1 allocation for non-DFB regions

Some rewrites need L1 storage that is not a DFB. 3.2.5
(receiver-DFB sharing) reuses an existing DFB so it needs no new
allocation; future rewrites that spill an intermediate accumulator
would need sized L1 regions sharing liveness with the surrounding
compute.

The mechanism tt-lang uses today is host-side `CreateSemaphore` /
`Buffer` with the address passed as a kernel runtime argument.
The existing PipeNet protocol already uses this for the
`senderSem` / `recvSem` pair (lowered in
`lib/Dialect/TTL/Transforms/PipeLowering.cpp`). The host-side
`CreateSemaphore(program, core_grid, init_value)` returns a 4-byte
L1 word usable from any core in `core_grid`; the address is wired
into the kernel as a compile-time arg, fetched at runtime via the
TTKernel op
[`ttkernel.get_semaphore`](https://github.com/tenstorrent/tt-mlir/blob/main/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td)
(declared at `TTKernelOps.td:3500`), and operated on with
`noc_semaphore_inc`, `noc_semaphore_set`, etc. For arbitrary L1
regions larger than 4 bytes, host-side `Buffer::create_l1_sharded`
returns an L1 base address that can be passed similarly. This
mechanism is sufficient for any future rewrite that needs a small
fixed-size L1 region (e.g., a per-PipeNet shared counter) without
any new dialect surface; allocate a fresh semaphore alongside the
existing `senderSem` / `recvSem` and operate on it with
`noc_semaphore_inc` etc.

For sized L1 regions whose lifetime overlaps surrounding compute,
the host-side approach above is insufficient and a proper
liveness-aware L1 allocator is needed. tt-lang will not adopt
D2M's allocator (`memref.alloc` with
[`ttcore::MemorySpace::DeviceL1`][ttcore-l1-enum] passed through
the [`D2MAllocate`][d2m-allocate] pass — `addScratchToGeneric` in
[`InsertScratchBuffers.cpp:130-177`][d2m-scratch] is the canonical
example) — the project's stance is that the D2M dialect dependency
is cut. If a rewrite ever needs allocator-managed L1, tt-lang
should grow its own TTL-side allocator targeting the same
underlying tt-metal `Buffer` mechanism but driven from PipeGraph
liveness rather than D2M's.

Distinction from the existing per-PipeNet receiver counter: that
counter is a `memref<1xi32>` with no memory-space attribute. The standard
MemRefToEmitC patterns lower it to a stack-allocated `int32_t
counter[1]` inside the kernel function — not L1-allocated. This is
sufficient for that counter (each kernel invocation needs a fresh
counter, no cross-core sharing). It is not sufficient for any
counter that must be visible to other cores; that case requires
the host-side semaphore mechanism above.

Recommendation by rewrite:

| Rewrite | L1 need | Mechanism |
|---|---|---|
| 3.2.5 receiver-DFB sharing | Reuses an existing DFB | None; no new allocation |
| Future cross-core counter | One 4-byte semaphore per PipeNet | Host-side `CreateSemaphore` |
| Future intermediate accumulator | Sized L1 region with liveness | TTL-side allocator (does not exist today; D2M's allocator is not an option) |

The host-side mechanism covers every L1 need the §3.2 rewrites
have today.
Path B becomes load-bearing only when a rewrite needs
allocator-managed L1, which none of the rewrites in §3.2 require.

[ttcore-l1-enum]: https://github.com/tenstorrent/tt-mlir/blob/main/include/ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td#L64
[d2m-allocate]: https://github.com/tenstorrent/tt-mlir/blob/main/lib/Dialect/D2M/Transforms/Allocate.cpp
[d2m-scratch]: https://github.com/tenstorrent/tt-mlir/blob/main/lib/Dialect/D2M/Transforms/InsertScratchBuffers.cpp#L130-L177

### 4.6 Multi-PipeNet composition

Realistic kernels have several PipeNets at once
(`test_minimal_matmul_pipes`: 2; `test_pipenet_collectives`: 7).
Composition rules per rewrite:

- **3.2.1 (chain), 3.2.2 (ring)** match per-PipeNet. Two chains in
  the same kernel share NoC bandwidth, so 3.2.1's per-Pipe cost
  estimate overstates the chain's win when both saturate the same
  NoC — the cost model in §4.1 needs an aggregate-per-NoC term.
  Two rings sharing a receiver-side accumulator DFB must each get a
  distinct accumulator DFB.
- **3.2.3 (pre-push)** is a TTKernel-level peephole, no
  interaction with PipeNet count.
- **3.2.4 (DM-thread assignment)** is kernel-wide. The two-thread
  constraint is on the kernel, so role-pair compatibility must
  span every PipeNet's roles together.
- **3.2.5 (receiver-DFB sharing)** is per-Pipe, but a DFB can be
  the merge target of at most one Pipe. When two Pipes from
  different PipeNets feed the same compute kernel, only one merges;
  the rewrite picks deterministically (e.g. by Pipe
  source-coordinate order).
- **3.2.6 (waves)** is per-receiver-DFB. Sender counts add across
  PipeNets converging on the same DFB, so two PipeNets
  contributing 20 pipes each to the same DFB trigger 3.2.6 even
  though neither alone exceeds the cap.

### 4.7 Compile-time impact

Each rewrite walks PipeGraph, matches per-Pipe patterns, and may
mutate IR. Compile-time delta against a representative kernel
should be measured before each rewrite is enabled by default;
rewrites with measurable impact stay behind a `PassOptions` flag.

### 4.8 Per-Pipe override and rewrite diagnostics

`ttl.PipeNet`'s `Pipe` constructor accepts an optional `lowering`
keyword: `"auto"` (default; cost-model decides), `"multicast"`,
`"chain"`, `"ring"`, `"gather"`. It lowers to a string attribute on
`ttl.create_pipe`. When set to anything other than `"auto"`, the
rewrite pass emits the named primitive if it is compatible with
the Pipe's source / destination shape and errors at compile time
otherwise (e.g. `"chain"` on a 2-D-rect destination). §3.2.2's
`reduce_reassoc=True` is a special case. clang precedent: `#pragma
clang loop vectorize(disable)` / `vectorize_width(N)`, parsed by
the [`LoopHintAttr` handler in `clang/lib/Sema/SemaStmtAttr.cpp:74`][clang-loop-hint].

A `--ttl-pipe-rewrite-remarks` `PassOption` emits one diagnostic
per `ttl.create_pipe` op listing the attempted rewrite, the
applied/skipped result, the reason (cost-model term values,
threshold comparison, user override), and the resulting primitive.
LLVM precedent: `-Rpass=loop-vectorize` via
[`OptimizationRemarkEmitter`][llvm-ore]
(`llvm/include/llvm/Analysis/OptimizationRemarkEmitter.h:33`).

## 5. Pattern detection inputs

All of the patterns above can be detected from data already present during
the TTL-to-TTKernel lowering:

- `PipeGraph` records all PipeNets, their pipes, and the
  `(receiver, cbIndex)` mapping that identifies which pipes target the
  same receiver dataflow buffer (`lib/Dialect/TTL/Transforms/PipeGraph.h`).
- `ttl.if_dst` callback bodies expose the receiver compute
  (whether it stores a single tile or accumulates).
- Pipe shape attributes (`I64Attr` source and destination ranges)
  expose contiguity along an axis.
- The kernel-thread `func.func` attributes and the active-set guard
  pass (see `PipeNets.md`) expose role assignment per node.

No new dialect or attribute is required to detect any of the patterns.
The rewrite pass (see 4.4) attempts them on each `ttl.create_pipe` op
before `convert-ttl-to-ttkernel`, with the existing per-shape
lowering as the fallback when no rewrite matches.

The pattern matching assumes Pipe coordinates are
compile-time-known integers, which is what `ttl.create_pipe`'s
`I64Attr` operands carry today. tt-lang's frontend resolves
`grid="full"` and `ttl.grid_size(dims=2)` to concrete integers in
`_resolve_grid` (`python/ttl/ttl_api.py:436-444`) before MLIR is
emitted, so a kernel using `for x in range(grid_x)` to construct
pipes materialises one Pipe per `x` with constant attributes. If
tt-lang ever supports compile-once kernels parameterised over grid
size at run time, Pipe shape would have to become symbolic (SSA
values in place of `I64Attr`) and the rewrite pass would either
need to run after a grid-resolution pass or detect patterns via
symbolic analysis. The §3.2 rewrites do not depend on which of
those choices is made; they require only that Pipe coordinates be
concrete by the time the rewrite runs.

## 6. Tests
Existing lit tests affected by §3.2 rewrites:

- `test/ttlang/Dialect/TTL/Transforms/convert_pipe_ops.mlir` —
  3.2.1, 3.2.5, and 3.2.6 change the emitted IR.
- `test/ttlang/Dialect/TTL/Transforms/convert_pipe_ops_overlap.mlir` —
  3.2.6 changes the wave-decomposed cases.
- `insert_pipenet_active_guards*.mlir` — unaffected.

3.2.2 reorders bf16 accumulation; pytests covering it require PCC
tolerance, not bit-exact comparison (per §3.2.2's testing
implication).

## 7. Possible implementation order

The rewrites are independent but ordering reduces risk:

1. 3.2.3 pre-push-before-forward peephole. Pure TTKernel-level
   rewrite. No PipeNet semantics change.
2. 3.2.4 data-movement kernel-thread assignment from the dataflow
   graph. Removes a class of user-visible deadlocks: when `if_src` and
   `if_dst` for the same PipeNet share a NOC thread, every core blocks
   on its own sender handshake before any `if_dst` block can signal
   ready. The rewrite assigns the two roles to distinct NOC threads
   automatically. Independent of the other rewrites.
3. 3.2.1 forwarding-chain lowering for 1→K rectangular broadcast.
   Depends on (1) for the pre-push idiom.
4. 3.2.5 receiver-DFB sharing with the next data-movement kernel.
   Depends on stable PipeNet receiver shape; independent of the
   chain rewrite.
5. 3.2.2 ring-decomposition lowering for N→1 reductions. Largest
   dataflow rewrite; introduces unicast pipes between every
   neighboring pair and moves the reduction compute into the
   per-step receiver block.
6. 3.2.6 wave decomposition for wide overlapping multicast. Bounded
   scope (one PipeNet -> many PipeNets in sequence); does not
   depend on the above.

## 8. Interaction with the active-set guard pass

The active-set guard pass (`PipeNets.md`) computes the union of every
pipe's source and destination range and wraps each kernel function
body in an `scf.if` predicate. The rewrite pass placed by 4.4 runs
after the first guard-pass run. Because the chain rewrite (3.2.1)
adds pipes and the receiver-DFB-sharing rewrite (3.2.5) removes
staging buffers (and may eliminate the corresponding `if_dst`
callback), the active set from the first run is stale by the time
`convert-ttl-to-ttkernel` runs.
The fix is to re-run the guard pass after the rewrite pass; the
guard pass is idempotent via the `ttl.pipenet_active_guard` marker
attribute (`PipeNets.md`), so the second run re-derives the
predicate from the rewritten pipes without disturbing already-guarded
functions.

## 9. Non-goals

* Cross-chip lowering. The rewrites here are intra-chip. Cross-chip
  lowering is captured separately in `PipeNets.md` Future work.
* New dialect ops. All rewrites emit existing TTKernel ops. The
  forwarding-chain lowering uses `noc_async_write` and
  `noc_semaphore_set_remote`, both already present.
* User-visible API changes. The PipeNet language stays as it is; the
  rewrites are internal to the lowering.

## 10. Open questions

* Should 3.2.6 (waves) become unconditional once it lands, replacing
  the slot-per-pipe protocol entirely as the lowering for any
  overlapping multicast (with `block_count` capped by an in-flight
  window rather than the pipe count)? Doing so would replace 3.2.6
  with the protocol-level slot-reuse alternative listed in §3.1 and
  remove the slot-table machinery from `PipeGraph` altogether.

## 11. References

LLVM / MLIR upstream (at `llvm-project` SHA `705cdc3a9d0adb4c0667aa840a1f23165eca297b`):

- [`TargetTransformInfo` class declaration](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/include/llvm/Analysis/TargetTransformInfo.h#L271)
- [`TargetTransformInfo::getInstructionCost`](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/include/llvm/Analysis/TargetTransformInfo.h#L483)
- [`LoopVectorizationCostModel` class](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L867)
- [`LoopVectorizationCostModel::expectedCost`](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L4996)
- [`LoopVectorizationCostModel::computeMaxVF`](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L3450)
- [`LoopVectorizationPlanner::computeBestVF`](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L6897)
- [`CallAnalyzer` (inliner cost)](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Analysis/InlineCost.cpp#L248)
- [`isFusionProfitable` (MLIR affine loop fusion)](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp#L500)
- [`OptimizationRemarkEmitter` (LLVM `-Rpass=` machinery)](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/include/llvm/Analysis/OptimizationRemarkEmitter.h#L33)
- [`LoopHintAttr` handler (clang `#pragma clang loop` parser)](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/clang/lib/Sema/SemaStmtAttr.cpp#L74)

Empirical performance modeling:

- Ritter, Naumann, Calotoiu, Rinke, Reimann, Hoefler, Wolf,
  "Cost-Effective Empirical Performance Modeling," *IEEE
  Transactions on Parallel and Distributed Systems*, Vol. 37, No.
  2, February 2026
  ([SPCL preprint](https://spcl.inf.ethz.ch/Publications/.pdf/2026_ritter.pdf))
- [Extra-P](https://github.com/extra-p/extrap) — open-source PMNF
  modeling tool

Autotuning frameworks:

- [OpenTuner](https://github.com/jansel/opentuner) — open-source
  framework for building program autotuners over a parametric
  search space
- [`arith.fastmath<reassoc>` flag (MLIR `ArithBase.td:117`)](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/mlir/include/mlir/Dialect/Arith/IR/ArithBase.td#L117)
- [`arith.addf` `fastmath` operand (MLIR `ArithOps.td:74-89`)](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td#L74-L89)
- [`useOrderedReductions` (LLVM LoopVectorize ordered FP reduction check)](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L948)
- [`--force-ordered-reductions` CLI flag](https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L344)

tt-metal (at SHA `c296ef469fe6aab65ab0d359e164b14b62d92bfc`):

- [`minimal_matmul/device/kernels/dm_in0_sender.cpp:287-315`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp#L287-L315) — forwarding-chain pattern (in0 sender)
- [`minimal_matmul/device/kernels/dm_in1_sender_out.cpp`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp) — in1 sibling
- [`minimal_matmul_program_factory.cpp:719-722`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp#L719-L722) — `in0_next_core_physical` / `in0_prev_core_physical` runtime args
- [`minimal_matmul_program_factory.cpp:770-773`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp#L770-L773) — same for in1
- [`minimal_matmul_program_factory.cpp:658`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp#L658) — "forwarding chain" comment
- [`minimal_matmul/device/kernels/dm_in0_sender.cpp:294-296`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp#L294-L296) — pre-push-before-forward comment
- [`minimal_matmul_program_factory.cpp:229-247`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp#L229-L247) — NOC0/NOC1 / RISCV0/RISCV1 policy
- [`reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp) — ring reduce-scatter compute
- [`ring_reduce_scatter_minimal_async_reader.cpp`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_reader.cpp) — ring reduce-scatter reader
- [`ring_reduce_scatter_minimal_async_writer.cpp`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_writer.cpp) — ring reduce-scatter writer
- [`llama_all_gather_matmul_async/.../reader_bmm_tile_layout_in0_ring_all_gather.cpp`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in0_ring_all_gather.cpp) — matmul A reader fused with all-gather receive
- [`bmm_large_block_zm_fused_bias_activation_gathered.cpp`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp) — fused matmul+bias compute kernel
- [`tests/tt_metal/microbenchmarks/noc/test_noc_unicast_vs_multicast_to_single_core_latency.py`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tests/tt_metal/microbenchmarks/noc/test_noc_unicast_vs_multicast_to_single_core_latency.py) — Python wrapper
- [`tests/tt_metal/tt_metal/perf_microbenchmark/noc/test_noc_unicast_vs_multicast_to_single_core_latency.cpp`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tests/tt_metal/tt_metal/perf_microbenchmark/noc/test_noc_unicast_vs_multicast_to_single_core_latency.cpp) — C++ driver
- [`tests/tt_metal/tt_metal/perf_microbenchmark/2_noc_adjacent/test_noc_adjacent.cpp`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tests/tt_metal/tt_metal/perf_microbenchmark/2_noc_adjacent/test_noc_adjacent.cpp) — adjacent-core NoC sweep
- [`tests/tt_metal/tt_metal/perf_microbenchmark/2_noc_rtor/test_noc_rtor.cpp`](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tests/tt_metal/tt_metal/perf_microbenchmark/2_noc_rtor/test_noc_rtor.cpp) — random-source-to-random-destination NoC sweep
- [tt-benchmarking repository](https://github.com/tenstorrent/tt-benchmarking) — op-level perf harnesses
- [`Kernel::compute_hash` (tt-metal JIT cache key)](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tt_metal/impl/kernels/kernel.cpp#L374-L399) — hashes emitted source, compile-time args, defines, and config
- [`NUM_CIRCULAR_BUFFERS = 32` (tt-metal per-Tensix CB cap)](https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tt_metal/llrt/hal.hpp#L409-L411) — the limit `block_count` is bounded by

<!--
The reference labels below back the in-text [text][label] links and
should be left in place; markdown renders them as invisible link
definitions.
-->

[llvm-tti-class]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/include/llvm/Analysis/TargetTransformInfo.h#L271
[llvm-tti-getinst]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/include/llvm/Analysis/TargetTransformInfo.h#L483
[llvm-lv-class]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L867
[llvm-lv-expectedcost]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L4996
[llvm-lv-computemaxvf]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L3450
[llvm-lv-computebestvf]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L6897
[llvm-inline-callanalyzer]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Analysis/InlineCost.cpp#L248
[mlir-affine-fusion]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp#L500
[llvm-ore]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/include/llvm/Analysis/OptimizationRemarkEmitter.h#L33
[clang-loop-hint]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/clang/lib/Sema/SemaStmtAttr.cpp#L74
[mlir-arith-fastmath-bit]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/mlir/include/mlir/Dialect/Arith/IR/ArithBase.td#L117
[mlir-arith-addf-op]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td#L74-L89
[llvm-lv-ordered]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L948
[llvm-lv-force-ordered]: https://github.com/llvm/llvm-project/blob/705cdc3a9d0adb4c0667aa840a1f23165eca297b/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L344
[dm-in0-chain]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp#L287-L315
[noc-mcast-docstring]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tt_metal/hw/inc/api/dataflow/dataflow_api.h#L885-L924
[hal-num-cb]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tt_metal/llrt/hal.hpp#L409-L411
[dm-in1-out]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp
[factory-in0-next]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp#L719-L722
[factory-in1-next]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp#L770-L773
[factory-chain-comment]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp#L658
[dm-in0-prepush]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp#L294-L296
[factory-noc-policy]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp#L229-L247
[rs-ring-reduction]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp
[rs-ring-reader]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_reader.cpp
[rs-ring-writer]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_writer.cpp
[llama-reader]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in0_ring_all_gather.cpp
[llama-compute]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp
[tt-metal-uvm-py]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tests/tt_metal/microbenchmarks/noc/test_noc_unicast_vs_multicast_to_single_core_latency.py
[tt-metal-uvm-cpp]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tests/tt_metal/tt_metal/perf_microbenchmark/noc/test_noc_unicast_vs_multicast_to_single_core_latency.cpp
[tt-metal-noc-adj]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tests/tt_metal/tt_metal/perf_microbenchmark/2_noc_adjacent/test_noc_adjacent.cpp
[tt-metal-noc-rtor]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tests/tt_metal/tt_metal/perf_microbenchmark/2_noc_rtor/test_noc_rtor.cpp
[tt-benchmarking-repo]: https://github.com/tenstorrent/tt-benchmarking
