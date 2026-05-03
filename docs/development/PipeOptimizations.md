# Pipe Optimizations

This document specifies proposed lowering rewrites for `ttl.create_pipe`
that select the NoC primitive based on the PipeNet pattern. The current
`convert-ttl-to-ttkernel` policy is one fixed primitive per Pipe shape
(`noc_async_write_multicast` for `slice` destinations,
`noc_async_write` for point destinations, slot-per-pipe receiver
dataflow buffers for converging pipes). This policy is independent of
the receiver compute and the destination geometry, and emits a
sub-optimal primitive for several common patterns. The
performance-tuned tt-metal kernels (`minimal_matmul`,
`llama_all_gather_matmul_async`, `reduce_scatter_minimal_async`)
select different primitives for these patterns by hand. Closing the
gap is a lowering-policy change, not a dialect change.

Goals:

1. Make the lowering policy pattern-driven, not Pipe-shape-driven.
2. Emit the primitives the tuned tt-metal kernels use for each pattern,
   without changing the user-facing PipeNet language.
3. Decouple the receiver dataflow buffer depth from the converging
   pipe count, so an `N`-way overlapping multicast no longer requires
   `block_count >= N`. The current coupling is enforced by
   `PipeGraph::assignGatherSlotIndices` and capped at 32 by the
   tt-metal dataflow buffer depth limit.

## Summary

| # | Rewrite | Pattern | Difficulty | Priority |
| --- | --- | --- | --- | --- |
| 1 | Pre-push-before-forward peephole | Node both receiver and sender of same data | Low | High |
| 2 | NOC0/NOC1 assignment from dataflow graph | All multi-NOC kernel emission | Medium | High |
| 3 | Forwarding-chain lowering for 1→K broadcast | Multicast Pipe with no receiver-side reduction | Medium–High | High |
| 4 | Receiver-compute fusion | Pipe receiver consumed by exactly one compute kernel | Medium | Medium |
| 5 | Ring-decomposition for N→1 reductions | Convergent PipeNet with associative-commutative receiver compute | High | High |
| 6 | Wave decomposition for wide overlapping multicast | PipeNet whose slot-per-pipe `block_count` exceeds 32 | Medium | Medium |
| 7 | Dynamic-slot streaming protocol | Same as (6), single-pass semantics required | High | Low |

Difficulty estimates assume the rewrite is implemented as a TTL pass
emitting existing TTKernel ops. Priority weights both the performance
gain on tuned tt-metal patterns (1→K broadcast and N→1 reductions
dominate the matmul + CCL workloads) and the size of the user-visible
problem the rewrite removes. (1) and (2) are low-risk prerequisites
that several later rewrites depend on. (5) is the largest correctness
win at scale because it removes the per-receiver `O(N)` L1 footprint
of slot-per-pipe gather. (6) and (7) target the same `block_count`
cliff; (6) is the simpler one and is preferred unless an explicit
single-pass requirement exists.

## Background: today's lowering

`convert-ttl-to-ttkernel` lowers each `ttl.create_pipe` to one fixed
primitive based on the Pipe shape. The mapping is:

| Pipe shape | Primitive | Receiver dataflow buffer |
| --- | --- | --- |
| `Pipe(src=p, dst=p')` (point) | `noc_async_write` + `noc_semaphore_inc` | `block_count = 2` |
| `Pipe(src=p, dst=slice(...))` (rectangular multicast) | `noc_async_write_multicast` + `noc_semaphore_inc_multicast` (after issue #505) | `block_count = max gather slot + 1` |
| Loopback multicast (`src` in dst range) | `noc_async_write_multicast_loopback_src` + remote `inc_multicast` + local `noc_semaphore_inc` | same as above |

Convergence is handled by `PipeGraph::assignGatherSlotIndices` (in
`lib/Dialect/TTL/Transforms/PipeGraph.h`), which greedy-colors pipes
that share a `(receiver, cbIndex)` pair so each pipe gets a distinct
slot index in the receiver dataflow buffer. `verifyGatherBlockCounts`
then requires `block_count >= max_slot_idx + 1` per receiver. This
constraint is what makes overlapping multicast unrepresentable when
`number-of-converging-pipes > 32`: the tt-metal dataflow buffer depth
is capped at 32 (enforced in `python/ttl/circular_buffer.py`), and the
slot table requires depth equal to the converging count.

## Topology-aware lowering catalog

Each of the following is a separate pattern rewrite at the
TTL-to-TTKernel boundary. Source IR is unchanged; only the emitted
TTKernel ops differ. Detection is local to one PipeNet plus its
receiver compute.

### Forwarding-chain lowering for 1→K rectangular broadcast

**Pattern.** A single multicast `Pipe(src=p, dst=slice_along_axis)`
where the receivers form a contiguous line along one axis and the
receiver compute does not reduce across senders (the receiver consumes
each block once and forwards to compute). The chain-vs-multicast
selection is K-dependent (see *Threshold selection* below).

**Rewrite.** Replace the one multicast write with `K-1` unicast hops
along the axis. The injector core (the source `p`) reads from DRAM (or
its dataflow buffer producer), pushes the block to its own dataflow
buffer, then issues `noc_async_write` to its successor in the line and
`noc_semaphore_set_remote` on the successor's "valid" semaphore. Each
downstream core, on receiving the valid signal, pushes the block to
its own dataflow buffer, then forwards to its successor by the same
primitive.

**Reference.** This is the pattern in [`minimal_matmul/device/kernels/dm_in0_sender.cpp:286-315`][dm-in0-chain]
and the in1 sibling [`dm_in1_sender_out.cpp`][dm-in1-out] for in0/in1
broadcast. The factory wires `next_core` and `prev_core` per row and
column at [`minimal_matmul_program_factory.cpp:719-722`][factory-in0-next]
and [770-773][factory-in1-next], and the comment at
[line 658][factory-chain-comment] calls the pattern a "forwarding
chain."

**Why this is faster than hardware multicast on Wormhole and
Blackhole.** A `noc_async_write_multicast` to a `K`-wide rectangle
serializes per-destination transactions at the source NoC port, so
the source port's egress bandwidth is the bottleneck and the other
`K-1` cores' NoC ports are idle during the broadcast. The chain issues
exactly one outbound transaction per core, so aggregate cross-section
bandwidth is `K * link_bw` instead of `1 * link_bw`. With the
pre-push-then-forward rewrite (next entry), the receiver compute
starts on each core in parallel with that core's forward to its
successor. Receiver dataflow buffer stays at `block_count = 2`.

**Threshold selection.** The chain-vs-multicast crossover depends on
`K`, the payload size, and the per-hop NoC latency. Multicast wins
when the multicast setup cost plus a single source-port-serialized
egress is less than `K-1` per-hop latencies plus `K` semaphore
handshakes. The crossover is not a constant: it varies with payload
size (large payloads amortize multicast setup; small payloads do
not) and with whether `cb_push_back` overlap is available on each
node (the pre-push rewrite shifts the chain's effective per-hop cost
toward zero when local compute is at least one hop in duration).

The threshold is therefore not a compile-time-known scalar. The
proposed approach: ship a microbenchmark under `test/python/perf/`
that sweeps `K ∈ {2, 4, 8, 16, 32}` and payload `∈ {1, 4, 16}` tiles,
records the per-pattern wall-clock, and emits a fitted decision
function (`use_chain(K, payload_tiles) -> bool`) consumed by the
lowering policy. Until the microbenchmark exists, the rewrite is
attempted only when `K > K_MIN_CHAIN` with `K_MIN_CHAIN` a
`PassOptions`-controlled integer that defaults to a value
empirically validated against the existing `test_mcast_matmul` and
`test_scatter_auto` baselines. Reference points: `minimal_matmul`
uses the chain unconditionally for any `K >= 2`;
`matmul_multi_core_reuse_mcast_*` uses multicast unconditionally for
weight broadcast at small `K`. Both are tuned for their workload, so
neither extreme is a sound default for arbitrary user PipeNets.

### Ring-decomposition lowering for N→1 reductions

**Pattern.** A PipeNet with `N` converging pipes whose receiver
compute is an associative-commutative reduction (the receiver compute
body matches the existing gather-then-sum shape: an initial
`acc_cb.store(t)` followed by `N-1` iterations of
`acc_cb.store(prev + t)` over `recv_cb.wait()`).

**Rewrite.** Replace the `N` gather pipes with a ring of `N-1` unicast
pipes carrying partial sums between neighbors. The reduction compute
moves into the per-step receiver block: each core adds its local
contribution to the incoming partial sum and forwards. After `N-1`
steps, the total lands at the designated reducer.

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

### Pre-push-before-forward peephole

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

### NOC0/NOC1 assignment from the dataflow graph

**Pattern.** Today the user manually splits work between `dm_read` and
`dm_write` to avoid the handshake deadlock that occurs when one
data-movement function does both `if_src` and `if_dst` work for an
overlapping multicast PipeNet (every core blocks on its own
`if_src`-issued sender handshake before any `if_dst` block can run to
release that handshake). The compiler has the dataflow graph and
should assign NOC and RISCV channels per kernel function rather than
relying on user partitioning.

**Rewrite.** A pre-emission pass partitions data-movement work across
the two NOCs based on PipeNet roles (source vs destination) and known
deadlock patterns, and emits the channel assignment as a kernel
attribute consumed by the program factory.

**Reference.** [`minimal_matmul_program_factory.cpp:229-247`][factory-noc-policy]
documents the explicit policy: small-input data movement on `RISCV_1`
/ `NOC_1`, large-input data movement on `RISCV_0` / `NOC_0`, with grid
transpose to keep the assignment symmetric for non-square outputs.

### Receiver-compute fusion

**Pattern.** A Pipe whose receiver dataflow buffer is consumed by
exactly one downstream compute kernel, with no reuse and no other
consumer.

**Rewrite.** Merge the receiver `if_dst` callback into the compute
kernel's read sequence. The receiver write lands directly in the
compute kernel's input register or its source dataflow buffer; the
intermediate staging dataflow buffer is removed.

**Reference.** [`llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in0_ring_all_gather.cpp`][llama-reader]
and the compute kernel [`bmm_large_block_zm_fused_bias_activation_gathered.cpp`][llama-compute]:
the matmul A reader is the all-gather receiver in one kernel.

**Why this is faster.** Removes one L1-to-L1 staging copy per
delivered tile, which reduces L1 bandwidth pressure on the receiver
core, and frees one of the 32 dataflow buffer slots that would
otherwise be reserved for the staging buffer.

### Wave decomposition for wide overlapping multicast

**Pattern.** A single PipeNet whose slot-per-pipe `block_count` exceeds
the L1 budget or the 32-slot dataflow buffer cap.

**Rewrite.** The pass splits the PipeNet into `K` narrower PipeNets
executed sequentially, each with `block_count = ceil(N/K)` where `N`
is the original converging-pipe count. The user writes one wide
PipeNet; the compiler emits a sequential loop over the `K` waves.
Receiver compute runs once per wave with that wave's `block_count`,
and the accumulator dataflow buffer rolls across waves.

**Why this matters.** Decouples the PipeNet width `N` from the
`block_count` constraint, so the dataflow buffer depth remains within
the 32-slot tt-metal cap regardless of `N`. Source IR is one PipeNet;
the rewrite is internal to lowering.

### Dynamic-slot streaming protocol

**Pattern.** Overlapping multicast where the sender count exceeds the
dataflow buffer depth and the wave-decomposition above is not
preferred (e.g. when the user wants single-pass semantics).

**Rewrite.** Replace the compile-time slot table with a per-PipeNet
shared atomic counter at each receiver. Senders do an
atomic-fetch-add on the counter to claim a slot before the multicast
write. The receiver dataflow buffer depth becomes the in-flight
window, not the pipe count. The cumulative `experimental::semaphore_wait_min`
already used for issue #505 is compatible: it counts arrivals, not
slot identity.

**Cost.** One atomic round-trip per send. Acceptable for patterns
where `N` is large enough that the per-send cost is amortized over a
multi-tile message.

## Pattern detection inputs

All of the patterns above can be detected from data already present during
the TTL-to-TTKernel lowering:

- `PipeGraph` records all PipeNets, their pipes, and the `(receiver, cbIndex)` 
  convergence relation (`lib/Dialect/TTL/Transforms/PipeGraph.h`).
- `ttl.if_dst` callback bodies expose the receiver compute
  (whether it stores a single tile or accumulates).
- Pipe shape attributes (`I64Attr` source and destination ranges)
  expose contiguity along an axis.
- The kernel-thread `func.func` attributes and the active-set guard
  pass (see `PipeNets.md`) expose role assignment per node.

No new dialect or attribute is required to detect any of the patterns.
The lowering policy can be a sequence of rewrites attempted in order at
the start of `convert-ttl-to-ttkernel`, with the existing per-shape
lowering as the fallback.

## Test strategy

Each rewrite will be tested with a combination of pattern-specific lit tests 
and end-to-end pytests:

1. Lit tests under `test/ttlang/Dialect/TTL/Transforms/` confirm the
   rewrite fires on the expected pattern and emits the expected
   TTKernel sequence (`noc_async_write` chain, ring `noc_semaphore_inc`,
   etc.). Negative tests confirm the fallback fires when the pattern
   does not match.
2. Pytests under `test/python/pipe/` confirm correctness against a
   torch reference for the same source-level PipeNet, before and after
   the rewrite is enabled. The same pytest source runs on hardware
   via the compiler and on the simulator via `test/scripts/ttlang-sim-pytest`,
   matching the test policy in `PipeNets.md`. A regression-grade
   pytest like the existing `test_scatter_gather` exercises the
   end-to-end lowering pick.

## Possible implementation order

The rewrites are independent but ordering reduces risk:

1. Pre-push-before-forward peephole. Pure TTKernel-level rewrite. No
   PipeNet semantics change.
2. NOC0/NOC1 assignment from the dataflow graph. Removes a class of
   user-visible deadlocks (`test_scatter_gather` had to be authored
   with manual NOC split). Independent of the other rewrites.
3. Forwarding-chain lowering for 1→K rectangular broadcast.
   Depends on (1) for the pre-push idiom.
4. Receiver-compute fusion. Depends on stable PipeNet receiver shape;
   independent of the chain rewrite.
5. Ring-decomposition lowering for N→1 reductions. Largest
   dataflow rewrite; depends on the receiver-compute pattern matcher
   built in (4).
6. Wave decomposition for wide overlapping multicast. Bounded scope
   (one PipeNet -> many PipeNets in sequence); does not depend on the
   above.
7. Dynamic-slot streaming protocol. Largest behavioral change in the
   sender lowering and the most invasive on the receiver semaphore
   protocol. Last.

## Interaction with the active-set guard pass

The active-set guard pass (`PipeNets.md`) computes the union of every
pipe's source and destination range and wraps every kernel function
body in an `scf.if` predicate. The rewrites above introduce new pipes
(forwarding chain) or eliminate pipes (receiver-compute fusion) at
lowering time. The active set must be recomputed after the rewrite,
or the rewrites must run after the guard pass on the rewritten pipes.
Running the guard pass twice (once before, once after) is the simplest
option and is consistent with the pass's existing idempotence
(`ttl.pipenet_active_guard` marker attribute). Pipeline placement is a
design choice for the rewrite pass.

## Non-goals

* Cross-chip lowering. The rewrites here are intra-chip. Cross-chip
  lowering is captured separately in `PipeNets.md` Future work.
* New dialect ops. All rewrites emit existing TTKernel ops. The
  forwarding-chain lowering uses `noc_async_write` and
  `noc_semaphore_set_remote`, both already present.
* User-visible API changes. The PipeNet language stays as it is; the
  rewrites are internal to the lowering.

## Open questions

* Should the lowering policy be controllable per PipeNet via an
  attribute on `ttl.create_pipe` (e.g. `lowering = "chain" | "multicast"
  | "ring"`) for cases where the user knows better than the heuristic?
  The default would remain heuristic-driven.
* Wave decomposition and dynamic-slot streaming both target the same
  `block_count` constraint with different cost profiles (extra
  semaphore traffic versus extra atomics per send). Picking one
  default and exposing the other behind a `PassOptions` flag is the
  simplest path.

## References

[dm-in0-chain]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp#L286-L315
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
