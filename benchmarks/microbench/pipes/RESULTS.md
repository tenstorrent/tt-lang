<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PipeNet Microbenchmark Results

This benchmark measures the per-transfer cost of streaming N single tiles
core-to-core, (0,0) -> (1,0), comparing the tt-lang PipeNet (computed vs
receiver-published addressing) against a sequence of hand-written C++ kernels
that runs from a naive per-tile copy up to a bounded producer/consumer ring.

Date: 2026-06-23
Status: implemented; verified on hardware (Blackhole)

## Goal

Measure the per-transfer NoC cost of a tt-lang PipeNet unicast under the two
receiver-address protocols (computed vs receiver-published), and compare both
against hand-written C++ reference variants. Establish reusable infrastructure
to compare tt-lang-generated kernels against C++ baselines for paired
microbenchmarks.

## Background

- `examples/pipe_address_protocol_benchmark.py` (removed in d3d46ad0, kept as an
  external gist) timed the two protocols with `time.perf_counter()` and
  `synchronize_device` inside the measured loop. That measures host launch and
  sync overhead, not the on-device per-transfer cost.
- The two receiver-address protocols:
  - computed addresses (default): the sender computes the destination dataflow
    buffer (DFB) address at compile time.
  - receiver-published addresses (`--no-ttl-pipe-computed-addresses`): the
    receiver publishes its DFB address at run time; the sender waits for it
    before sending (a rendezvous).
  The expected difference is in fixed per-protocol setup, not steady-state
  per-tile NoC cost.
- The existing microbenchmarks run handwritten C++ kernels through
  `ttnn.generic_op` and read per-RISC Tracy device-profiler zones from
  `profile_log_device.csv`; the tt-lang compiler is not involved. The pipes
  benchmark adds the tt-lang side.

## Measurement

Primary metric: sender-core data-movement kernel duration from the Tracy device
profiler.

- Tracy device profiler: `TT_METAL_DEVICE_PROFILER=1`,
  `TT_METAL_PROFILER_MID_RUN_DUMP=1`. The profiler auto-records whole-kernel
  zones `BRISC-KERNEL` / `NCRISC-KERNEL` / `TRISC-KERNEL` per RISC per core into
  `profile_log_device.csv`. tt-lang user code cannot insert named zones, so
  whole-kernel duration is the common metric for both tt-lang-generated and C++
  kernels. The CSV-dump zones are used, not the Tracy GUI `.tracy` trace.
- Per-transfer extraction: sweep N (transfer count) and regress
  `sender_dm_us = fixed_us + per_transfer_us * N` (reuse `fit.py`'s linear
  model). The coefficient of N is the per-transfer NoC cost; the intercept is the
  protocol's one-time setup. A low fit residual confirms steady state.
- Cycles to microseconds via `CHIP_FREQ[MHz]` from the CSV header
  (`profiler.parse_chip_info`), the repo's canonical conversion.
- Sender duration is primary; receiver duration is reported for context.

Measured-loop structure for each N:

- N is the number of single-tile transfers in the compiled program.
- W is the number of warmup executions.
- R is the number of measured executions used for profiler sampling.

Execution sequence:

1. Build the N-transfer program.
2. Execute it once to compile; discard this run.
3. Execute W warmup runs.
4. Synchronize once, read the profiler once, and discard the warmup zones.
5. Execute R measured runs back-to-back with no sync between runs.
6. Synchronize once, then read the profiler once.
7. Reduce the R whole-kernel profiler samples per RISC with the median.

R does not enter the per-transfer regression directly. It only provides repeated
profiler samples for the same N. The per-transfer cost comes from sweeping N and
fitting `sender_dm_us = fixed_us + per_transfer_us * N`.

R is bounded so the on-device profiler buffer holds R x cores x zones; for the
2-core pipe this is small.

Two-pass fallback (documented, for benchmarks whose R x cores x zones would
overflow the buffer): a wall-only pass with the profiler quiet, plus the Tracy
pass using one profiler read per measured run, matching the existing C++ harness.

### Optional wall-time capability (shared)

Add a `--wall` flag to the shared measurement layer (`harness.py` and the new
`ttl_harness`). When set, wrap the R measured runs with one
`perf_counter` pair (one sync at the end, outside the loop) and emit:

- `wall_ms_per_run`: wall time divided by R.
- `wall_us_per_transfer`: wall time divided by `R * N`.

Wall time is end-to-end host cost (program launch plus sync), complementary to
the Tracy per-transfer number, not a substitute. Default off globally so
existing benchmark CSVs are unchanged; on for the PipeNet benchmark. CSV columns
are additive; `extrasaction="ignore"` and named-column readers (`fit.py`) are
unaffected.

## Layout

```
benchmarks/microbench/
  ttlang/
    __init__.py
    ttl_harness.py      # compile+run a @ttl.operation; warmup; R measured runs;
                        # one sync + one ReadDeviceProfiler; per-RISC
                        # kernel-duration summary; optional --wall
  compare.py            # union variant CSVs; per-variant regression over N;
                        # per_transfer_us, fixed_us, ratios vs baseline,
                        # computed-vs-published delta
  pipes/
    __init__.py
    ttlang_pipes.py     # ttlang_computed + ttlang_published; sweep N
    baseline_pipes.py   # one ideal C++ unicast; sweep N (2-core generic_op)
    kernels/
      pipe_sender.cpp   # core (0,0): semaphore handshake + N single-tile NoC
                        # sends to (1,0)
      pipe_receiver.cpp # core (1,0): signal-ready + drain
    RESULTS.md
```

Existing flat C++ benchmarks and `profiler.py` are unchanged. `harness.py` and
`runner.py` gain only the additive optional `--wall` capability (default off).

## tt-lang variants (`ttlang_pipes.py`)

Mirror the removed benchmark's DSL:

- `ttlang_computed`: `@ttl.operation(grid=(2, 1))`.
- `ttlang_published`:
  `@ttl.operation(grid=(2, 1), options="--no-ttl-pipe-computed-addresses")`.

Both: `net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])`, a compile-time
loop of N single-tile sends via `net.if_src` / `net.if_dst` with
`ttl.copy(blk, pipe)` / `ttl.copy(pipe, blk)`. The source is a DRAM tensor read
once per transfer into the send dataflow buffer (matching the original
benchmark); that per-transfer read is identical across both variants and the
C++ baseline, so the per-transfer cost difference attributes to the address
protocol. Isolating the PipeNet send from the read (L1-resident source) is a
future refinement; it does not affect the computed-vs-published comparison.

N is fixed per compile; the sweep recompiles per N (consistent with the C++
sweeps recompiling per config). The compiled program is cached, so the R
measured runs execute without recompiling.

Correctness: bit-exact. A PipeNet transfer is pure data movement (no compute, no
dtype conversion, no accumulation), so the destination tile must equal the
source tile exactly (`torch.equal`). PCC is a compute-era tolerance for bf16
rounding; here it would pass a PipeNet transfer that drops, duplicates, or
misaddresses tiles, which is exactly the failure the receiver-address protocols
could introduce. Widening bf16 to float32 for the compare is lossless, so it stays
bit-exact. The check is a hard assertion; any mismatch fails the run. This
applies to all three variants.

## C++ reference variants (`baseline_pipes.py` + kernels)

Hand-written unicast variants (`--mode all|floor|synced|dfb|optimized|ring`),
building from a naive per-tile transfer up to the bounded ring:

`cpp_baseline` (naive per-tile, no per-transfer flow control):
- `pipe_sender.cpp` on core (0,0): per transfer, read the source tile from DRAM
  into local L1 then `noc_async_write` it to (1,0)'s L1 (the destination
  dataflow buffer is allocated on both cores so the sender addresses it by the
  shared `get_write_ptr`). The per-transfer read mirrors the tt-lang PipeNet variant.
  After the last write, increment the receiver's done semaphore once.
- `pipe_receiver.cpp` on core (1,0): wait on the done semaphore, then write the
  received tile to the output DRAM tensor.

`cpp_baseline_synced` (double-buffered per-transfer credit handshake):
`pipe_sender_synced.cpp` / `pipe_receiver_synced.cpp` use two destination slots
and two cumulative-counter semaphores. `data` is receiver-owned and incremented
by the sender after each write. `free` is sender-owned and incremented by the
receiver after draining each slot. Each semaphore has a single incrementer, so
the kernels' `noc_semaphore_wait_min` is race-free. This isolates the cost of
flow control.

`cpp_baseline_dfb` (same-bookkeeping C++ baseline): `pipe_sender_dfb.cpp` does
everything synced does and stages each tile through a dataflow buffer with the
full `reserve_back`/`push_back`/`wait_front`/`pop_front` cycle (c_1,
block_count 2), matching the current tt-lang PipeNet sender's per-transfer DFB
bookkeeping. It reuses the synced receiver. This is the C++ baseline for
comparing the current PipeNet lowering. The difference from synced isolates the
DFB-staging cost.

`cpp_optimized` (the ceiling): `pipe_sender_optimized.cpp` leaves the per-tile
regime and follows the hand-tuned stateful-NoC pattern. It reads all N tiles
with one read barrier, programs the NoC write command once
(`noc_async_write_one_packet_set_state`) and reuses it per transfer
(`noc_async_write_one_packet_with_state`), and uses one write barrier. It has no
per-transfer barrier or command setup. It holds all N tiles in L1 (source and
destination), so the driver caps N at 128. Reuses the floor receiver. It is a
bulk-transfer ceiling, not a pipe: it drops bounded buffering, DFB staging, and
flow control, so the current PipeNet lowering cannot reach it.

`cpp_bounded_ring` (lower-level comparison target):
`bounded_ring_sender.cpp` / `bounded_ring_receiver.cpp` implement a bounded
producer/consumer ring. They preserve bounded slot reuse and data/free credit
flow control, but they are not a PipeNet implementation and they do not preserve
tt-lang's per-transfer DFB reserve/push/wait/pop bookkeeping. They use stateful
NoC writes and batch the read barrier, write barrier, and credit once per ring
chunk instead of once per tile. The buffer is two rings of `--ring-depths` slots
(lookahead 2): the sender writes the next ring while the receiver drains the
current one, so the two cores overlap. A single ring serializes them.
`--ring-depths` takes a comma list. Each depth is written as variant
`cpp_bounded_ring_r<ring>`. Deeper rings amortize the per-chunk handshake and
approach the bulk ceiling. Correctness is the bounded-buffer invariant: a ring
half is reused only after its previous occupant was drained.

Self-contained 2-core `ttnn.generic_op` program in `pipes/` (the
`MicroBenchmark` runner is single-core). Same N sweep, same regression, same CSV
schema, directly comparable as the floor. Kernels follow the existing
`kernels/` style: SPDX 2026 header, `api/dataflow/dataflow_api.h`, a contract
header comment, and documented runtime args.

## compare.py and CSV schema

Shared row schema written by both sweeps:

- config columns: N, dtype.
- `variant` names the tt-lang variant or C++ reference variant.
- sender per-RISC microseconds (brisc_us, ncrisc_us, dm_max_us), receiver dm
  microseconds.
- optional wall columns when `--wall`.
- bitexact (1/0) and mismatch_tiles (0 expected) instead of a PCC column.

`compare.py`:

- reads the variant CSVs (or one combined CSV), groups by config, regresses
  sender `dm_max_us` over N per variant into per_transfer_us and fixed_us, and
  the fit residual.
- emits per_transfer_us and fixed_us per variant, the ratio vs
  `cpp_bounded_ring_r8`, and the ttlang_computed minus ttlang_published delta.
  Measurement shows the protocol difference is in per_transfer_us (the
  rendezvous is paid per transfer), not fixed_us.
- keyed by the `variant` column so future paired benchmarks reuse it unchanged.

## Verification

Run in the hardware test environment over an N sweep (1,2,4,8,16,32,64,128,256;
cpp_optimized N<=128), all variants, bit-exact correctness required. Measured on
Blackhole, bf16, distinct tiles per transfer (bit-exact validates correct
per-tile delivery; r2 >= 0.999 for every variant, so the duration-vs-N fit is
linear and per_transfer_us is well defined).

cpp_bounded_ring is swept over ring depths, each a variant
cpp_bounded_ring_r<ring>. For each ring depth, `per_transfer_us` is the slope of
sender duration versus N. Ring depth changes that slope because one ring chunk
shares one read barrier, one write barrier, and one credit update across up to
`ring` transfers. Larger rings amortize that per-chunk work over more transfers.
Ratio is relative to cpp_bounded_ring_r8, a modest-depth bounded-ring target:

| variant                | per_transfer_us | vs target | regime                             |
|------------------------|-----------------|-----------|------------------------------------|
| cpp_optimized          | 0.050           | 0.38x     | bulk transfer ceiling (artificial) |
| cpp_bounded_ring_r64   | 0.062           | 0.48x     | bounded ring, ring 64              |
| cpp_bounded_ring_r32   | 0.073           | 0.56x     | bounded ring, ring 32              |
| cpp_bounded_ring_r16   | 0.091           | 0.70x     | bounded ring, ring 16              |
| cpp_bounded_ring_r8    | 0.129           | 1.00x     | bounded ring, ring 8 (target)      |
| cpp_baseline           | 0.606           | 4.69x     | naive raw NoC write                |
| cpp_baseline_synced    | 0.637           | 4.93x     | + credit handshake                 |
| cpp_baseline_dfb       | 0.695           | 5.37x     | + DFB staging                      |
| ttlang_computed        | 0.857           | 6.63x     | PipeNet, computed address          |
| ttlang_published       | 0.931           | 7.20x     | PipeNet, published address         |

Findings:

- The tt-lang PipeNet is ~6.6x the bounded-ring target (ring 8). Larger rings
  reduce the gap to the bulk ceiling: ring 8 is 0.129 µs/transfer, and ring 64
  is 0.062 µs/transfer (1.24x the bulk ceiling). The measured C++ bounded-ring
  variants do not show an inherent floor much above the bulk ceiling. An earlier
  draft claimed a ~0.082 µs/transfer floor and described the gap as inherent to
  bounded streaming. That claim was an artifact of a single-ring,
  non-overlapping bounded-ring variant.
- The tt-lang PipeNet does not show sender-ahead execution in this benchmark.
  `sender_dm_max` and `recv_dm_max` are comparable, and increasing `block_count`
  from 2 to 8 does not reduce sender time. The lowering emits generic
  per-transfer NoC writes instead of `set_state`/`with_state`, waits for the NoC
  barrier inside the transfer loop, and makes a send slot reusable only after
  that single-tile transfer completes. Removing the per-transfer `.wait()` is
  rejected as undefined behavior because the slot can be reused before the
  in-flight transfer completes. Sender/receiver overlap requires a PipeNet
  lowering change; buffer depth alone does not provide it.
- `cpp_bounded_ring` models the lower-level mechanism a sender-ahead PipeNet
  lowering would need: bounded slot reuse, a data/free credit handshake,
  stateful NoC writes, one barrier per ring, and two-ring lookahead. It is not
  the baseline for matching tt-lang's per-transfer DFB bookkeeping; that role
  belongs to `cpp_baseline_dfb`. Correctness is the bounded-buffer invariant: a
  ring half is reused only after its previous occupant drained. The distinct-tile
  bit-exact run is the empirical proof.
- Among the naive variants, flow control is cheap (+0.034 µs/transfer from
  cpp_baseline to synced) and DFB staging costs more (+0.060 from synced to dfb).
- Computed beats published by 0.074 µs/transfer, in per_transfer_us not fixed_us:
  the per-transfer receiver-address rendezvous the computed protocol removes.
- The reserve/push/wait/pop cycle is required for the current per-tile PipeNet
  lowering. A
  single-reserve send compiles but deadlocks for N > block_count (the staging
  block is never freed).
- All variants are bit-exact on distinct tiles, validating correct per-tile
  delivery.

Hardware verification precedes any MLIR lit tests or docs updates.

## Scope and non-goals (initial)

- Single tile per transfer, bf16, block_count=2. Tile-size and dtype axes are
  deferred; the regression over N is the deliverable.
- Only the unicast (0,0) to (1,0) topology, matching the removed benchmark.
  Multicast and scatter are out of scope here.
- No Tracy GUI `.tracy` trace capture; CSV-dump zones only.

## Open items

- The tt-lang per-RISC assignment of the PipeNet send (BRISC vs NCRISC) is read
  from the profiler, not assumed; `ttl_harness` takes the max over the sender
  core's data-movement kernels.
- Confirm the on-device profiler buffer holds R reps for the 2-core case during
  verification; pick R accordingly.
