# All-gather matmul optimization experiments

This file preserves the implementation-search results. The concise accepted
comparison is in [Performance](PERFORMANCE.md).

Unless stated otherwise, rows use four Blackhole P150b devices; global M/K/N =
9472/5120/15360; BF16 input/output; FP32 destination accumulation; and device
time from first kernel start through final kernel end. Accepted measurements
use three warmups and ten samples. Screening rows state their smaller counts.
Every timed row passed the correctness checks described in
[Performance](PERFORMANCE.md).

The [raw report archive](https://gist.github.com/brnorris03/fa7ab25c12872de92dc0727f28f16104)
is the authoritative provenance record. Each report records the TT-Lang Git
revision, modified-source SHA-256 values, compiler binary SHA-256, dependency
pins, firmware, and timestamp. Modified operation content differs between
experiments, so its SHA-256, not the Git revision alone, identifies the tested
implementation. The selected 1.847 ms result has provenance
[P4](PERFORMANCE.md#p4-four-device-column-parallel).

## Results

| Experiment | Device median ms (min-max) | Warmups/samples | Change | Result |
| --- | ---: | ---: | ---: | --- |
| 10 communication workers, row multicast | 3.577 (3.540-3.603) | 3/10 | control | Rejected; point-to-point was 4.5% faster. |
| 10 communication workers, point-to-point row forwarding | 3.416 (3.312-3.435) | 3/10 | -4.5% | Accepted in place of multicast. |
| Activation only: 10 communication workers, row multicast | 2.943 (2.834-2.993) | 3/10 | control | Local-distribution control. |
| Activation only: 10 communication workers, point-to-point row forwarding | 2.847 (2.700-2.883) | 3/10 | -3.3% | Confirms the complete-operation result. |
| Activation only: four communication workers, direct fabric-to-compute injection | 2.521 (2.511-2.575) | 1/3 | -11.5% vs 10 workers | Selected. |
| Activation only: six communication workers | 2.731 (2.728-2.733) | 1/3 | -4.1% vs 10 workers | Rejected; relay workers add L1 transfers and synchronization. |
| Activation only: eight communication workers | 2.516 (2.509-2.558) | 1/3 | -11.6% vs 10 workers | Equivalent to four workers in this screen. |
| Complete operation: eight communication workers | 3.205 (3.196-3.239) | 1/3 | -6.2% vs 10 workers | Rejected; no benefit over four workers. |
| Complete operation: four communication workers, direct injection | 3.200 (3.163-3.222) | 3/10 | -6.3% vs 10 workers | Accepted. |
| Complete operation: 10-worker adjacent control | 3.393 (3.382-3.451) | 1/3 | control | Confirms a 5.7% four-worker reduction. |
| Split M-worker rows between ring directions | 3.869 (3.852-3.908) | 3/10 | +8.2% | Rejected; each M group requires its own weight stream. |
| Reduce compute K block from ten to five tiles | 4.547 (4.535-4.560) | 3/10 | +27.1% | Rejected; doubles DFB and matmul message granularity. |
| Direct publication, one-block output DFB, four-tile M block | 3.189 (3.153-3.215) | 1/3 | control | Direct publication alone was equivalent to the prior result. |
| Same DFB configuration, five-tile M block | 2.631 (2.593-2.688) | 3/10 | -17.5% | Accepted; M rounds fell from seven to five. |
| Group three compute rows per fabric transfer | 2.306 (2.268-2.318) | 3/10 | -11.8% vs 2.613 ms control | Accepted; transfers/worker fell from 180 to 60. |
| Submit intermediate packets without completion waits | 2.269 (2.232-2.320) | 3/10 | -1.6% | Accepted; the final write-and-atomic remains blocking. |
| Bidirectional L1 transport, one mux buffer/client channel | 2.286 (2.267-2.338) | 3/10 | +1.1% vs adjacent controls | Rejected; one slot serializes 12 packets per activation half. |
| Bidirectional L1 transport, 22 mux buffers/client channel | 2.243 (2.179-2.255) | 3/10 | +0.2% vs adjacent controls | Accepted; maximum uniform depth fitting mux L1. |
| Defer output writes to the weight thread | 1.959 (1.945-1.983) | 3/10 | -11.0% vs adjacent controls | Accepted; paired native was 1.969 ms. |
| Publish received weight halves directly into matmul DFB | 1.847 (1.802-1.879) | 3/10 | -6.2% vs 1.968 ms control | Accepted; paired native was 1.970 ms. |
| Reduce weight DFB from three half-blocks to two | 1.986 (1.980-1.993) | 1/3 | +7.6% | Rejected; three half-blocks are required to hide delivery. |
| Push source weight DFB before row-multicast wait | 1.827 (1.825-1.832) | 1/3 | -1.1% | Rejected; generated C++ retained the prior ordering. |
| 13 x 10 compute grid, M/K/N blocks 4/10/12 | not measured | full-size launch | n/a | Rejected; no nodes remain for eight mux workers. |
| Alternate complete ten-tile K blocks between ring directions | not measured | full-size compile | n/a | Rejected; required L1 exceeded the budget by 13,184 bytes. |
| Eight direct fabric managers | not measured | full-size launch | n/a | Rejected; four forwarding links could not bind eight interfering managers. |
| Four bidirectional managers, two-block receive and relay DFBs | not measured | full-size launch | n/a | Rejected; finite-capacity protocol deadlock, issue 1037. |
| Four bidirectional managers, one-row staging and six-block receive/relay DFBs | not measured | full-size compile | n/a | Rejected; communication DFB exceeded available L1. |
| Per-entry-pipe full-block exchange, one receive and two relay blocks | not measured | full-size launch | n/a | Rejected; full workload exceeded 60 seconds. |
| Direction-major per-entry-pipe consumption | not measured | small correctness | n/a | Rejected; small four-device case exceeded 180 seconds. |
| Per-row bidirectional exchange | not measured | small correctness | n/a | Rejected; PCC 0.257 from repeated local activation pairing. |
| Bidirectional exchange, five-tile K blocks | 7.002 (6.961-7.033) | 1/3 | +118.8% | Rejected; doubled matmul and DFB granularity. |
| Bidirectional K halves assembled into ten-tile blocks | 4.349 (4.323-4.362) | 1/3 | +36.7% | Rejected; eight row-segment L1 copies/block. |
| Unchanged four-worker control after K-half assembly | 3.182 (3.161-3.204) | 1/3 | control | Confirms the assembly regression. |
| Receive K halves directly into matmul-block subviews | 10.963 (10.879-10.969) | 1/3 | +242.6% | Rejected; eight receive/forward transactions/block. |
| Assemble K halves at compute-row head | 4.134 (4.127-4.143) | 1/3 | +29.7% | Rejected; synchronization cost remained dominant. |
| Unchanged four-worker control after row-head assembly | 3.186 (3.177-3.213) | 1/3 | control | Confirms the row-head assembly regression. |

## Selected changes

| Change | Measured effect | Mechanism |
| --- | ---: | --- |
| Four direct communication workers | -6.3% vs 10 workers | Removed six relay workers and their L1 transfers. |
| Five-tile M blocks | -17.5% | Reduced M rounds from seven to five and padded M tiles from 336 to 300. |
| Three-row grouped transfers | -11.8% | Reduced fabric transfers/worker from 180 to 60 without changing payload bytes. |
| Asynchronous intermediate packets | -1.6% | Removed intermediate remote-completion waits; retained final completion ordering. |
| Deferred output writes | -11.0% | Overlapped preceding output writes with publication of the next inputs. |
| Direct weight publication | -6.2% | Removed 2.359 GB/device of local staging traffic. |

Direct weight publication reduced weight wait by 83.8 us and matmul/control by
44.7 us; activation wait increased by 14.1 us. The selected implementation
therefore reduced complete device time from 1.968 to 1.847 ms.

Point-to-point row forwarding replaced multicast because it reduced
activation-only time by 3.3% and complete-operation time by 4.5% with identical
fabric and compute configurations.

## Supporting measurements

| Measurement | Result |
| --- | --- |
| Grouped-transfer adjacent control | 2.613 ms (2.605-2.616) |
| Deferred-output adjacent controls | 2.212 and 2.193 ms |
| Direct-weight activation-counter run | 1.842 ms |
| Direct-weight weight-counter run | 1.820 ms |
| Direct-weight local copies removed | 19.661 MB/worker; 2.359 GB/device |
| Bidirectional half payload | 51,200 bytes in 12 packets; 4,352-byte maximum packet payload |
| Direct-subview receive granularity | eight 10,240-byte receive/forward transactions per activation block |
| Full-block exchange storage | four blocks/manager after reduction from thirteen |
| Six-block receive/relay communication DFB | 491,520 bytes required; 313,600 bytes available |
| Alternate ten-tile K blocks | 1,474,560 bytes required; 1,461,376-byte L1 budget |

## All-shape configuration search

The paired all-shape table in [Performance](PERFORMANCE.md) reports one
confirmed configuration per implementation and semantic input. This section
records how those configurations were selected. Screening used one warmup and
three samples; confirmation reruns used three warmups and ten samples at one
TT-Lang revision. Raw screen reports are archived outside Git.

### Native

The pinned manifest lists 155 AGMM/SAGMM rows covering 63 semantic inputs that
the native operation supports; 46 of them (plain and QKV) are also comparable
with TT-Lang. Several inputs appear in multiple rows that differ only in the
upstream source grid.

- Source-grid sweep: every legal source-grid row was measured with the upstream
  blocking heuristic and `FABRIC_1D_RING`. 132 candidates completed: 107
  passed and 25 failed. The rows cover 45 inputs; 41 had a passing candidate.
  Failures were oversized or invalid alternate grids, validation results that
  the dtype-aware BF16 tolerance later superseded, and process-contention
  artifacts. Winners are selected per input, not per row.
- Recovery screen: the four inputs without a passing row were rerun on the
  `12x9` grid; all passed. A `13x9` grid is invalid because the device
  exposes at most 12 compute columns.
- Full-K 768 and 1536 screen: the upstream heuristic returns an eight-tile K
  block for local K shards of 6 or 12 tiles, which the operation rejects. A
  constrained screen ranked local-K divisors, upstream-style block candidates,
  the `12x9` source grid, and configurations below a 1400 KiB L1 estimate,
  then measured the three highest-ranked configurations for each of the 18
  affected inputs: 54 candidates, 53 passed. `16384x768x4608` with
  `M4/K6/N8` terminated its isolated worker with `SIGBUS` and is excluded;
  the crash left the UMD system-memory mapping in place, and the next launch
  failed before program creation until the mapping was released.
- Confirmation: the fastest correct candidate per input across the three
  screens was rerun with three warmups and ten samples.

### TT-Lang

The operation was not modified during this search. A roofline model
enumerates compute grids whose worker count plus the eight fabric-mux workers
fits the device's Tensix worker grid, even K-block divisors of the per-device
K shard, N blocks that divide the per-device N shard across the grid rows,
and modeled DFB allocations below 1,350,000 bytes. Candidates are ordered by
the maximum of the compute, fabric, and DRAM lower bounds plus a
per-matmul-call term. With the 130-worker grid of the accepted result's
host, the model ranks the accepted `12x10`, `M5/K10/N12` configuration first
for the 9472/5120/15360 input; that agreement is the control before applying
the model to the remaining inputs.

The P150b boards on the all-shape host expose a 12x10 worker grid (120
workers), so the accepted 12x10 configuration cannot host its eight mux
workers there, and the all-shape screen is limited to 112 compute workers.
For 9472/5120/15360 the model then ranks `11x10`, `M9/K8/N12` first. The
three highest-ranked configurations for each of the 46 comparable inputs
were measured: 138 candidates, 138 passed. An earlier pass that admitted
13-column grids failed 25 candidates before the device grid became an
explicit model input; those grids are excluded, not re-ranked failures. The
fastest correct candidate per input was confirmed with three warmups and ten
samples.

### Where TT-Lang loses on the all-shape inputs

Measured on the 13x10 host with the same builds, three warmups and ten
samples, using per-call signpost scopes on the compute thread's activation
and weight waits, on its prologue, output reserve, and output store, and on
the weight thread's previous-block output write; native used accumulated
wait counters in its compute kernel. Times are on the operation-ending
compute core.

| Input (M, full K, full N) | TT-Lang ms | Native ms | TT-Lang activation wait | TT-Lang weight wait | TT-Lang output store | Native activation wait | Native weight wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096, 6144, 18432 (both 12x9 M11/K8/N8) | 1.381 | 1.143 | 187 us | 157 us | 142 us (2 blocks) | 200 us | 22 us |
| 16384, 768, 18432 (12x9 M11/K6/N8; native M4/K6/N16) | 1.312 | 1.174 | 163 us | 4 us | 674 us (8 blocks) | 501 us | 197 us |
| 16384, 6144, 9216 (12x9 M11/K8/N8; native M12/K6/N8) | 2.712 | 2.666 | >= 236 us | >= 157 us | 344 us (4 blocks) | 799 us | 102 us |

The matmul itself is not the gap: per-call time is 8.9 us for an 11x4x8-tile
half block and 6.7 us for 11x3x8, within 7% of the compute bound. Two
structural costs remain.

- Output write burst. The output store costs 30.8 us for the first block of
  a core and 88 to 121 us for every later block. The extra time is the pack
  thread's `reserve_back` on the one-block output DFB, which waits for the
  weight thread to write the previous block to DRAM. That write takes 7 us
  on the core nearest the NoC origin and 70 to 100 us at the far corner
  because all 108 cores write a 176 KiB block at the same block boundary,
  a 19 MB burst that DRAM absorbs at about 200 GB/s in NoC arbitration
  order. Native writes output subblocks during the matmul and never forms
  the burst. The cost scales with output blocks per core: eight for
  16384/768/18432, four for 16384/6144/9216, two for 4096/6144/18432.
- Source-last weight publish. The column-0 worker multicasts each weight
  half along its row and publishes it to its own compute only after the
  multicast completes, so weight wait is about 210 us on the two columns
  nearest the source and about 2 us on the far column. Native's weight wait
  on the ending core is 22 us for the same blocks.

With a two-block output DFB the store becomes a constant 30.7 us and the
reserve wait disappears, but device time improves only 1 to 4% (4096/6144/
18432 1.358 ms; 16384/6144/9216 2.692 ms; 16384/768/18432 1.255 ms; accepted
9472/5120/15360 1.821 ms versus 1.843 ms) because the slowest core moves from
the far corner to the source-adjacent column, whose input waits are of the
same size. The two-block DFB fits 41 of the 46 confirmed TT-Lang configurations
within the 1,461,248-byte L1 budget (the 12x9 M11/K8/N8 rows keep 134,144
bytes of margin) and is now the operation's default (`output_block_count`,
with 1 available for the five 11x10 M9/K8/N12 and 11x9 M6/K8/N16 rows that
exceed it); the all-shape table was measured with the one-block DFB. On the all-shape host the second
block changes nothing for the worst rows (4768/5376/21504 QKV 3.785 to 3.773
ms; 4096/6144/18432 2.462 to 2.468; 8192/6144/36864 9.835 to 9.853;
8192/6144/18432 QKV 4.918 to 4.928; 8192/1536/36864 2.959 to 3.018), and the
11x10 M9/K8/N12 configuration of the 9472/5120/15360 input cannot host it
(1,548,288 bytes). The same decomposition on the all-shape host for 4096/6144/18432
(TT-Lang 2.502 ms, native 1.402 ms) attributes the whole difference to
activation delivery: on the critical compute core TT-Lang waits 1365 us for
activation (187 us on the 13x10 host) and 88 us for weights, while native
waits 466 us for activation (200 us on the 13x10 host) and 17 us for weights;
matmul and control is 1011 us for TT-Lang and 868 us for native on both hosts.
TT-Lang's per-half waits grow with the source distance (10, 26, and 57 us for
one, two, and three hops at the far core) because its two distribution DFBs
are one block deep and each hop relays a half only after both halves have been
multicast and copied locally, so a slower link is paid three times per K block;
native's transport has 24 channel buffers per client and DRAM staging, and
absorbs the same slower hops. The hosts differ in Ethernet provisioning: the
13x10 host's ring has four channels per edge, while the all-shape host is an
eight-chip cube with two channels per edge whose participants 7, 3, 1, 5 form
a face (direct neighbors, two channels per edge). TT-Lang plans four links
per direction with 24 fabric clients per direction (12 senders and 12
receivers that only return credits), so on the four-channel host every mux
serves 6 clients with 21 buffers each, while on the two-channel host one
direction is served by two muxes of 12 clients with 10 buffers and the other
by a single mux of 24 clients with 5 buffers. Native uses two links per
direction with 6 clients per link on both hosts. Reducing TT-Lang's client
count per direction (header-only channels for credit-only receivers, or
fewer endpoint rows when links are scarce) is therefore a host-independent
improvement alongside the transport depth. Deepening the distribution DFBs to two blocks
produces incorrect output on every input tried (the smallest input gives PCC
0.63), a multi-slot pipe-endpoint lowering defect that is recorded with its
reproducer and not adopted. Relaying each half as soon as the sender row
receives it, with the left relay issued after the right multicast receive is
posted (the schedule verifier rejects the relay before that receive as a
wait-for cycle), is correct and gives one to two percent on both hosts: 4096/6144/18432
2.455 and 2.480 ms in two runs on the all-shape host (2.449 ms with the
two-block output DFB) against 2.502 ms, and 1.357 ms on the 13x10 host
(1.356 ms with two output blocks) against 1.381 ms. Removing the source-last
publish is the next step on the 13x10 host; the earlier "push source
weight DFB before row-multicast wait" experiment targeted it and was
rejected only because the generated C++ kept the original ordering.
16384/768/18432 is DRAM-bound for both implementations: its output rate
during compute is about 350 GB/s.

## Eight-device configuration search

The native screen retained the published 12 x 9 transport configuration and
measured all eight M/K/N block combinations in {5, 7} x {5, 10} x {7, 8}.
Every candidate passed correctness with one warmup and three samples.
M7/K10/N8 had the lowest median at 1.705 ms and was confirmed with three
warmups and ten samples at 1.709 ms (1.690-1.722). M7/K10/N7 measured
1.721 ms; the 0.7% difference does not establish a decisive separation.

TT-Lang screened M9/K10/N6 against an M3 control on an 11 x 10 grid. M9 was
the best measured TT-Lang candidate and was confirmed at 1.798 ms
(1.784-1.852). Neither implementation exhaustively varied every grid,
communication-worker count, link count, channel depth, output chunk count, and
legal block size.
