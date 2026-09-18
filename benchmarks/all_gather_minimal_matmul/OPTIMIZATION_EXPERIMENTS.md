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
