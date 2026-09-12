# All-gather matmul performance

This comparison measures column-parallel all-gather matmul with K-sharded
activation and N-sharded weight, bias, and output. The four output shards
collectively contain one `M x N` result.

## Results

Four Blackhole P150b devices; global `M/K/N=9472/5120/15360`; per-device
`M/K/N=9472/5120/3840`.

| Implementation | Device median ms (min-max) | TT-Lang/native | Warmups/samples |
| --- | ---: | ---: | ---: |
| TT-Lang | 3.200 (3.163-3.222) | 1.621 | 3/10 |
| Native `all_gather_minimal_matmul_async` | 1.974 (1.959-2.004) | 1.000 | 3/10 |

Both results passed PCC >= 0.99 and elementwise relative/absolute tolerances of
0.05 against FP32 PyTorch for every warmup and sample.

## Configurations

Each implementation uses its selected grid, blocking, and communication
configuration. Equal resource use is not required.

| Setting | TT-Lang | Native |
| --- | --- | --- |
| Activation input/device | `9472 x 1280`, K-sharded; storage padded to `10752 x 1280` | `9472 x 1280`, K-sharded |
| Weight input/device | distinct `5120 x 3840` N shard | distinct `5120 x 3840` N shard |
| Bias input/device | distinct `1 x 3840` N shard | distinct `1 x 3840` N shard |
| Output/device | one distinct `9472 x 3840` N shard | three adjacent `9472 x 1280` chunks comprising one distinct `9472 x 3840` N shard |
| Compute grid | `12 x 10`; 120 compute workers | `12 x 9`; 108 compute workers |
| M/K/N blocks | `4/10/12` tiles | `7/5/16` tiles |
| Output subblock | `1 x 4` tiles; direct FP32 packer accumulation | `1 x 2` tiles |
| Communication workers | 4 fabric workers directly inject activation blocks; point-to-point forwarding through each 10-node compute chain | 24 compute workers are fabric clients; 4 mux-only workers |
| Activation collective | one-direction TT-Lang ring into L1 | bidirectional native ring into gathered-activation DRAM storage |
| Fabric configuration | 2D, strict initialization | 1D ring, strict initialization |
| Payload | 8192 bytes | 8192 bytes |
| Links/workers/channel buffers | one directed ring connection per fabric worker | 2 links; 6 workers/link; 24 buffers/channel |
| Arithmetic | BF16 input/output; HiFi2; FP32 destination and packer accumulation | same; `math_approx_mode=true`; three output chunks |

The native grid and block configuration is the entry for
`(9472, 5120, 3840)` in TT-Metal's
[`grid_12_9_configs`](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/models/tt_dit/utils/matmul.py#L164-L168).
The corresponding
[`sweep_mm_block_sizes.py`](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/models/tt_dit/utils/sweep_mm_block_sizes.py#L130-L136)
case uses the same per-device `M/K/N`, `12 x 9` grid, three output chunks, and
approximate-math setting. Validation concatenates the three adjacent native
chunks on the host; that untimed concatenation reconstructs the same N shard
that TT-Lang returns as one tensor.

## Optimization experiments

All rows use the four-device workload and timing definition above. Accepted
results use three warmups and ten samples; screening and adjacent-control rows
state their smaller counts. Activation-only measurements keep the same
activation transport and DFB sequence, then discard each block after the
consumer wait; they isolate communication and local distribution from matmul.

| Experiment | Device median ms (min-max) | Warmups/samples | Change | Result |
| --- | ---: | ---: | ---: | --- |
| Complete operation: 10 communication workers, row multicast | 3.577 (3.540-3.603) | 3/10 | control | Rejected; point-to-point was 4.5% faster. |
| Complete operation: 10 communication workers, point-to-point row forwarding | 3.416 (3.312-3.435) | 3/10 | -4.5% vs multicast | Accepted in place of multicast. |
| Activation only: 10 communication workers, row multicast | 2.943 (2.834-2.993) | 3/10 | control | Matched local-distribution control. |
| Activation only: 10 communication workers, point-to-point row forwarding | 2.847 (2.700-2.883) | 3/10 | -3.3% vs multicast | Confirms that multicast underperforms point-to-point without matmul. |
| Activation only: 4 communication workers, direct fabric-to-compute injection | 2.521 (2.511-2.575) | 1/3 | -11.5% vs 10 workers | Selected for complete-operation measurement. |
| Activation only: 6 communication workers | 2.731 (2.728-2.733) | 1/3 | -4.1% vs 10 workers | Rejected. Two relay workers add L1 transfers and synchronization. |
| Activation only: 8 communication workers | 2.516 (2.509-2.558) | 1/3 | -11.6% vs 10 workers | Statistically equivalent to 4 workers in this screen. |
| Complete operation: 8 communication workers | 3.205 (3.196-3.239) | 1/3 | -6.2% vs 10 workers | Rejected; no complete-operation benefit over 4 workers. |
| Complete operation: 4 communication workers, direct fabric-to-compute injection | 3.200 (3.163-3.222) | 3/10 | -6.3% vs 10 workers | Accepted. Removes six relay workers and their L1 transfers. |
| Complete operation: 10 communication workers, post-candidate control | 3.393 (3.382-3.451) | 1/3 | control | Adjacent control confirms a 5.7% four-worker reduction. |
| Split M-worker rows between both ring directions | 3.869 (3.852-3.908) | 3/10 | +8.2% vs one direction | Rejected. Each M group requires its own weight stream; this does not reproduce native's K-half exchange. |
| Reduce the compute K block from ten to five tiles | 4.547 (4.535-4.560) | 3/10 | +27.1% vs ten tiles | Rejected. The smaller block doubles DFB and matmul message granularity. |
| Remove the compute-chain receive DFB and its local copy; three-block activation DFB | 3.200 (3.179-3.232) | 3/10 | +0.6% vs 3.180 adjacent control | Rejected. The result is statistically equivalent and provides no complete-operation benefit. |
| Unchanged four-worker implementation after direct-DFB experiment | 3.180 (3.155-3.243) | 3/10 | control | Confirms no measurable benefit from removing the local copy. |
| Alternate complete ten-tile K blocks across both ring directions | not measured | full-size compile | n/a | Rejected. The small four-device BF16 streaming case passed, but the full workload required 1,474,560 L1 bytes, 13,184 bytes over the 1,461,376-byte budget. |
| Eight direct fabric managers to reduce per-manager DFB capacity | not measured | full-size launch | n/a | Rejected. Compilation and PipeNet verification passed, but four physical forwarding links could not bind eight interfering managers. |
| Four bidirectional managers with two-block receive and relay DFBs | not measured | full-size launch | n/a | Rejected. The local relay filled while fabric sends waited for peers to post receives, producing a protocol deadlock. |
| Four bidirectional managers with one-row local staging and six-block receive and relay DFBs | not measured | full-size compile | n/a | Rejected. The small full-grid case passed; at full size a 491,520-byte communication DFB had only 313,600 bytes available. |
| Per-entry-pipe full-block bidirectional exchange with one receive and two relay blocks | not measured | full-size launch | n/a | Rejected. The small full-grid case passed and the full configuration fit L1, but the full workload did not complete within 60 seconds after compilation. |
| Direction-major consumption of per-entry-pipe full blocks | not measured | small correctness | n/a | Rejected. It compiled at the full 12x10 grid, but the four-device small case did not complete within 180 seconds. Grouping K blocks by direction did not resolve the protocol stall. |
| Per-row bidirectional exchange with one-block staging/receive and two-block relay DFBs | not measured | small correctness | n/a | Rejected. Four-device PCC was 0.257; a two-device diagnostic proved that remote K weights were paired with a repeated local activation shard. |
| Bidirectional exchange with five-tile K blocks | 7.002 (6.961-7.033) | 1/3 | +118.8% vs selected result | Rejected. It passed full-size correctness but the doubled matmul and DFB granularity exceeded the benefit of the second fabric direction. |
| Bidirectional exchange with two K halves assembled into each ten-tile matmul block | 4.349 (4.323-4.362) | 1/3 | +36.7% vs 3.182 adjacent control | Rejected. It passed full-grid and full-size correctness but required eight row-segment L1 copies per activation block. |
| Unchanged four-worker implementation after K-half assembly experiment | 3.182 (3.161-3.204) | 1/3 | control | Adjacent control confirms the K-half assembly regression. |
| Bidirectional K halves received directly into ten-tile matmul-block subviews | 10.963 (10.879-10.969) | 1/3 | +242.6% vs selected result | Rejected. Eight receive/forward transactions per activation block cost more than the eight eliminated L1 assembly copies. |
| Bidirectional K halves assembled at each compute-row head, then forwarded as one ten-tile block | 4.134 (4.127-4.143) | 1/3 | +29.7% vs 3.186 adjacent control | Rejected. Row-head-only assembly improved the 4.349 ms all-node assembly result by 4.9%, but remained slower than one-direction transport. |
| Unchanged four-worker implementation after row-head assembly experiment | 3.186 (3.177-3.213) | 1/3 | control | Adjacent control confirms the row-head assembly regression. |

In the multicast implementation, each compute-row head sent every activation
block to the other nine workers in that row. Point-to-point forwarding sent
each block once per adjacent worker. With identical ten-worker fabric and
compute configurations, point-to-point reduced activation-only time by 3.3%
and complete-operation time by 4.5%; multicast therefore underperformed and
was removed.

The four-worker result removes the separate L1 distribution stage. Its 2.521 ms
activation-only screening result still exceeds the 2.109 ms isolated matmul
time. The complete operation is 0.679 ms above their maximum, so activation
communication and communication-compute overlap remain optimization targets.

The bidirectional experiments require both directions to populate one ten-tile
activation block and the matching two weight slices before one matmul. Splitting
the matmul into five-tile blocks is correct but slow; retaining ten-tile matmul
granularity requires receiver offsets within one reserved DFB block. The
subview prototype restored that granularity, but its row-segment assembly still
measured 36.7% slower than the adjacent one-direction control.

Receiving directly into the final activation-block subviews removed the local
assembly copies. It instead issued eight receive/forward transactions per
activation block: four M rows for each of two K halves, with 10,240 bytes per
transaction in the measured configuration. The 10.963 ms result shows that the
additional address-publication and synchronization work exceeded the removed
L1-copy cost.

Assembling both K halves only at each compute-row head reduced assembly from
ten nodes per row to one. Each row head then forwarded one complete ten-tile
block through the row. This reduced the all-node assembly time from 4.349 ms to
4.134 ms, but the two half-block fabric streams and their synchronization still
exceeded the one-direction implementation's cost.

The per-entry-pipe full-block revision reordered each manager's exchange from
six buffered M rows per source device to all source devices for one M row. This
reduced activation storage per manager from thirteen full blocks (six receive,
six relay, and one staging) to four (one receive, two relay, and one staging).
The small full-grid case passed, and the full configuration compiled within the
L1 budget. The full workload did not complete, so the protocol was rejected
without reporting a device time. Consuming all increasing-direction K blocks
before decreasing-direction K blocks also stalled, including on the small
four-device case; producer-consumer ordering alone is not the cause.

## TT-Lang matmul kernel

The compute kernel converts each bias block to FP32 and broadcasts it into the
accumulator. Each BF16 activation/weight block matmul packs directly into that
FP32 accumulator, preserving destination accumulation across the global K loop.
The result is converted to BF16 once, immediately before the output DFB write.
The compiler emits the full matmul initialization and suppresses the redundant
short initialization that would reset packer accumulation.

## Measurement

Device profiling measures first kernel start through final kernel end, averaged
across the four devices. Host tensor creation, compilation, dispatch,
correctness checks, and profiler processing are excluded.

Measured 2026-09-11 20:33-20:34 UTC (TT-Lang) and 18:40-18:41 UTC (native):
TT-Lang source `26e3dff9120`, operation SHA-256 `639b6d92e539`, comparison
runner SHA-256 `8f27dcbf7d4b`; TT-Metal
`ea042c4ad623`; LLVM `37aca9d384347`; firmware 18.12.1; IRD v1.1.9.

[Raw device-profiler reports](https://gist.github.com/brnorris03/fa7ab25c12872de92dc0727f28f16104).
[Reproduction command and timing definition](README.md#run-the-comparison).
