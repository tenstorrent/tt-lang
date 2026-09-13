# All-gather matmul performance

This comparison measures column-parallel all-gather matmul with K-sharded
activation and N-sharded weight, bias, and output. The four output shards
collectively contain one `M x N` result.

## Results

Four Blackhole P150b devices; global `M/K/N=9472/5120/15360`; per-device
`M/K/N=9472/5120/3840`.

| Implementation | Device median ms (min-max) | TT-Lang/native | Warmups/samples |
| --- | ---: | ---: | ---: |
| TT-Lang grouped-row L1 | 2.269 (2.232-2.320) | 1.154 | 3/10 |
| Native `all_gather_minimal_matmul_async` | 1.965 (1.954-1.992) | 1.000 | 3/10 |

Both results passed PCC >= 0.99 and elementwise relative/absolute tolerances of
0.05 against FP32 PyTorch for every warmup and sample.

## Component measurements

These isolated measurements identify which component accounts for the fused
performance difference. They are not additive: the complete operation overlaps
activation movement with matmul, and each row retains the configuration stated
below.

| Measured operation | Devices | TT-Lang device median ms (min-max) | TT-Metal device median ms (min-max) | TT-Lang/TT-Metal | Warmups/samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standalone matmul, `M/K/N=9472/5120/3840` | 1 | 2.813 (2.806-2.824) | 3.003 (2.999-3.009) | 0.937 | 3/10 |
| TT-Lang activation all-gather and compute-grid distribution | 4 | 2.521 (2.511-2.575) | -- | -- | 1/3 |

The standalone comparison uses BF16 DRAM inputs and output, HiFi4, FP32
destination accumulation, and packer L1 accumulation. TT-Lang uses 120 workers,
a `12x10` grid, M/K/N blocks `8/8/5`, and pads M from 9472 to 10240; native
[`ttnn.matmul`](https://github.com/tenstorrent/tt-metal/tree/ea042c4ad6237678103cd7cbceb346e060f0f9a3/ttnn/cpp/ttnn/operations/matmul)
uses automatic program selection without that padding. Both standalone matmul
implementations measure first kernel start through final kernel end and pass
PCC >= 0.99 against FP32 PyTorch.

The activation-only measurement uses four communication workers, a `12x10`
compute grid, and four-tile M blocks. It runs the complete activation
all-gather and L1 distribution sequence, then discards each block after its
consumer wait. The composed operation validates the same data movement through
its matmul output. No isolated native activation result was measured.

## Configurations

Each implementation uses its selected grid, blocking, and communication
configuration. Equal resource use is not required.

| Setting | TT-Lang | Native |
| --- | --- | --- |
| Activation input/device | `9472 x 1280`, K-sharded; storage padded to `9600 x 1280` | `9472 x 1280`, K-sharded |
| Weight input/device | distinct `5120 x 3840` N shard | distinct `5120 x 3840` N shard |
| Bias input/device | distinct `1 x 3840` N shard | distinct `1 x 3840` N shard |
| Output/device | one distinct `9472 x 3840` N shard | three adjacent `9472 x 1280` chunks comprising one distinct `9472 x 3840` N shard |
| Compute grid | `12 x 10`; 120 compute workers | `12 x 9`; 108 compute workers |
| M/K/N blocks | `5/10/12` tiles | `7/5/16` tiles |
| Output subblock | `1 x 4` tiles; direct FP32 packer accumulation | `1 x 2` tiles |
| Communication workers | 4 fabric workers; each serves 3 compute rows and injects their activation subviews into separate 10-node compute chains | 24 compute workers are fabric clients; 4 mux-only workers |
| Activation collective | one-direction TT-Lang ring; each transfer contains three contiguous M blocks in L1 | bidirectional native ring into gathered-activation DRAM storage |
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
| Direct fabric-to-compute publication and one-block output DFB; four-tile M block | 3.189 (3.153-3.215) | 1/3 | control | Direct publication alone is statistically equivalent to the preceding four-worker result. |
| Same DFB configuration; five-tile M block with a partial final block | 2.631 (2.593-2.688) | 3/10 | -17.5% vs adjacent control | Accepted. Reduces M rounds from seven to five and padded M tiles from 336 to 300. |
| Group three compute rows per fabric transfer and inject DFB subviews | 2.306 (2.268-2.318) | 3/10 | -11.8% vs 2.613 ms adjacent control | Accepted. Reduces each communication worker's fabric transfers from 180 to 60 without changing payload bytes or matmul blocking. |
| Submit intermediate fabric packets without per-packet completion waits | 2.269 (2.232-2.320) | 3/10 | -1.6% vs 2.306 ms | Accepted. Retains blocking completion for the final write-and-atomic packet. |
| Alternate complete ten-tile K blocks across both ring directions | not measured | full-size compile | n/a | Rejected. The small four-device BF16 streaming case passed, but the full workload required 1,474,560 L1 bytes, 13,184 bytes over the 1,461,376-byte budget. |
| Eight direct fabric managers to reduce per-manager DFB capacity | not measured | full-size launch | n/a | Rejected. Compilation and PipeNet verification passed, but four physical forwarding links could not bind eight interfering managers. |
| Four bidirectional managers with two-block receive and relay DFBs | not measured | full-size launch | n/a | Rejected. The local relay filled while fabric sends waited for peers to post receives, producing the finite-capacity protocol deadlock reported in [#1037](https://github.com/tenstorrent/tt-lang/issues/1037). |
| Four bidirectional managers with one-row local staging and six-block receive and relay DFBs | not measured | full-size compile | n/a | Rejected. The small full-grid case passed; at full size a 491,520-byte communication DFB had only 313,600 bytes available. |
| Per-entry-pipe full-block bidirectional exchange with one receive and two relay blocks | not measured | full-size launch | n/a | Rejected. The small full-grid case passed and the full configuration fit L1, but the full workload did not complete within 60 seconds after compilation. |
| Direction-major consumption of per-entry-pipe full blocks | not measured | small correctness | n/a | Rejected. It compiled at the full 12x10 grid, but the four-device small case did not complete within 180 seconds. Grouping K blocks by direction did not resolve the protocol stall. |
| Per-row bidirectional exchange with one-block staging/receive and two-block relay DFBs | not measured | small correctness | n/a | Rejected. Four-device PCC was 0.257; a two-device diagnostic proved that remote K weights were paired with a repeated local activation shard. |
| Bidirectional exchange with five-tile K blocks | 7.002 (6.961-7.033) | 1/3 | +118.8% vs 3.200 ms one-direction control | Rejected. It passed full-size correctness but the doubled matmul and DFB granularity exceeded the benefit of the second fabric direction. |
| Bidirectional exchange with two K halves assembled into each ten-tile matmul block | 4.349 (4.323-4.362) | 1/3 | +36.7% vs 3.182 adjacent control | Rejected. It passed full-grid and full-size correctness but required eight row-segment L1 copies per activation block. |
| Unchanged four-worker implementation after K-half assembly experiment | 3.182 (3.161-3.204) | 1/3 | control | Adjacent control confirms the K-half assembly regression. |
| Bidirectional K halves received directly into ten-tile matmul-block subviews | 10.963 (10.879-10.969) | 1/3 | +242.6% vs 3.200 ms one-direction control | Rejected. Eight receive/forward transactions per activation block cost more than the eight eliminated L1 assembly copies. |
| Bidirectional K halves assembled at each compute-row head, then forwarded as one ten-tile block | 4.134 (4.127-4.143) | 1/3 | +29.7% vs 3.186 adjacent control | Rejected. Row-head-only assembly improved the 4.349 ms all-node assembly result by 4.9%, but remained slower than one-direction transport. |
| Unchanged four-worker implementation after row-head assembly experiment | 3.186 (3.177-3.213) | 1/3 | control | Adjacent control confirms the row-head assembly regression. |

In the multicast implementation, each compute-row head sent every activation
block to the other nine workers in that row. Point-to-point forwarding sent
each block once per adjacent worker. With identical ten-worker fabric and
compute configurations, point-to-point reduced activation-only time by 3.3%
and complete-operation time by 4.5%; multicast therefore underperformed and
was removed.

The selected implementation groups the three contiguous M blocks served by
each communication worker into one fabric transfer. It extracts three L1 DFB
subviews after reception and injects one into each compute-row chain. This
retains the five-tile M block and identical fabric payload bytes while reducing
each communication worker's fabric transfers from 180 to 60. The adjacent
direct-L1 control measured 2.613 ms (2.605-2.616), an 11.8% difference.

Each intermediate 8 KiB fabric packet is submitted without waiting for remote
completion; a local NoC flush protects packet-header reuse. The final fused
write-and-atomic packet remains blocking so the completion signal cannot
precede its payload. This reduced device time from 2.306 ms to 2.269 ms.

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

Measured 2026-09-13 20:02-20:04 UTC. TT-Lang parent `f52fd7a52ad5`, operation
SHA-256 `f6e57fa266f8`, routing helper SHA-256 `ff8df2b5d84b`, comparison runner
SHA-256 `7b2fe3eb0750`; TT-Metal `ea042c4ad623`; LLVM `37aca9d384347`; firmware
18.12.1; IRD v1.1.9.

[Raw device-profiler reports](https://gist.github.com/brnorris03/fa7ab25c12872de92dc0727f28f16104).
[Reproduction command and timing definition](README.md#run-the-comparison).
