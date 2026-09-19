# All-gather matmul performance

This comparison measures column-parallel all-gather matmul with K-sharded
activation and N-sharded weight, bias, and output. The four output shards
collectively contain one M x N result.

## All-shape four-device comparison

Measured 2026-09-19 01:29-16:24 UTC on an Exabox eight-chip host: four Blackhole P150b
devices (IDs 7, 3, 1, 5) in a physical ring, firmware 19.8.1, KMD 2.8.0,
1350 MHz. TT-Lang `07118bf88eb29070ca064ab382387f04c84ef409`; TT-Metal
`0e9d200db976120c129ab0deb13aa3f6d972b723`; IRD image
`ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-24-04:v1.1.10`
(`sha256:c64126d459afd77bf1b8c5cf68c5941c1561515d28b1b125d33fc57f950e09c7`).
Device time is the interval from first kernel start through last kernel end,
averaged across the four devices; three warmups and ten measured samples;
median (minimum-maximum) in ms. Every warmup and sample passed the dtype-aware
tolerance against FP32 PyTorch. Provenance
[PA](#pa-all-shape-four-device-comparison).

The inputs are the semantic rows of the
[pinned TT-Metal manifest](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/models/tt_dit/utils/sweep_mm_block_sizes.py#L140)
(full K and per-device N; full N is four times the per-device value). Each
device holds an M x K/4 activation shard and a distinct K x N/4 weight and
bias shard, and returns its distinct M x N/4 output shard. BF16 TILE inputs
and output, HiFi2, FP32 destination accumulation; TT-Lang uses its bidirectional
L1 activation transport under `FABRIC_2D`, native uses `FABRIC_1D_RING`. Each
implementation uses its own independently selected grid and blocks; the
configuration column lists TT-Lang, then native, as grid and M/K/N block tiles.
This host's boards expose a 12x10 Tensix worker grid, so TT-Lang rows use at
most 112 compute workers plus eight fabric-mux workers; the accepted 12x10
configuration below was measured on a 13x10 host.
A ratio is reported only for plain and QKV rows that both implementations
support. Rows with native-only fused epilogues show `--`. The selection method
is in [Optimization experiments](OPTIMIZATION_EXPERIMENTS.md#all-shape-configuration-search).

| Use case | M | Full K | Full N | TT-Lang ms | Native ms | TT-Lang/native | Configuration (TT-Lang; native) or status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| plain | 64 | 1536 | 36864 | 0.166 (0.163-0.170) | 0.215 (0.212-0.218) | 0.772 | 2x9 M1/K12/N16; 12x9 M2/K6/N16 sub 2x2, 1 chunk |
| plain | 64 | 6144 | 36864 | 0.615 (0.611-0.624) | 0.701 (0.694-0.708) | 0.877 | 2x9 M1/K16/N16; 12x9 M6/K4/N16 sub 2x2, 1 chunk |
| plain | 128 | 768 | 18432 | 0.066 (0.065-0.067) | 0.073 (0.072-0.077) | 0.903 | 4x9 M1/K6/N16; 12x9 M2/K3/N16 sub 2x2, 1 chunk |
| plain | 128 | 6144 | 18432 | 0.342 (0.335-0.348) | 0.357 (0.354-0.364) | 0.957 | 4x9 M1/K16/N16; 12x9 M2/K8/N8 sub 2x2, 1 chunk |
| plain | 512 | 1536 | 36864 | 0.231 (0.228-0.235) | 0.228 (0.226-0.231) | 1.010 | 8x9 M2/K12/N16; 12x9 M2/K6/N16 sub 2x2, 1 chunk |
| plain | 512 | 6144 | 36864 | 0.713 (0.699-0.729) | 0.749 (0.743-0.756) | 0.951 | 11x9 M2/K16/N16; 12x9 M4/K8/N8 sub 2x2, 1 chunk |
| plain | 576 | 1536 | 36864 | 0.233 (0.230-0.240) | 0.231 (0.229-0.233) | 1.008 | 9x9 M2/K12/N16; 12x9 M2/K6/N16 sub 2x2, 1 chunk |
| plain | 576 | 6144 | 36864 | 0.711 (0.699-0.721) | 0.752 (0.751-0.760) | 0.946 | 10x9 M2/K16/N16; 12x9 M2/K8/N8 sub 2x2, 1 chunk |
| plain | 1024 | 768 | 18432 | 0.138 (0.138-0.140) | 0.121 (0.118-0.125) | 1.146 | 11x9 M3/K6/N16; 12x9 M3/K3/N16 sub 1x4, 1 chunk |
| plain | 1024 | 6144 | 18432 | 0.508 (0.497-0.518) | 0.509 (0.505-0.515) | 0.997 | 12x9 M3/K12/N16; 12x8 M3/K6/N10 sub 3x1, 1 chunk |
| plain | 1152 | 768 | 18432 | 0.145 (0.144-0.146) | 0.132 (0.130-0.137) | 1.096 | 9x9 M4/K6/N16; 12x9 M3/K6/N8 sub 1x4, 1 chunk |
| plain | 1152 | 6144 | 18432 | 0.506 (0.493-0.521) | 0.489 (0.486-0.492) | 1.034 | 12x9 M3/K12/N16; 12x8 M3/K8/N10 sub 3x1, 1 chunk |
| plain | 2048 | 1536 | 18432 | 0.327 (0.322-0.374) | 0.294 (0.292-0.296) | 1.111 | 12x9 M6/K6/N16; 12x9 M6/K6/N16 sub 2x2, 1 chunk |
| plain | 2048 | 1536 | 36864 | 0.614 (0.605-0.657) | 0.458 (0.453-0.463) | 1.342 | 12x9 M6/K6/N16; 12x9 M6/K6/N16 sub 2x2, 1 chunk |
| plain | 2048 | 6144 | 36864 | 1.632 (1.612-1.665) | 1.290 (1.272-1.296) | 1.265 | 11x9 M6/K8/N16; 12x9 M6/K8/N12 sub 2x2, 1 chunk |
| plain | 2112 | 1536 | 36864 | 0.614 (0.600-0.648) | 0.461 (0.459-0.466) | 1.333 | 12x9 M3/K12/N16; 12x9 M6/K6/N16 sub 2x2, 1 chunk |
| plain | 2112 | 6144 | 36864 | 1.649 (1.609-1.660) | 1.329 (1.320-1.336) | 1.241 | 11x9 M6/K8/N16; 12x9 M8/K6/N12 sub 2x2, 1 chunk |
| plain | 3072 | 5120 | 5120 | 0.695 (0.687-0.734) | 0.637 (0.635-0.637) | 1.092 | 11x10 M9/K10/N4; 8x8 M12/K8/N8 sub 1x4, 1 chunk |
| plain | 3072 | 5120 | 15360 | 0.822 (0.818-0.867) | 0.833 (0.825-0.839) | 0.988 | 11x10 M9/K8/N12; 12x9 M8/K8/N12 sub 1x4, 1 chunk |
| plain | 4096 | 768 | 18432 | 0.470 (0.461-0.502) | 0.361 (0.358-0.380) | 1.302 | 12x9 M11/K6/N8; 12x9 M11/K3/N8 sub 1x4, 1 chunk |
| plain | 4096 | 6144 | 18432 | 2.462 (2.443-2.520) | 1.403 (1.388-1.406) | 1.756 | 12x9 M11/K8/N8; 12x9 M11/K8/N8 sub 1x4, 1 chunk |
| plain | 4224 | 768 | 18432 | 0.477 (0.465-0.495) | 0.374 (0.372-0.399) | 1.276 | 12x9 M11/K6/N8; 12x9 M11/K3/N8 sub 1x4, 1 chunk |
| plain | 4224 | 6144 | 18432 | 2.464 (2.437-2.508) | 1.415 (1.404-1.419) | 1.742 | 12x9 M11/K8/N8; 12x9 M11/K8/N8 sub 1x4, 1 chunk |
| plain | 4768 | 7168 | 5376 | 1.575 (1.567-1.582) | 1.147 (1.141-1.151) | 1.372 | 12x7 M13/K8/N6; 12x9 M8/K8/N6 sub 2x2, 1 chunk |
| plain | 8192 | 1536 | 18432 | 1.494 (1.476-1.543) | 0.896 (0.892-0.915) | 1.668 | 12x9 M11/K6/N8; 12x9 M11/K6/N8 sub 1x4, 1 chunk |
| plain | 8192 | 1536 | 36864 | 2.959 (2.835-3.045) | 1.498 (1.495-1.500) | 1.975 | 12x9 M11/K6/N8; 12x9 M11/K6/N8 sub 1x4, 1 chunk |
| plain | 8192 | 6144 | 36864 | 9.835 (9.764-9.886) | 4.582 (4.553-4.621) | 2.146 | 12x9 M11/K8/N8; 12x9 M8/K6/N12 sub 2x2, 1 chunk |
| plain | 8256 | 1536 | 36864 | 2.976 (2.906-3.045) | 1.501 (1.498-1.511) | 1.982 | 12x9 M11/K6/N8; 12x9 M11/K6/N8 sub 1x4, 1 chunk |
| plain | 8256 | 6144 | 36864 | 9.828 (9.733-9.881) | 4.612 (4.597-4.641) | 2.131 | 12x9 M11/K8/N8; 12x9 M8/K6/N12 sub 2x2, 1 chunk |
| plain | 16384 | 768 | 18432 | 1.793 (1.772-1.867) | 1.174 (1.134-1.202) | 1.527 | 12x9 M11/K6/N8; 12x9 M4/K6/N16 sub 2x2, 1 chunk |
| plain | 16384 | 6144 | 18432 | 6.327 (6.312-6.385) | 5.512 (5.496-5.536) | 1.148 | 12x9 M4/K12/N16; 12x9 M10/K6/N10 sub 2x2, 1 chunk |
| plain | 16512 | 768 | 18432 | 1.790 (1.698-1.843) | 1.172 (1.117-1.205) | 1.528 | 12x9 M11/K6/N8; 12x9 M4/K6/N16 sub 2x2, 1 chunk |
| plain | 16512 | 6144 | 18432 | 6.355 (6.288-6.409) | 5.528 (5.505-5.558) | 1.150 | 12x9 M4/K12/N16; 12x9 M10/K6/N10 sub 2x2, 1 chunk |
| QKV | 64 | 6144 | 18432 | 0.315 (0.307-0.319) | 0.356 (0.353-0.361) | 0.885 | 2x9 M1/K16/N16; 12x9 M4/K4/N16 sub 2x2, 3 chunks |
| QKV | 128 | 6144 | 9216 | 0.187 (0.180-0.190) | 0.195 (0.191-0.197) | 0.960 | 4x9 M1/K24/N8; 12x9 M2/K4/N8 sub 2x2, 3 chunks |
| QKV | 512 | 6144 | 18432 | 0.372 (0.367-0.382) | 0.424 (0.418-0.428) | 0.878 | 10x9 M2/K16/N16; 12x9 M8/K8/N8 sub 2x2, 3 chunks |
| QKV | 1024 | 6144 | 9216 | 0.371 (0.362-0.377) | 0.303 (0.299-0.309) | 1.221 | 11x9 M3/K24/N8; 12x8 M3/K8/N10 sub 3x1, 3 chunks |
| QKV | 1152 | 6144 | 9216 | 0.396 (0.389-0.405) | 0.312 (0.307-0.316) | 1.269 | 12x9 M3/K24/N8; 12x8 M3/K8/N10 sub 3x1, 3 chunks |
| QKV | 2048 | 6144 | 18432 | 0.842 (0.825-0.884) | 0.816 (0.806-0.824) | 1.032 | 11x9 M6/K8/N16; 12x8 M6/K8/N16 sub 1x4, 3 chunks |
| QKV | 4096 | 768 | 9216 | 0.263 (0.260-0.305) | 0.231 (0.228-0.252) | 1.140 | 12x9 M11/K6/N8; 12x9 M11/K3/N8 sub 1x4, 3 chunks |
| QKV | 4096 | 6144 | 9216 | 1.246 (1.234-1.284) | 0.941 (0.924-0.943) | 1.325 | 12x9 M11/K8/N8; 12x9 M11/K8/N8 sub 1x4, 3 chunks |
| QKV | 4768 | 5376 | 21504 | 3.785 (3.747-3.821) | 1.744 (1.730-1.762) | 2.170 | 12x8 M13/K6/N7; 12x9 M8/K7/N12 sub 2x2, 3 chunks |
| QKV | 8192 | 6144 | 18432 | 4.918 (4.872-4.952) | 3.033 (2.961-3.073) | 1.622 | 12x9 M11/K8/N8; 12x9 M6/K6/N16 sub 2x2, 3 chunks |
| QKV | 9472 | 5120 | 15360 | 2.373 (2.344-2.429) | 2.473 (2.444-2.515) | 0.960 | 11x10 M9/K8/N12; 12x9 M7/K5/N16 sub 1x2, 3 chunks |
| QKV | 16384 | 768 | 9216 | 0.929 (0.898-0.951) | 0.775 (0.749-0.784) | 1.198 | 12x9 M9/K6/N8; 12x9 M4/K6/N8 sub 2x2, 3 chunks |
| QKV | 16384 | 6144 | 9216 | 4.899 (4.874-4.985) | 3.838 (3.814-3.855) | 1.276 | 12x9 M11/K8/N8; 12x9 M12/K6/N8 sub 2x2, 3 chunks |
| to_out | 64 | 6144 | 6144 | -- | 0.142 (0.140-0.144) | -- | native-only fused epilogue; 12x9 M2/K6/N6 sub 2x2, 1 chunk |
| to_out | 128 | 6144 | 3072 | -- | 0.098 (0.093-0.105) | -- | native-only fused epilogue; 12x9 M2/K12/N6 sub 2x2, 1 chunk |
| to_out | 512 | 6144 | 6144 | -- | 0.210 (0.208-0.211) | -- | native-only fused epilogue; 12x8 M2/K8/N8 sub 1x4, 1 chunk |
| to_out | 1024 | 6144 | 3072 | -- | 0.240 (0.238-0.241) | -- | native-only fused epilogue; 12x9 M3/K8/N6 sub 1x3, 1 chunk |
| to_out | 1024 | 24576 | 3072 | -- | 0.890 (0.883-0.895) | -- | native-only fused epilogue; 12x9 M3/K8/N4 sub 1x4, 1 chunk |
| to_out | 1152 | 24576 | 3072 | -- | 0.911 (0.905-0.916) | -- | native-only fused epilogue; 12x9 M3/K8/N4 sub 1x4, 1 chunk |
| to_out | 2048 | 6144 | 6144 | -- | 0.494 (0.492-0.497) | -- | native-only fused epilogue; 12x8 M6/K8/N8 sub 1x4, 1 chunk |
| to_out | 4096 | 6144 | 3072 | -- | 0.786 (0.784-0.788) | -- | native-only fused epilogue; 12x7 M11/K8/N4 sub 1x4, 1 chunk |
| to_out | 8192 | 6144 | 6144 | -- | 1.818 (1.810-1.823) | -- | native-only fused epilogue; 12x9 M8/K12/N6 sub 2x2, 1 chunk |
| to_out | 9472 | 5120 | 5120 | -- | 1.630 (1.622-1.634) | -- | native-only fused epilogue; 12x9 M10/K8/N8 sub 2x1, 1 chunk |
| to_out | 16384 | 6144 | 3072 | -- | 3.031 (3.026-3.040) | -- | native-only fused epilogue; 12x9 M16/K8/N3 sub 4x1, 1 chunk |
| FF1 GELU | 9472 | 5120 | 13824 | -- | 2.173 (2.161-2.185) | -- | native-only fused epilogue; 12x9 M9/K5/N12 sub 1x2, 1 chunk |
| plain GELU | 3072 | 5120 | 13824 | -- | 0.843 (0.840-0.847) | -- | native-only fused epilogue; 12x9 M8/K8/N12 sub 1x4, 1 chunk |
| FF1 SwiGLU | 128 | 6144 | 18432 | -- | 0.371 (0.368-0.375) | -- | native-only fused epilogue; 12x9 M2/K8/N8 sub 2x2, 1 chunk |
| FF1 SwiGLU | 1024 | 6144 | 18432 | -- | 0.549 (0.541-0.554) | -- | native-only fused epilogue; 12x9 M3/K8/N8 sub 1x4, 1 chunk |
| FF1 SwiGLU | 1152 | 6144 | 18432 | -- | 0.530 (0.528-0.535) | -- | native-only fused epilogue; 12x8 M3/K8/N10 sub 3x1, 1 chunk |
| FF1 SwiGLU | 4768 | 5376 | 28672 | -- | 2.453 (2.447-2.488) | -- | native-only fused epilogue; 12x9 M8/K3/N14 sub 2x2, 1 chunk |

All 46 comparable inputs are paired. TT-Lang is faster on 12 of them; the geometric mean of TT-Lang/native is 1.23, ranging from 0.77 to 2.17. The largest ratios are inputs with many output blocks per core; on this host they do not respond to the output-buffer change that helps on the 13x10 host, and their decomposition is recorded in [Optimization experiments](OPTIMIZATION_EXPERIMENTS.md#where-tt-lang-loses-on-the-all-shape-inputs). The table was measured with the one-block output DFB; the operation now defaults to two blocks (`output_block_count`).

## Accepted 9472/5120/15360 result

Blackhole P150b, four-device ring, BF16 input/output, FP32 destination
accumulation, three warmups, and ten measured samples. Times are device kernel
intervals reported as median (minimum-maximum). Every invocation passed PCC >=
0.99 and elementwise relative/absolute tolerances of 0.05 against FP32 PyTorch.

| M | Full K | Full N | TT-Lang ms | Native ms | TT-Lang/native | Provenance |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 9472 | 5120 | 15360 | 1.800 (1.776-1.820) | 1.980 (1.951-1.996) | 0.909 | [P4](#p4-four-device-column-parallel) |

Native blocking for this result is the upstream
[model configuration heuristic](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/models/tt_dit/layers/linear.py#L415).

## Configurations

Each implementation uses its independently selected grid, blocking, and
communication configuration for the same inputs and N-sharded output.

| Setting | TT-Lang | Native |
| --- | --- | --- |
| Activation/device | 9472 x 1280, K-sharded; storage padded to 9600 x 1280 | 9472 x 1280, K-sharded |
| Weight/device | distinct 5120 x 3840 N shard | distinct 5120 x 3840 N shard |
| Bias/device | distinct 1 x 3840 N shard | distinct 1 x 3840 N shard |
| Output/device | one distinct 9472 x 3840 N shard | three adjacent 9472 x 1280 chunks comprising the same N shard |
| Compute grid | 12 x 10; 120 compute workers | 12 x 9; 108 compute workers |
| M/K/N blocks | 5/10/12 tiles | 7/5/16 tiles |
| Output subblock | 1 x 4 tiles; direct FP32 packer accumulation | 1 x 2 tiles |
| Output blocks in L1 | two blocks (`output_block_count`) | output written per subblock during compute |
| Communication workers | 48 fabric clients in four compute rows; eight mux-only workers | 24 compute workers are fabric clients; four mux-only workers |
| Activation collective | bidirectional ring into L1; boundary rows multicast each K half through compute columns | bidirectional ring into DRAM, followed by unicast worker-chain distribution |
| Fabric configuration | 2D, strict initialization | 1D ring, strict initialization |
| Payload | 8192 bytes | 8192 bytes |
| Links/clients/buffers | four links/direction; six clients/link; 21 buffers/client channel | two links/direction; six clients/link; 24 buffers/client channel |
| Arithmetic | BF16 input/output; HiFi2; FP32 destination and packer accumulation | same; approximate math; three output chunks |

The native grid and blocks are the TT-Metal
[grid_12_9_configs entry](https://github.com/tenstorrent/tt-metal/blob/f8c4ce59dd04a3eeeb11abf01ffc9dbce0059eba/models/tt_dit/utils/matmul.py#L224-L247)
for per-device M/K/N = 9472/1280/3840. Host concatenation of the three native
chunks is excluded from device time and reconstructs the N shard returned
directly by TT-Lang.

## Component measurements

These measurements are not additive because the complete operation overlaps
activation movement and matmul.

| Operation | Devices | TT-Lang ms | TT-Metal ms | Ratio | Warmups/samples | Provenance |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Matmul, M/K/N = 9472/5120/3840 | 1 | 2.813 (2.806-2.824) | 3.003 (2.999-3.009) | 0.937 | 3/10 | [PC](#pc-component-measurements) |
| TT-Lang activation all-gather and compute-grid distribution | 4 | 2.521 (2.511-2.575) | -- | -- | 1/3 | [PC](#pc-component-measurements) |

Standalone matmul uses BF16 DRAM inputs and output, HiFi4, FP32 destination
accumulation, and packer L1 accumulation. TT-Lang uses 120 workers, a 12 x 10
grid, M/K/N blocks 8/8/5, and pads M to 10240. Native
[ttnn.matmul](https://github.com/tenstorrent/tt-metal/tree/ea042c4ad6237678103cd7cbceb346e060f0f9a3/ttnn/cpp/ttnn/operations/matmul)
uses automatic program selection without that padding.

The activation-only run uses four communication workers, a 12 x 10 compute
grid, and four-tile M blocks. It executes the complete activation all-gather
and L1 distribution sequence, then discards each block after its consumer wait.

## Complete-operation decomposition

The rows are mutually exclusive and sum to the four-device medians. DFB waits
are aggregate counters on the operation-ending unpack thread. TT-Lang
matmul/control is the mean of independent activation-counter and weight-counter
runs; their half-spread is 9.591 us.

| Component | TT-Lang us | Native us | Difference us |
| --- | ---: | ---: | ---: |
| Activation DFB wait | 223.876 | 376.137 | -152.261 |
| Weight DFB wait | 105.233 | 90.951 | +14.282 |
| Matmul and control | 1494.671 | 1488.576 | +6.095 |
| Compute-thread alignment | 1.149 | 0.223 | +0.926 |
| Post-compute data movement | 6.259 | 9.579 | -3.320 |
| Start and cross-run residual | 15.783 | 5.028 | +10.755 |
| **Complete operation** | **1846.973** | **1970.495** | **-123.522** |

## Retained results

### Two-dimensional decomposition

The 2 x 2 operation partitions K across two device groups and N across two
device groups. Each device returns one M/2 x N/2 shard. It is not compared
with the column-parallel result because the input and output placement differ.

| Schedule | Device ms | Change | Warmups/samples | Provenance |
| --- | ---: | ---: | ---: | --- |
| Reduce each output immediately | 3.531 (3.523-3.536) | control | 3/10 | [P2D](#p2d-two-dimensional-decomposition) |
| Compute the next outgoing partial before reducing the preceding output | 3.387 (3.354-3.455) | -4.08% | 3/10 | [P2D](#p2d-two-dimensional-decomposition) |
| Write the preceding output on the partial-exchange thread | 3.233 (3.211-3.256) | -8.45% | 3/10 | [P2D](#p2d-two-dimensional-decomposition) |

An unchanged-source confirmation measured 3.240 ms (3.222-3.247). All runs use
global M/K/N = 9472/5120/15360, an 11 x 10 compute grid, M/K/N blocks 7/10/8,
BF16 input/output, HiFi2, and FP32 destination accumulation.

### Eight-device comparison

This earlier scaling measurement is retained for reference; the current
comparison scope is four devices.

| M | Full K | Full N | TT-Lang ms | Native ms | Ratio | Warmups/samples | Provenance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 9472 | 5120 | 15360 | 1.798 (1.784-1.852) | 1.709 (1.690-1.722) | 1.052 | 3/10 | [P8](#p8-eight-device-column-parallel) |

TT-Lang used an 11 x 10 grid and M/K/N blocks 9/10/6. Native used a 12 x 9
grid and M/K/N blocks 7/10/8. Both returned the same N-sharded output.

The complete optimization record, including accepted and rejected experiments,
is in [Optimization experiments](OPTIMIZATION_EXPERIMENTS.md).

## Measurement and provenance

Device time is the interval from first kernel start through final kernel end,
averaged across participating devices. Host tensor creation, compilation,
dispatch, correctness checks, profiler processing, and native output
concatenation are excluded. Full raw reports are in the
[device-profiler archive](https://gist.github.com/brnorris03/fa7ab25c12872de92dc0727f28f16104).
See the [reproduction commands](README.md#run).

### PA: All-shape four-device comparison

- Measured: 2026-09-19 01:29-16:24 UTC on an Exabox eight-chip host; devices 7, 3, 1, 5.
- TT-Lang source: 07118bf88eb29070ca064ab382387f04c84ef409, clean worktree.
- TT-Metal source and runtime: 0e9d200db976120c129ab0deb13aa3f6d972b723.
- IRD v1.1.10 image digest
  sha256:c64126d459afd77bf1b8c5cf68c5941c1561515d28b1b125d33fc57f950e09c7.
- Firmware 19.8.1; KMD 2.8.0; 1350 MHz; timing scope
  first_kernel_start_to_last_kernel_end; three warmups and ten samples.
- Summaries and selected configurations: [all-shape archive](https://gist.github.com/brnorris03/08c4d1151b56d773d3cb0a54cdf130f1); raw reports are retained outside Git.

### P4: Four-device column-parallel

- Measured: 2026-09-19 22:26-22:31 UTC.
- TT-Lang source: b0eb413d78c27efdb1f3c560bc681c5460246b88, clean worktree.
- Operation SHA-256: 4e2e0fdcb1d30a35deaa6117e2cc1e8b50a681fdcfca06de34ca5a7925b7980a.
- TT-Lang compiler SHA-256: 8c5ac733d8395705c95883845447625bd6c8b92f5c9eec006994e7d291689a79.
- TT-Metal runtime: 0e9d200db976120c129ab0deb13aa3f6d972b723.
- Native binary SHA-256: 4efa4f824c0d90d0bc15ed0e8eea714b66f62d1b3be0df1fc5754b0e4c5adf80.
- libtt_metal SHA-256: cd74b5c1468eaaad72688cf3bc78ac7db2408add2ee4dc2a0aef4a293ee7eae4.
- LLVM: d6b0c1b9a79d5fdd3d6f94ae36adb266930bd68e.
- Firmware 18.12.1; 1350 MHz; IRD v1.1.10 image digest
  sha256:e153267665bf1d1f48577698979b2b166f78810c74a2a49a083d37bf18249b52.
- The previous accepted pair, 1.847 and 1.970 ms at 71b471e6c72c9fa67221230d87cb2094786d9302 on 2026-09-15,
  preceded the two-block output DFB and the early activation relay.

### PC: Component measurements

- Measured: 2026-09-11 through 2026-09-12 UTC.
- TT-Lang source: 99f10aae99d2af16c123fbf7fb05feaebcd3cce5.
- Operation SHA-256: 6dad67c4b3e2bb9e22992035f2a6dfe1455d9d4a6aaf3bd418808b59005b68fe.
- Comparison runner SHA-256: 4055a010ff271900706eb7a7b67f2d912d68a3e05330494472e99e8a4c386f58.
- TT-Lang compiler SHA-256: 9943bc49ea2355fe478917a7572b02f9e9e8e0e4d07babb430ecab7df3578e3b.
- TT-Metal: ea042c4ad6237678103cd7cbceb346e060f0f9a3.
- LLVM: 37aca9d384347f4f965fa137b0f5463156ba590f.
- Firmware 18.12.1; IRD v1.1.9. The container image digest was not recorded.

### P2D: Two-dimensional decomposition

- Measured: 2026-09-15 15:41-16:44 UTC.
- TT-Lang source: eff91878110147c5787c1618d70dd7080c1f8986.
- Operation SHA-256: 07a577b95bc3041ef6db3e6cd9fd1c01162d1b7b12e2f99dd7466458f7c4fee3.
- TT-Lang compiler SHA-256: 6f4f849342e3064c03182172ed15d4d73539bb5a595b5b872d9f960d33288dc8.
- TT-Metal: 41859079d93962001854fb9e0c466a693e8789e2.
- LLVM: 37aca9d384347f4f965fa137b0f5463156ba590f.
- Firmware 18.12.1; IRD v1.1.9.

### P8: Eight-device column-parallel

- Measured: 2026-09-16 05:26-05:50 UTC on a second Exabox eight-chip host.
- TT-Lang source: 1de0765a4eaf5d1f77f96264fcf8b4b9cfb353dc.
- Operation SHA-256: cd1877ea6787c28d16b02e1286d7d8a47b87e10ddf2645d6faeed78e2567cc88.
- TT-Lang compiler SHA-256: ddc32628bb6efa687c3b9d0e9864eccdb9d695b95cf5e7a2d47f319ed6c36462.
- TT-Metal runtime: 41859079d93962001854fb9e0c466a693e8789e2.
- Native binary SHA-256: 9815624f3813041d7b322e09d9b220b1989726e49bbc97606843af21ff2927a8.
- libtt_metal SHA-256: 019f18c42901dfe9c9d47bb6194ee02f562027a6333d4b6384b116772cc71cdc.
- LLVM: 37aca9d384347f4f965fa137b0f5463156ba590f.
- Firmware 19.8.1; IRD v1.1.9 image digest
  sha256:76489cf9e5fbe953ff1d057fcd359ec6e7e3a78360824d3f1ea6571734a7d669.
