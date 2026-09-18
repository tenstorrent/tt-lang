# All-gather matmul performance

This comparison measures column-parallel all-gather matmul with K-sharded
activation and N-sharded weight, bias, and output. The four output shards
collectively contain one M x N result.

## Four-device comparison

Blackhole P150b, four-device ring, BF16 input/output, FP32 destination
accumulation, three warmups, and ten measured samples. Times are device kernel
intervals reported as median (minimum-maximum). Every invocation passed PCC >=
0.99 and elementwise relative/absolute tolerances of 0.05 against FP32 PyTorch.

| M | Full K | Full N | TT-Lang ms | Native ms | TT-Lang/native | Provenance |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 9472 | 5120 | 15360 | 1.847 (1.802-1.879) | 1.970 (1.950-2.035) | 0.937 | [P4](#p4-four-device-column-parallel) |

The broader comparison uses the 156 AGMM/SAGMM rows in the
[pinned TT-Metal manifest](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/models/tt_dit/utils/sweep_mm_block_sizes.py#L140).
Native blocking is derived from the
[model configuration heuristic](https://github.com/tenstorrent/tt-metal/blob/0e9d200db976120c129ab0deb13aa3f6d972b723/models/tt_dit/layers/linear.py#L415),
not a dense local sweep. Additional paired results will be added to this table.

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
| Communication workers | 48 fabric clients in four compute rows; eight mux-only workers | 24 compute workers are fabric clients; four mux-only workers |
| Activation collective | bidirectional ring into L1; boundary rows multicast each K half through compute columns | bidirectional ring into DRAM, followed by unicast worker-chain distribution |
| Fabric configuration | 2D, strict initialization | 1D ring, strict initialization |
| Payload | 8192 bytes | 8192 bytes |
| Links/clients/buffers | four links/direction; six clients/link; 22 buffers/client channel | two links/direction; six clients/link; 24 buffers/client channel |
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

### P4: Four-device column-parallel

- Measured: 2026-09-15 09:34-09:35 UTC.
- TT-Lang source: 71b471e6c72c9fa67221230d87cb2094786d9302.
- Operation SHA-256: 6d7cfba54a5b37ca401ad03abd8865f1c113a9a8663e0219483eb35be630b793.
- TT-Lang compiler SHA-256: 6f4f849342e3064c03182172ed15d4d73539bb5a595b5b872d9f960d33288dc8.
- TT-Metal runtime: 41859079d93962001854fb9e0c466a693e8789e2.
- Native binary SHA-256: 9815624f3813041d7b322e09d9b220b1989726e49bbc97606843af21ff2927a8.
- libtt_metal SHA-256: 019f18c42901dfe9c9d47bb6194ee02f562027a6333d4b6384b116772cc71cdc.
- LLVM: 37aca9d384347f4f965fa137b0f5463156ba590f.
- Firmware 18.12.1; IRD v1.1.9. The container image digest was not recorded.

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

- Measured: 2026-09-16 05:26-05:50 UTC on bh-lb-120-a08u28.
- TT-Lang source: 1de0765a4eaf5d1f77f96264fcf8b4b9cfb353dc.
- Operation SHA-256: cd1877ea6787c28d16b02e1286d7d8a47b87e10ddf2645d6faeed78e2567cc88.
- TT-Lang compiler SHA-256: ddc32628bb6efa687c3b9d0e9864eccdb9d695b95cf5e7a2d47f319ed6c36462.
- TT-Metal runtime: 41859079d93962001854fb9e0c466a693e8789e2.
- Native binary SHA-256: 9815624f3813041d7b322e09d9b220b1989726e49bbc97606843af21ff2927a8.
- libtt_metal SHA-256: 019f18c42901dfe9c9d47bb6194ee02f562027a6333d4b6384b116772cc71cdc.
- LLVM: 37aca9d384347f4f965fa137b0f5463156ba590f.
- Firmware 19.8.1; IRD v1.1.9 image digest
  sha256:76489cf9e5fbe953ff1d057fcd359ec6e7e3a78360824d3f1ea6571734a7d669.
