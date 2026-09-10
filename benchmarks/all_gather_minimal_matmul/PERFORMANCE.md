# All-gather matmul performance

**These committed four-device results use only 20 TT-Lang compute workers per device (2x10), versus native's 108 (12x9). Full-device TT-Lang utilization is work in progress. These results do not establish performance parity with native.**

Four Blackhole P150b devices; global M=3072 and K=5120. Both implementations
return the same replicated M x N result. Inputs/output are BF16 TILE tensors
in interleaved DRAM; matmul uses HiFi2, FP32 destinations and packer accumulation,
with row bias.

Device trace replay, three warmups and five samples (ten for replicated-weight N=1280). Each sample is the mean
across four devices. N-sharded TT-Lang includes matmul and output all-gather,
from the first kernel start to the final kernel end on each device. Replicated-weight
TT-Lang and native each use one fused program. Host preparation and synchronization are outside the interval.
Parentheses show sample minimum and maximum.

| Global N | TT-Lang N-sharded + gather, 20 workers/device, ms (range) | TT-Lang replicated weights, 20 workers/device, ms (range) | Native, 108 workers/device, ms (range) | N-sharded / native | Replicated / native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1280 | 1.608 (1.607-1.611) | 1.926 (1.920-1.937) | 0.356 (0.353-0.360) | 4.515 | 5.405 |
| 3840 | 2.582 (2.576-2.589) | 5.822 (5.758-5.856) | 0.692 (0.686-0.693) | 3.732 | 8.416 |

Per-program medians from the same N-sharded traces; the device-clock gap between programs is below 1 us.

| Global N | All-gather + matmul + bias ms | Output all-gather ms |
| --- | ---: | ---: |
| 1280 | 1.263 | 0.344 |
| 3840 | 1.892 | 0.688 |

| Parameter | TT-Lang N=1280 | TT-Lang N=3840 | Native, both N values |
| --- | --- | --- | --- |
| Compute grid per device | 2x10 transposed, 20 workers | 2x10 transposed, 20 workers | 12x9 transposed, 108 workers |
| M/K/N block, tiles | 8/10/1 | 4/10/3 | 8/8/8; 2x2 subblock |
| Activation/output all-gather | Direct/direct | Direct/direct | Native bidirectional ring |
| Output-gather message, row x column tiles | 4x10 | 4x30 | Fused operation |
| Fabric | 2D, strict initialization | 2D, strict initialization | 1D ring, strict initialization |
| Router payload | 8192 bytes | 8192 bytes | 8192 bytes |
| Communication workers | 2 | 2 | 6 per link, 2 links, 24 channel buffers |

Replicated-weight TT-Lang uses the same 2x10 grid and 2D fabric. N=1280 uses
6/8/4 blocks, streaming and direct all-gather. N=3840 uses 2/10/6 blocks,
full-K L1 reuse and ring all-gather.

Replicated N=1280 measured 2026-09-10 12:57:31 UTC at TT-Lang `906716f37`; other results measured 2026-09-10 11:38:33-11:40:42 UTC at `85da7e527797`. Shared TT-Metal/LLVM source pins `ea042c4ad623`/`37aca9d384347`; CAPI/TTNN/Metal SHA-256 prefixes `20a74e405369`/`62edde2b1f61`/`65380f11dc15`; IRD v1.1.9 image digest `6eaf96b4b00d`.

Every measured output replica passes BF16 checks against FP32 PyTorch:
Pearson correlation >= 0.99 and elementwise relative/absolute tolerances of 0.05.

N=1280 sweeps tested 294 native and 137 TT-Lang replicated-weight block/cache
configurations on their respective grids; N=3840 tuning remains incomplete. TT-Lang is
slower in both cases; the worker-grid and fabric differences prevent attributing
the entire difference to matmul code generation. The native binaries come from
the pinned v1.1.9 image; their exact TT-Metal build commit was not recorded.

The README provides [reproduction commands and full binary identities](README.md#reproduce-the-measurements),
the [timing definition](README.md#measurement-contract), and
[configuration differences from the upstream benchmark](README.md#comparison-with-the-native-benchmark).
