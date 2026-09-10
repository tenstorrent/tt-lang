# All-gather matmul performance

Four Blackhole P150b devices; global M=3072 and K=5120. Both implementations
return the same replicated M x N result. Inputs/output are BF16 TILE tensors
in interleaved DRAM; matmul uses HiFi2, FP32 destinations and packer accumulation,
with row bias.

Device trace replay, three warmups and five samples. Each sample is the mean
across four devices. TT-Lang includes both matmul and output all-gather, from
the first kernel start to the final kernel end on each device. Native uses its
fused program. Host preparation and synchronization are outside the interval.
Parentheses show sample minimum and maximum.

| Global N | N-sharded + gather ms (range) | Replicated weights ms (range) | Native ms (range) | N-sharded / native | Replicated / native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1280 | 1.608 (1.607-1.611) | 3.516 (3.488-3.526) | 0.356 (0.353-0.360) | 4.515 | 9.869 |
| 3840 | 2.582 (2.576-2.589) | 5.822 (5.758-5.856) | 0.692 (0.686-0.693) | 3.732 | 8.416 |

| Parameter | TT-Lang N=1280 | TT-Lang N=3840 | Native, both N values |
| --- | --- | --- | --- |
| Compute grid per device | 2x10 transposed, 20 workers | 2x10 transposed, 20 workers | 12x9 transposed, 108 workers |
| M/K/N block, tiles | 8/10/1 | 4/10/3 | 8/8/8; 2x2 subblock |
| Activation/output all-gather | Direct/direct | Direct/direct | Native bidirectional ring |
| Output-gather message, row x column tiles | 4x10 | 4x30 | Fused operation |
| Fabric | 2D, strict initialization | 2D, strict initialization | 1D ring, strict initialization |
| Router payload | 8192 bytes | 8192 bytes | 8192 bytes |
| Communication workers | 2 | 2 | 6 per link, 2 links, 24 channel buffers |

Replicated-weight TT-Lang uses the same 2x10 grid and 2D fabric, full-K L1 reuse,
and M/K/N blocks 2/10/2 or 2/10/6 for N=1280 or 3840. Activation all-gather is
direct for N=1280 and ring for N=3840.

Measured 2026-09-10 11:38:33-11:40:42 UTC; TT-Lang `85da7e527797` (clean); TT-Metal/LLVM source pins `ea042c4ad623`/`37aca9d384347`; CAPI/TTNN/Metal SHA-256 prefixes `20a74e405369`/`62edde2b1f61`/`65380f11dc15`; IRD v1.1.9 image digest `6eaf96b4b00d`.

Every measured output replica passes BF16 checks against FP32 PyTorch:
Pearson correlation >= 0.99 and elementwise relative/absolute tolerances of 0.05.

These are measured configurations, not completed tuning sweeps. TT-Lang is
slower in both cases; the worker-grid and fabric differences prevent attributing
the entire difference to matmul code generation. The native binaries come from
the pinned v1.1.9 image; their exact TT-Metal build commit was not recorded.

The README provides [reproduction commands and full binary identities](README.md#reproduce-the-measurements),
the [timing definition](README.md#measurement-contract), and
[configuration differences from the upstream benchmark](README.md#comparison-with-the-native-benchmark).
