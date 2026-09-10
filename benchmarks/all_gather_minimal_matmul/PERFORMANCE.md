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
| 1280 | 2.742 (2.738-2.743) | 3.846 (3.846-3.849) | 0.358 (0.357-0.360) | 7.660 | 10.744 |
| 3840 | 3.426 (3.423-3.429) | 6.238 (6.226-6.250) | 0.688 (0.685-0.691) | 4.978 | 9.066 |

| Parameter | TT-Lang N=1280 | TT-Lang N=3840 | Native, both N values |
| --- | --- | --- | --- |
| Compute grid per device | 2x10 transposed, 20 workers | 2x10 transposed, 20 workers | 12x9 transposed, 108 workers |
| M/K/N block, tiles | 4/8/1 | 4/10/3 | 8/8/8; 2x2 subblock |
| Activation/output all-gather | Direct/direct | Ring/ring | Native bidirectional ring |
| Output-gather message, tiles | 10 | 30 | Fused operation |
| Fabric | 2D, strict initialization | 2D, strict initialization | 1D ring, strict initialization |
| Router payload | 8192 bytes | 8192 bytes | 8192 bytes |
| Communication workers | 2 | 2 | 6 per link, 2 links, 24 channel buffers |

Replicated-weight TT-Lang uses the same 2x10 grid and 2D fabric, ring activation
all-gather, full-K L1 reuse, and M/K/N blocks 2/10/2 or 2/10/6 for N=1280 or 3840.

Measured 2026-09-10 UTC: native 09:55:09-09:55:36; TT-Lang 10:18:11-10:19:22. TT-Lang `3da13fd78158` plus working-tree changes; TT-Metal/LLVM source pins `ea042c4ad623`/`37aca9d384347`; TT-Lang CAPI/TTNN/Metal binary SHA-256 prefixes `be08c3e387c69`/`62edde2b1f61`/`65380f11dc15`; IRD v1.1.9.

Every measured output replica passes BF16 checks against FP32 PyTorch:
Pearson correlation >= 0.99 and elementwise relative/absolute tolerances of 0.05.

These are measured configurations, not completed tuning sweeps. TT-Lang is
slower in both cases; the worker-grid and fabric differences prevent attributing
the entire difference to matmul code generation. The native binaries come from
the pinned v1.1.9 image; their exact TT-Metal build commit was not recorded.

The README provides [reproduction commands and full binary identities](README.md#reproduce-the-measurements),
the [timing definition](README.md#measurement-contract), and
[configuration differences from the upstream benchmark](README.md#comparison-with-the-native-benchmark).
