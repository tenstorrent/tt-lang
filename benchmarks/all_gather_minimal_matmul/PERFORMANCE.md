# All-gather matmul performance

Four Blackhole P150b devices; global M/K/N=9472/5120/15360. All versions
return replicated M x N output, including row bias. Inputs/output are BF16
TILE tensors in interleaved DRAM; matmul uses HiFi2 and FP32 destination/packer
accumulation. Configurations are tuned independently; these are the best
measured configurations, not proven optima.

| Implementation | Compute workers/device | Device median ms (min-max) | / Native | Warmups/samples |
| --- | ---: | ---: | ---: | ---: |
| TT-Lang N-sharded + output gather | 60 | 24.156 (23.011-24.184) | 3.510 | 2/5 |
| TT-Lang replicated weights | 130 | 10.343 (10.242-10.688) | 1.503 | 3/5 |
| Native replicated weights | 108 | 6.883 (6.873-6.889) | 1.000 | 2/3 |

Device trace replay measures first kernel start through final kernel end,
including both programs and their gap for TT-Lang; native uses one fused
program. Each sample averages the four device intervals. Host preparation,
dispatch and correctness checks are excluded. Every output replica passes
PCC >= 0.99 and elementwise relative/absolute tolerances of 0.05 against FP32
PyTorch.

| Configuration | TT-Lang N-sharded + gather | TT-Lang replicated | Native |
| --- | --- | --- | --- |
| Transposed compute grid | 6x10 | 13x10 | 12x9 |
| M/K/N blocks, tiles | 2/8/4 | 8/8/8 | 8/8/8; 2x2 subblock |
| Activation storage | Full-K L1 cache | Gathered DRAM tensor; streaming matmul | Gathered DRAM tensor |
| Activation collective | Ring, two workers | Direct, two workers; 2x40-tile messages | Bidirectional ring |
| Final output collective | Direct, two workers; 2x30-tile messages | Not required | Not required |
| Fabric | 2D | 2D | 1D ring |
| Fabric initialization/payload | Strict / 8192 bytes | Strict / 8192 bytes | Strict / 8192 bytes |
| Native link configuration | Not applicable | Not applicable | Two links; six workers/link; 24 channel buffers |

Measured 2026-09-10 UTC: native 15:37:07 (`ff72bcb06859`), N-sharded 18:17:56 (`4c1a5677da6c` plus operation-file split), replicated 17:42:29 (`cc5010e45a46` plus changes committed as `4c1a5677d`). TT-Metal/LLVM source pins: `ea042c4ad623`/`37aca9d384347`; CAPI/TTNN/Metal SHA-256 prefixes: `20a74e405369`/`62edde2b1f61`/`65380f11dc15`; IRD v1.1.9 image digest: `6eaf96b4b00d`. [Raw reports](https://gist.github.com/brnorris03/da754d5cef08ed989cc241b023fdaccb) record dirty-tree state and source hashes. The installed native binary's exact build commit is unavailable.

Replicated TT-Lang remains 50.3% slower than native. Its activation gather
completes before matmul starts; native overlaps activation communication with
matmul. N-sharded TT-Lang additionally transfers the complete output across
devices. These end-to-end timings do not isolate matmul code generation.

[Run commands](README.md#run-the-comparison),
[timing definition](README.md#measurement-contract), and
[native benchmark references](README.md#comparison-with-the-native-benchmark).
