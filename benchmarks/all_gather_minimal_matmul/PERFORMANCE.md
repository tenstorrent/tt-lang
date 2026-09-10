# All-gather matmul performance

Two Blackhole devices in a 2x1 participant mesh; M=3072, full K=5120;
transposed 2x5 compute grid per device; M/K/N blocks of 2/40/2 tiles.
Both implementations use BF16 inputs/output, HiFi2 math, FP32 destinations
and intermediate storage, tile layout, interleaved DRAM inputs/output, and bias.

Times are device-kernel milliseconds, not host latency. Each sample is the
mean duration across the two devices, using TT-Metal's
`device_kernel_duration` analysis. Five samples follow three warmups, with
ordinary launches rather than trace replay. Parentheses show minimum and
maximum samples, not confidence intervals. Ratios below 1 favor TT-Lang.

| Per-device N | TT-Lang median ms (range) | Native median ms (range) | TT-Lang / native |
| --- | ---: | ---: | ---: |
| 1280 | 3.439 (3.434-3.448) | 3.690 (3.687-3.699) | 0.932 |
| 3840 | 9.040 (8.959-9.170) | 10.396 (10.311-10.449) | 0.870 |

Measured 2026-09-09 23:00:22-23:01:27 UTC; TT-Lang `7ad660d45227`; TT-Metal/LLVM source pins `ea042c4ad623`/`37aca9d384347`; compiler/TTNN/Metal binary SHA-256 prefixes `a31540fb2fad`/`62edde2b1f61`/`65380f11dc15`; container digest `6eaf96b4b00d5`.

All measured outputs pass the unchanged BF16 checks against FP32 PyTorch:
Pearson correlation >= 0.99 and elementwise relative/absolute tolerances of 0.05.

These results compare identical compute parameters, not each implementation's
best configuration. They do not reproduce TT-Metal's four- or eight-device
Galaxy trace benchmark. Larger meshes and small-workload performance of this
implementation have not been measured. The native binaries come from the
pinned v1.1.9 image; their exact TT-Metal build commit was not recorded.

The README provides [reproduction commands and full binary identities](README.md#reproduce-the-measurements),
the [timing definition](README.md#measurement-contract), and
[configuration differences from the upstream benchmark](README.md#comparison-with-the-native-benchmark).
