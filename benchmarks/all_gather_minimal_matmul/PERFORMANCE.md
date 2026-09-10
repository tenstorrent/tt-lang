# All-gather matmul performance

Four Blackhole P150b devices; global M/K/N=9472/5120/15360. Inputs and outputs
are BF16 TILE tensors in interleaved DRAM. Matmul uses HiFi2 and FP32
destination/packer accumulation. Each implementation uses its best measured
configuration; equal worker counts and blocks are not required.

## Replicated output

These implementations return `M x N` on every device, including row bias.

| Implementation | Compute + communication workers/device | Device median ms (min-max) | / Native | Warmups/samples |
| --- | ---: | ---: | ---: | ---: |
| TT-Lang V4 N-sharded matmul + output gather | 120 + 4; gather: 2 | 16.012 (15.954-16.082) | 2.326 | 3/10 |
| TT-Lang V3 replicated weights | 130 + 2 | 10.248 (10.151-10.713) | 1.489 | 1 per group / 4 |
| Native replicated weights | 108 + 12 | 6.883 (6.873-6.889) | 1.000 | 2/3 |

V4 computes N-sharded output, then runs a separate output all-gather. The
complete two-program device interval is reported, including the interval
between programs. V3 first gathers activations to DRAM, then runs replicated
matmul. Native uses one fused program.

## N-sharded output

These results stop after each device computes its `M x N/4` shard. Collectively,
the four shards contain one complete output, but no device has a replicated
`M x N` tensor; comparison with the native replicated-output result is not
equivalent.

| TT-Lang implementation | Compute + communication workers/device | M/K/N blocks, tiles | Device median ms (min-max) | Warmups/samples |
| --- | ---: | ---: | ---: | ---: |
| V2 two-worker ring | 60 + 2 | 2/8/4 | 13.488 (13.432-13.509) | 2/5 |
| V4 dedicated communication | 120 + 4 | 2/10/12 | 4.870 (4.864-4.922) | 1/3 |

V4 is 63.9% faster than V2. It assigns activation exchange to four workers in
a separate physical column. The other 120 workers concurrently distribute
weights, initialize accumulators from bias, accumulate matmul in L1, and write
N-sharded output. Bounded L1 DFBs provide backpressure; this configuration
streams K and does not allocate gathered-activation DRAM storage.

An instrumented replay recorded 500 activation fabric-send regions and 15,000
matmul regions on each device. Every send region overlapped matmul; 98.8-99.6%
of the send-region interval time was concurrent with matmul. Instrumented
timings are excluded from the tables. The compact overlap report is in the
[timing archive](https://gist.github.com/brnorris03/da754d5cef08ed989cc241b023fdaccb).

## Measured configurations

| Configuration | TT-Lang V4 N-sharded | TT-Lang V4 + output gather | TT-Lang V3 replicated | Native |
| --- | --- | --- | --- | --- |
| Transposed compute grid | 12x10 | 12x10 | 13x10 | 12x9 |
| M/K/N blocks, tiles | 2/10/12 | 2/8/12 | 8/8/8 | 8/8/8; 2x2 subblock |
| Activation storage | Streamed L1 DFBs | Streamed L1 DFBs | Gathered DRAM tensor | Gathered DRAM tensor |
| Activation collective | Ring, four dedicated workers | Ring, four dedicated workers | Direct, two workers; 2x40-tile messages | Bidirectional ring, 12 workers |
| Final output collective | None | Direct, two workers; 2x30-tile messages | None | None |
| Fabric | 2D | 2D | 2D | 1D ring |
| Fabric initialization/payload | Strict / 8192 bytes | Strict / 8192 bytes | Strict / 8192 bytes | Strict / 8192 bytes |
| Native link configuration | Not applicable | Not applicable | Not applicable | Two links; six workers/link; 24 channel buffers |

Device trace replay measures first kernel start through final kernel end on
each device, then averages the four device intervals. Host preparation,
dispatch, correctness checks, and profiler processing are excluded. Every
result passes PCC >= 0.99 and elementwise relative/absolute tolerances of 0.05
against FP32 PyTorch.

Measured 2026-09-10 UTC: native 15:37:07 (`ff72bcb06859`), V2 23:10:01
(`bf55d854e3c1` plus source hashes in its report), V3 18:24:46 (`e1aef7532`),
V4 N-sharded 23:45:36 (`bf55d854e3c1` plus source hashes), and V4 with output
gather 23:40:07 (`bf55d854e3c1` plus source hashes). TT-Metal/LLVM source pins:
`ea042c4ad623`/`37aca9d384347`; CAPI/TTNN/Metal SHA-256 prefixes:
`20a74e405369`/`62edde2b1f61`/`65380f11dc15`; IRD v1.1.9 image digest:
`6eaf96b4b00d`. [Full reports and hashes](https://gist.github.com/brnorris03/da754d5cef08ed989cc241b023fdaccb).

[Run commands](README.md#run-the-comparison),
[timing definition](README.md#measurement-contract), and
[native benchmark references](README.md#comparison-with-the-native-benchmark).
