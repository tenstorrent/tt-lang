# All-gather matmul performance

Four Blackhole P150b devices; global M/K/N=9472/5120/15360. Inputs and outputs
are BF16 TILE tensors in interleaved DRAM. Matmul uses HiFi2 and FP32
destination/packer accumulation. Each implementation uses its best measured
configuration; equal worker counts and blocks are not required.

## Replicated output

These implementations return `M x N` on every device, including row bias.

| Implementation | Worker roles/device | Device median ms (min-max) | / Native | Warmups/samples |
| --- | ---: | ---: | ---: | ---: |
| TT-Lang V4 N-sharded matmul + output gather | 120 compute; 4 fabric; 6 local distribution; gather: 2 | 52.025 (51.721-52.213) | 7.522 | 3/10 |
| TT-Lang V3 replicated weights | 130 compute; 2 activation exchange | 10.248 (10.151-10.713) | 1.482 | 1 per group / 4 |
| Native replicated weights | 108 compute; 24 also fabric clients; 4 mux-only | 6.916 (6.900-6.989) | 1.000 | 2/3 |

V4 uses two programs; the complete interval includes both programs and the
interval between them. Its accepted matmul configuration leaves L1 capacity
for `1x5`-tile output-gather messages, so output replication dominates the
complete time. V3 first gathers activations to DRAM, then runs replicated
matmul. Native uses one fused program.

## N-sharded output

These results stop after each device computes its `M x N/4` shard. Collectively,
the four shards contain one complete output, but no device has a replicated
`M x N` tensor; comparison with the native replicated-output result is not
equivalent.

| TT-Lang implementation | Worker roles/device | M/K/N blocks, tiles | Device median ms (min-max) | Warmups/samples |
| --- | ---: | ---: | ---: | ---: |
| V2 two-worker ring | 60 compute; 2 activation exchange | 2/8/4 | 13.488 (13.432-13.509) | 2/5 |
| V4 dedicated communication | 120 compute; 4 fabric; 6 local distribution | 4/10/12 | 3.548 (3.516-3.575) | 3/10 |

V4 is 73.7% faster than V2. Four column-zero workers exchange activations over
fabric and multicast four compute rows directly. Six more column-zero workers
multicast the other eight rows after local relays. The 120 compute workers
concurrently distribute weights, initialize accumulators from bias, accumulate
matmul in L1, and write N-sharded output. Bounded L1 DFBs provide backpressure;
this configuration streams K and does not allocate gathered-activation DRAM
storage. Compute-private bias-conversion and accumulation DFBs each hold one
block, permitting four-row M blocks within the per-node L1 budget. Remote
activation blocks multicast directly from the fabric receive DFB, eliminating
one intervening L1 copy.

## Matmul lowering

Each V4 compute worker converts its BF16 row-bias block to FP32, broadcasts it
into the accumulator, then consumes matching activation and weight DFB pages
for each K block. `ttl.math.matmul(..., dtype=accumulator.dtype)` lowers the
BF16 operands to `matmul_block_init` and `matmul_block` with direct FP32
packing. `llk_pack_reconfig_l1_acc` accumulates subsequent K blocks in the same
L1 DFB; the completed accumulator is converted once to BF16. The
[native compute kernel](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp#L472-L525)
uses the same direct-packing accumulation mechanism. The V4 data-movement
kernels produce operand pages concurrently, so compute waits only when a
required page is unavailable.

## Measured configurations

| Configuration | TT-Lang V4 N-sharded | TT-Lang V4 + output gather | TT-Lang V3 replicated | Native |
| --- | --- | --- | --- | --- |
| Transposed compute grid | 12x10 | 12x10 | 13x10 | 12x9 |
| M/K/N blocks, tiles | 4/10/12 | 4/10/12 | 8/8/8 | 8/8/8; 2x2 subblock |
| Activation storage | Streamed L1 DFBs | Streamed L1 DFBs | Gathered DRAM tensor | Gathered DRAM tensor |
| Activation collective | Ring, four fabric workers and six local distribution workers | Ring, four fabric workers and six local distribution workers | Direct, two workers; 2x40-tile messages | Bidirectional ring; 24 compute workers as fabric clients, four mux-only workers |
| Final output collective | None | Direct, two workers; 1x5-tile messages | None | None |
| Fabric | 2D | 2D | 2D | 1D ring |
| Fabric initialization/payload | Strict / 8192 bytes | Strict / 8192 bytes | Strict / 8192 bytes | Strict / 8192 bytes |
| Native link configuration | Not applicable | Not applicable | Not applicable | Two links; six workers/link; 24 channel buffers |

Device trace replay measures first kernel start through final kernel end on
each device, then averages the four device intervals. Host preparation,
dispatch, correctness checks, and profiler processing are excluded. Every
result passes PCC >= 0.99 and elementwise relative/absolute tolerances of 0.05
against FP32 PyTorch.

Measured 2026-09-10/11 UTC: V2 23:10:01 (`bf55d854e3c1` plus source hashes in
its report), V3 18:24:46 (`e1aef7532`), native 09:12:02 (`0546a5523`), and V4
N-sharded 14:15:46 (operation SHA-256 `0242a4652c1d`, committed in
`fb674329a`). V4 with output gather was measured at 14:39:26 with the same
operation source. TT-Metal/LLVM source pins:
`ea042c4ad623`/`37aca9d384347`; CAPI/TTNN/Metal SHA-256 prefixes:
`3b10ee20db65`/`62edde2b1f61`/`65380f11dc15`; IRD v1.1.9 image digest:
`6eaf96b4b00d`. [Full reports and hashes](https://gist.github.com/brnorris03/da754d5cef08ed989cc241b023fdaccb).

[Run commands](README.md#run-the-comparison),
[timing definition](README.md#measurement-contract), and
[native benchmark references](README.md#comparison-with-the-native-benchmark).
