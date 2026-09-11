# All-gather matmul performance

This comparison measures column-parallel all-gather matmul with K-sharded
activation and N-sharded weight, bias, and output. The four output shards
collectively contain one `M x N` result.

## Results

Four Blackhole P150b devices; global `M/K/N=9472/5120/15360`; per-device
`M/K/N=9472/5120/3840`.

| Implementation | Device median ms (min-max) | TT-Lang/native | Warmups/samples |
| --- | ---: | ---: | ---: |
| TT-Lang | 3.577 (3.540-3.603) | 1.812 | 3/10 |
| Native `all_gather_minimal_matmul_async` | 1.974 (1.959-2.004) | 1.000 | 3/10 |

Both results passed PCC >= 0.99 and elementwise relative/absolute tolerances of
0.05 against FP32 PyTorch for every warmup and sample.

## Configurations

Each implementation uses its selected grid, blocking, and communication
configuration. Equal resource use is not required.

| Setting | TT-Lang | Native |
| --- | --- | --- |
| Activation input/device | `9472 x 1280`, K-sharded; storage padded to `10752 x 1280` | `9472 x 1280`, K-sharded |
| Weight input/device | distinct `5120 x 3840` N shard | distinct `5120 x 3840` N shard |
| Bias input/device | distinct `1 x 3840` N shard | distinct `1 x 3840` N shard |
| Output/device | one distinct `9472 x 3840` N shard | three adjacent `9472 x 1280` chunks comprising one distinct `9472 x 3840` N shard |
| Compute grid | `12 x 10`; 120 compute workers | `12 x 9`; 108 compute workers |
| M/K/N blocks | `4/10/12` tiles | `7/5/16` tiles |
| Output subblock | `1 x 4` tiles; direct FP32 packer accumulation | `1 x 2` tiles |
| Communication workers | 4 fabric; 6 L1 distribution | 24 compute workers are fabric clients; 4 mux-only workers |
| Activation collective | one-direction TT-Lang ring into L1 | bidirectional native ring into gathered-activation DRAM storage |
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

Measured 2026-09-11 18:40-18:41 UTC: TT-Lang source `d9c8fee3c`, operation
SHA-256 `7ef4767cded3`, comparison runner SHA-256 `8f27dcbf7d4b`; TT-Metal
`ea042c4ad623`; LLVM `37aca9d384347`; firmware 18.12.1; IRD v1.1.9.

[Raw device-profiler reports](https://gist.github.com/brnorris03/fa7ab25c12872de92dc0727f28f16104).
[Reproduction command and timing definition](README.md#run-the-comparison).
