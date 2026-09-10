# All-gather matmul: TT-Lang vs TT-Metal

This benchmark compares all-gather, matrix multiplication and row-bias addition
with TT-Metal's
[`ttnn.experimental.all_gather_minimal_matmul_async`](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async.cpp).
Both compared versions return the same replicated output.

## Files

| File or directory | Purpose |
| --- | --- |
| [`__main__.py`](__main__.py) | Inputs, correctness checks, device timing and provenance for both implementations. |
| [`profile.py`](profile.py) | Python dispatch profiling; not used for device-performance results. |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Four-device results and exact measured configurations. |
| [`images/`](images/) | Four-device dataflow diagrams. |
| [Examples](../../examples/all_gather_minimal_matmul/README.md) | Shared implementation and separate N-sharded/replicated-weight entry points. |
| [Device timer](../device_timing.py) | TT-Metal profiler analysis and multi-program device intervals. |
| [Single-device matmul](../matmul/README.md) | Matmul-only benchmarks. |

## Comparison with the native benchmark

`M`, `K` and `N` denote complete matrix dimensions; `D=4`.
Activations are initially K-sharded. All inputs/output use TILE layout and
interleaved DRAM. Both TT-Lang variants use the same
[matmul implementation](https://github.com/tenstorrent/tt-lang/blob/bnorris/all-gather-output-replication/examples/all_gather_minimal_matmul/operation.py).

| Tensor on each of four devices | TT-Lang N-sharded + output gather | TT-Lang replicated weights | Native |
| --- | --- | --- | --- |
| Activation input | M x K/4 | M x K/4 | M x K/4 |
| Weight input | K x N/4 | K x N | K x N |
| Bias input | 1 x N/4 | 1 x N | 1 x N |
| Computed output | M x N/4 | M x N | M x N |
| Returned output | Replicated M x N | Replicated M x N | Replicated M x N |

`--gather-output --n-tiles N_TILES` selects N-sharded compute followed by
output all-gather. `--variant replicated --n-tiles N_TILES` instead computes
the complete output on each device. `N_TILES=N/32`.
Without either selection, TT-Lang returns N-sharded output; that result is not
equivalent to the native replicated output.

| Measurement condition | TT-Lang | Native reference |
| --- | --- | --- |
| Devices | Four Blackhole P150b | Same four devices |
| Global M / K | 3072 / 5120 | 3072 / 5120 |
| Global N | 1280 or 3840 | 1280 or 3840 |
| Precision | BF16 input/output, HiFi2, FP32 destination and packer accumulation | Same |
| Bias | Included | Included |
| Timing | Device trace replay; includes final gather when selected | Device trace replay of fused program |
| Compute grid and blocking | Measured settings in [PERFORMANCE.md](PERFORMANCE.md) | Upstream Blackhole test: transposed 12x9, 8/8/8 tiles, 2x2 subblock |
| Fabric | 2D, strict initialization, 8192-byte payload | 1D ring, strict initialization, 8192-byte payload |
| Native communication settings | Not applicable | Two links, six workers/link, 24 channel buffers |

Reference sources:
[Wan2.2 test](https://github.com/tenstorrent/tt-metal/blob/f69f924c6b4f38daa0a6f25716731f36c573dc0e/models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py)
and [block sweep](https://github.com/tenstorrent/tt-metal/blob/f69f924c6b4f38daa0a6f25716731f36c573dc0e/models/tt_dit/utils/sweep_mm_block_sizes.py).
These source revisions and the installed binary hashes are recorded separately
in each report.

## Run the comparison

Activate the TT-Lang build environment and ensure the four devices are idle.
The runner discovers the physical 2x2 mesh and reshapes its logical coordinates
to 4x1 for the native operation's single collective axis.

N-sharded compute plus replicated output, global N=1280:

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --implementation ttlang --gather-output --mesh-shape 4x1 \
    --fabric-config 2d --fabric-reliability strict --fabric-router-payload 8192 \
    --m-tiles 96 --k-tiles-per-device 40 --n-tiles 40 \
    --worker-grid 2 10 --transpose \
    --m-block-tiles 4 --k-block-tiles 8 --n-block-tiles 1 \
    --output-gather-block-tiles 10 --no-reuse-activation \
    --warmup 3 --samples 5 --json /tmp/ttlang-n1280.json
```

For the measured N=3840 case, use `--n-tiles 120`,
`--k-block-tiles 10`, `--n-block-tiles 3`,
`--output-gather-block-tiles 30`,
`--activation-all-gather ring --output-all-gather ring`.

Native, global N=1280:

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --implementation ttmetal --variant replicated --mesh-shape 4x1 \
    --fabric-reliability strict --fabric-router-payload 8192 \
    --native-num-links 2 --native-workers-per-link 6 --native-channel-buffers 24 \
    --m-tiles 96 --k-tiles-per-device 40 --n-tiles 40 \
    --worker-grid 12 9 --transpose \
    --m-block-tiles 8 --k-block-tiles 8 --n-block-tiles 8 --native-subblock 2 2 \
    --warmup 3 --samples 5 --json /tmp/native-n1280.json
```

For N=3840, change only `--n-tiles 120` and the report filename.

TT-Lang replicated-weight measurements use `--variant replicated`,
`--activation-all-gather ring --reuse-activation`, transposed 2x10,
M/K blocks 2/10, and N blocks 2 or 6 for global N=1280 or 3840.

## Collective comparison

`--activation-all-gather all_to_all|ring` selects direct peer transfers or
ring forwarding. `--output-all-gather` independently selects the final gather.
`--output-gather-block-tiles` controls its message size independently of
matmul blocking.

`--collective-only activation|output --implementation ttlang` measures a
standalone collective with exact replica checks. Use
`--n-tiles-per-device` for its output shard width. Standalone activation gather
writes a complete DRAM result for validation; fused activation gather remains
in L1. Select collectives using complete-operation timing, not standalone
timings alone.

## Measurement contract

The runner uses TT-Metal's
[`device_kernel_duration` analysis](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/tools/tracy/device_post_proc_config.py)
through `tracy.process_device_log.import_log_run_stats`.

1. Compile, warm up and validate against FP32 PyTorch.
2. Prepare/reset runtime resources outside capture, then capture and replay
   one invocation. Replicated-output selections enable trace replay automatically.
3. On each device, measure first kernel start through last kernel end.
   N-sharded compute plus output gather includes both programs and their gap.
4. Average the four device durations, then report the median of five samples.
   `--device-aggregation max` instead selects the slowest device per sample.

Device clocks are independent; timestamps from different devices are never
subtracted. The interval includes device communication, compute and waits.
It excludes host dispatch, resource preparation, correctness checking and
profiler processing.

Reports contain UTC timestamps, source revisions/hashes, dirty-tree status,
binary hashes, actual device IDs, configurations, correctness results and
per-device cycles. Raw profiler reports remain in private worker directories
beside the requested JSON, outside the repository. A failed run exits nonzero;
an existing combined report is not replaced.

## Dataflow

These diagrams show N-sharded TT-Lang matmul with optional final output
gather, and the replicated-weight native operation. Dedicated device DRAM is
described in the
[TT-Metalium architecture introduction](https://github.com/tenstorrent/tt-metal/blob/f69f924c6b4f38daa0a6f25716731f36c573dc0e/docs/source/tt-metalium/tt_metal/labs/matmul/lab1/lab1.rst#L227-L230).

![TT-Lang four-device dataflow](images/ttlang_four_device.svg)

![TT-Metal four-device dataflow](images/ttmetal_four_device.svg)

| Algorithm | TT-Lang | Native |
| --- | --- | --- |
| Activation communication | Direct peer transfers or ring forwarding | Bidirectional ring forwarding |
| Gathered activation storage | L1 dataflow buffers | DRAM scratch followed by L1 dataflow buffers |
| Activation reuse | Full-K L1 cache when selected; streaming otherwise | Gathered activation read from DRAM |
| Output replication | Optional final gather for N-sharded weights; inherent for replicated weights | Inherent with replicated weights |
