# All-gather matmul: TT-Lang vs TT-Metal

This benchmark compares all-gather, matrix multiplication and row-bias addition
with TT-Metal's
[`ttnn.experimental.all_gather_minimal_matmul_async`](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async.cpp).
Both compared versions return the same replicated output.

Compare the best measured configuration of each implementation for identical
global inputs, precision and replicated output. Tune compute grids, blocks and
communication independently; equal resource usage is not required. Report
the selected worker count and configuration for each result.

For M/K/N=9472/5120/15360 on four Blackhole P150b devices, replicated TT-Lang
uses 130 compute workers/device and takes 10.248 ms; native uses 108 and takes
6.883 ms (TT-Lang/native: 1.489). N-sharded TT-Lang plus output gather uses
60 workers/device and takes 24.154 ms. See [results and configurations](PERFORMANCE.md).
Performance parity with native has not been established.

## TT-Lang versions

Oldest to newest; all remain runnable.

| Version | Implementation | Output on each device |
| --- | --- | --- |
| 1. Per-row all-gather + matmul | [Operation](../../examples/all_gather_minimal_matmul/per_row_all_gather/operation.py); one communication worker per M-worker row | N-sharded; optionally gathered to replicated output |
| 2. Two-worker ring + matmul | [Operation](../../examples/all_gather_minimal_matmul/two_worker_ring/operation.py); two communication workers serve all M-worker rows | N-sharded; optionally gathered to replicated output |
| 3. DRAM all-gather + replicated matmul | [Operation](../../examples/all_gather_minimal_matmul/replicated/operation.py); gather completes before replicated matmul | Replicated output; no final gather |

[Entry points and selection flags](../../examples/all_gather_minimal_matmul/README.md).
The performance comparison uses versions 2 and 3 against the native TT-Metal reference.

## Files

| File or directory | Purpose |
| --- | --- |
| [`__main__.py`](__main__.py) | Inputs, correctness checks, device timing and provenance for both implementations. |
| [`profile.py`](profile.py) | Python dispatch profiling; not used for device-performance results. |
| [`ccl_comparison.py`](ccl_comparison.py) | Identical activation collective measured alone and within replicated matmul. |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Four-device results and exact measured configurations. |
| [`images/`](images/) | Four-device dataflow diagrams. |
| [Examples](../../examples/all_gather_minimal_matmul/README.md) | Separate N-sharded and replicated-output implementations, with shared collectives. |
| [Device timer](../device_timing.py) | TT-Metal profiler analysis and multi-program device intervals. |
| [Single-device matmul](../matmul/README.md) | Matmul-only benchmarks. |

## Comparison with the native benchmark

`M`, `K` and `N` denote complete matrix dimensions; `D=4`.
Activations are initially K-sharded. All inputs/output use TILE layout and
interleaved DRAM. The [N-sharded implementation](../../examples/all_gather_minimal_matmul/two_worker_ring/operation.py)
fuses activation gathering with matmul. The [replicated implementation](../../examples/all_gather_minimal_matmul/replicated/operation.py)
gathers activations into DRAM first, then computes the complete output on every device.

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
| Global M / K | 9472 / 5120 | 9472 / 5120 |
| Global N | 15360 | 15360 |
| Precision | BF16 input/output, HiFi2, FP32 destination and packer accumulation | Same |
| Bias | Included | Included |
| Timing | Device trace replay; includes final gather when selected | Device trace replay of fused program |
| Compute workers per device | Replicated: 130 (transposed 13x10); N-sharded: 60 (transposed 6x10) | 108 (transposed 12x9) |
| Blocking | Measured settings in [PERFORMANCE.md](PERFORMANCE.md) | 8/8/8 tiles, 2x2 subblock |
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

All commands below use global M/K/N=9472/5120/15360.

TT-Lang replicated output:

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --implementation ttlang --variant replicated --mesh-shape 4x1 \
    --fabric-config 2d --fabric-reliability strict --fabric-router-payload 8192 \
    --m-tiles 296 --k-tiles-per-device 40 --n-tiles 480 \
    --worker-grid 13 10 --transpose \
    --m-block-tiles 8 --k-block-tiles 8 --n-block-tiles 8 \
    --no-reuse-activation --activation-all-gather all_to_all \
    --math-fidelity HiFi2 --fp32-dest-acc \
    --warmup 3 --samples 5 --json /tmp/ttlang-replicated-n15360.json
```

N-sharded compute plus output all-gather:

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --implementation ttlang --gather-output --mesh-shape 4x1 \
    --fabric-config 2d --fabric-reliability strict --fabric-router-payload 8192 \
    --m-tiles 296 --k-tiles-per-device 40 --n-tiles 480 \
    --worker-grid 6 10 --transpose \
    --m-block-tiles 2 --k-block-tiles 8 --n-block-tiles 4 \
    --reuse-activation --activation-all-gather ring \
    --output-all-gather all_to_all --output-gather-workers 2 \
    --output-gather-block-tiles 30 --output-gather-m-block-tiles 2 \
    --math-fidelity HiFi2 --fp32-dest-acc \
    --warmup 2 --samples 5 --json /tmp/ttlang-n-sharded-n15360.json
```

Native replicated output:

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --implementation ttmetal --variant replicated --mesh-shape 4x1 \
    --fabric-config 1d-ring --fabric-reliability strict --fabric-router-payload 8192 \
    --native-num-links 2 --native-workers-per-link 6 --native-channel-buffers 24 \
    --m-tiles 296 --k-tiles-per-device 40 --n-tiles 480 \
    --worker-grid 12 9 --transpose \
    --m-block-tiles 8 --k-block-tiles 8 --n-block-tiles 8 --native-subblock 2 2 \
    --warmup 2 --samples 3 --json /tmp/native-n15360.json
```

For the smaller native M/K/N=3072/5120/3840 case, change `--m-tiles 96`,
`--n-tiles 120`, and the report filename. TT-Lang requires independently tuned
blocks and grids; the larger-case configurations above are not smaller-case optima.

## Collective comparison

`--activation-all-gather all_to_all|ring` selects direct peer transfers or
ring forwarding. `--output-all-gather` independently selects the final gather.
`--output-gather-m-block-tiles` and `--output-gather-block-tiles` control its
message's row and column tile counts independently of matmul blocking.

`--compare-activation-ccl --variant replicated --implementation ttlang` measures
the full operation, its activation collective alone, then the full operation
again. All three stages reuse the same collective instance and input/output
tensors; matmul's allocations remain live during the isolated measurement.
The report records actual collective geometry, including padded M, and saves
each completed stage. Full timings include the activation gather, matmul and
device gap; isolated timings include only the collective. Every sample checks
the replicated result, with exact checks for the collective.

Run each existing algorithm with the same full-operation configuration:

```bash
for algorithm in all_to_all ring; do
    python -m benchmarks.all_gather_minimal_matmul \
        --compare-activation-ccl --implementation ttlang --variant replicated \
        --mesh-shape 4x1 --fabric-config 2d --fabric-reliability strict \
        --fabric-router-payload 8192 --activation-all-gather "$algorithm" \
        --m-tiles 296 --k-tiles-per-device 40 --n-tiles 480 \
        --worker-grid 13 10 --transpose \
        --m-block-tiles 8 --k-block-tiles 8 --n-block-tiles 8 \
        --no-reuse-activation --math-fidelity HiFi2 --fp32-dest-acc \
        --warmup 2 --samples 5 --worker-timeout 600 \
        --json "/tmp/activation-ccl-$algorithm.json"
done
```

This configuration uses logical M/K/N=9472/5120/15360, padded M=9984,
130 matmul workers and two collective workers per device, with 2x40-tile
collective messages. Both algorithms are TT-Lang implementations. The operation
gathers into DRAM before matmul; this does not measure native's overlapping
collective or the N-sharded implementation's fused L1 transfers.

`--collective-only activation|output` instead constructs an independent
standalone collective. Its worker/block flags need not match a full operation;
use the comparator above when selecting the replicated operation's collective.

## Measurement contract

The runner uses TT-Metal's
[`device_kernel_duration` analysis](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/tools/tracy/device_post_proc_config.py)
through `tracy.process_device_log.import_log_run_stats`.

1. Allocate the output gather's persistent L1 buffers before matmul's buffers
   to preserve contiguous matmul workspace. Compile, warm up and validate
   against FP32 PyTorch.
2. Prepare/reset runtime resources outside capture, then capture and replay
   one invocation. Replicated-output selections enable trace replay automatically.
3. On each device, measure first kernel start through last kernel end.
   Both N-sharded compute plus output gather and replicated activation gather
   plus matmul include both programs and their gap. The workload records its
   program count explicitly; single-device replicated matmul has one program.
4. Average the four device durations, then report the median of the requested samples.
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

| Algorithm | TT-Lang N-sharded | Native |
| --- | --- | --- |
| Activation communication | Direct peer transfers or ring forwarding | Bidirectional ring forwarding |
| Gathered activation storage | L1 dataflow buffers | DRAM scratch followed by L1 dataflow buffers |
| Activation reuse | Full-K L1 cache when selected; streaming otherwise | Gathered activation read from DRAM |
| Output replication | Optional final gather | Inherent with replicated weights |
