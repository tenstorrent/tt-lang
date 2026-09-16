# All-gather matmul benchmark

This benchmark compares the [TT-Lang bidirectional-L1 column-parallel operation](../../examples/all_gather_minimal_matmul/operation_bidirectional_l1.py)
with TT-Metal's
[`all_gather_minimal_matmul_async`](https://github.com/tenstorrent/tt-metal/tree/967ce00c724cd27bf107e00fbfe7406014cfc14e/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async).
Both receive K-sharded activation and N-sharded weight and bias, then return one
distinct N-sharded output per device.

The runner also measures the TT-Lang
[`P_K x P_N` matmul reduce-scatter](../../examples/matmul_reduce_scatter_2d/operation.py),
which preserves two-dimensional tensor placement for larger device counts.

| File | Contents |
| --- | --- |
| [`__main__.py`](__main__.py) | Workload construction, correctness checks, and device-profiler timing |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Accepted result, complete configuration, and provenance |

## Run

The defaults reproduce the accepted four-device comparison documented in
[`PERFORMANCE.md`](PERFORMANCE.md):

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --json /tmp/all-gather-minimal-matmul.json
```

The parent process runs TT-Lang and TT-Metal in separate profiler processes.
All tensor, worker-grid, block, fabric, and native collective parameters are
CLI options; `--help` lists the measured defaults.

List the pinned AGMM rows imported from the TT-Metal sweep and their current
semantic support status:

```bash
python -m benchmarks.all_gather_minimal_matmul --list-sweep-cases
```

Run one comparable row on a four-device physical ring. `--sweep-case` derives
the four-device tensor dimensions from the upstream full-K/per-device-N tuple;
`--native-heuristic` applies TT-Metal's published blocking rule:

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --sweep-case 3072x5120x3840_8x8_agmm_plain \
    --fabric-config 1d-ring \
    --topology ring \
    --native-heuristic \
    --warmup 3 \
    --samples 10 \
    --json /tmp/agmm-3072x5120x3840-ring.json
```

Use `--fabric-config 1d-line --topology linear` for the secondary line
comparison. Rows marked unsupported require matching TT-Lang fused-epilogue or
operation-kind support and are not timed as plain AGMM.

The earlier eight-device comparison remains archived in `PERFORMANCE.md`; it is
not part of the current four-device sweep.

Four-device 2D result:

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --implementation ttlang \
    --ttlang-operation 2d-reduce-scatter \
    --mesh-shape 2x2 \
    --m-tiles 296 \
    --k-tiles-per-device 40 \
    --n-tiles 480 \
    --dtype bf16 \
    --math-fidelity HiFi2 \
    --fp32-dest-acc \
    --ttlang-compute-grid 11 10 \
    --ttlang-m-block-tiles 7 \
    --ttlang-k-block-tiles 10 \
    --ttlang-n-block-tiles 8 \
    --warmup 3 \
    --samples 10 \
    --json /tmp/matmul-reduce-scatter-2d.json
```

## Measurement

Each sample uses TT-Metal's `device_kernel_duration`: first kernel start through
last kernel end for one invocation. The reported value is the mean duration
across participating devices. Host dispatch, synchronization, tensor transfer,
correctness checks, and profiler processing are excluded. Every warmup and
sample is checked against FP32 PyTorch.

The JSON report records the complete arguments, detected devices, source and
binary hashes, dependency revisions, individual device samples, and numerical
error. Reports are written outside the repository. The accepted raw reports are
archived in [this gist](https://gist.github.com/brnorris03/fa7ab25c12872de92dc0727f28f16104).
