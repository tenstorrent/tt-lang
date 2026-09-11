# All-gather matmul benchmark

This benchmark compares the [TT-Lang column-parallel operation](../../examples/all_gather_minimal_matmul/operation.py)
with TT-Metal's
[`all_gather_minimal_matmul_async`](https://github.com/tenstorrent/tt-metal/tree/ea042c4ad6237678103cd7cbceb346e060f0f9a3/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async).
Both receive K-sharded activation and N-sharded weight and bias, then return one
distinct N-sharded output per device.

| File | Contents |
| --- | --- |
| [`__main__.py`](__main__.py) | Workload construction, correctness checks, and device-profiler timing |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Accepted result, complete configuration, and provenance |

## Run

The defaults reproduce the four-device Wan2.2 QKV comparison documented in
[`PERFORMANCE.md`](PERFORMANCE.md):

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --json /tmp/all-gather-minimal-matmul.json
```

The parent process runs TT-Lang and TT-Metal in separate profiler processes.
All tensor, worker-grid, block, fabric, and native collective parameters are
CLI options; `--help` lists the measured defaults.

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
