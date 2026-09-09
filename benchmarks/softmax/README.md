# Softmax benchmarks

## Files

- [`flash_chain_8node.py`](flash_chain_8node.py): single-query attention over
  eight worker-local K/V shards. Each worker computes streaming-softmax state;
  a binary tree combines the states and normalizes the output. Eight nodes
  means eight workers on one device, not eight devices.
- [`online_softmax_accumulators.py`](online_softmax_accumulators.py): single-core
  dependent maximum, denominator, and weighted-value updates. `--variant staged`
  uses explicit dataflow-buffer state; `--variant ssa` uses compiler-managed
  accumulator state.
- [`__init__.py`](__init__.py): package marker for module execution.

Both drivers validate against PyTorch, use shared warmed wall-time measurement
helpers, and write CSV results. See `--help` for workload and timing controls.

## Run

Inside the device-test container, after checking device availability:

```bash
source build-docker/env/activate
set -o pipefail
timeout 180 python -m benchmarks.softmax.online_softmax_accumulators \
    --variant staged --runs 10 2>&1 | tee /tmp/device_test.log
timeout 180 python -m benchmarks.softmax.flash_chain_8node \
    --runs 10 2>&1 | tee /tmp/device_test.log
```

See the [benchmark index](../README.md) for other workloads and shared helpers.
