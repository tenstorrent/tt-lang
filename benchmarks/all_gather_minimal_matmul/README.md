# Local all-gather matmul comparison

Compares the TT-Lang example with the installed TT-Metal
[`ttnn.experimental.all_gather_minimal_matmul_async`](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async.cpp), using TT-Metal's standard
device profiler and `device_kernel_duration` analysis.

The source permalink uses the repository dependency pin; it does not establish
the build revision of the installed native binary.

## Files

- [`__main__.py`](__main__.py): CLI, matched workload construction, correctness
  checks, isolated profiler workers, and JSON results/provenance.
- [`profile.py`](profile.py): optional cProfile diagnosis of Python/runtime
  overhead. Its host timings are not the kernel comparison.
- [`../device_timing.py`](../device_timing.py): adapter to TT-Metal's
  `tracy.process_device_log.import_log_run_stats`; no local CSV parser.
- [PERFORMANCE.md](PERFORMANCE.md): measurements and generated/native C++ analysis.
  Detailed reports and source snapshots are kept in the
  [external artifact archive](https://gist.github.com/brnorris03/79c57b196efe09355699d40165780088), not in the repository.
- [`__init__.py`](__init__.py): package marker, not a kernel.
- [`operation.py`](../../examples/all_gather_minimal_matmul/operation.py):
  TT-Lang all-gather + matmul + bias implementation, shared with the
  [example](../../examples/all_gather_minimal_matmul/README.md) and
  [device pytest](../../test/python/fabric/test_all_gather_minimal_matmul.py).
  Native kernels come from the installed `ttnn` package.

The [matmul directory](../matmul/README.md) contains the single-device
SUMMA/K-split comparison.

## Run

Inside the fabric container, after verifying devices are unused:

```bash
source build-docker/env/activate
set -o pipefail
timeout 360 python -m benchmarks.all_gather_minimal_matmul \
    --dtype bf16 --samples 5 --json /tmp/all_gather_matmul_bf16.json \
    2>&1 | tee /tmp/device_test.log
```

Use `--dtype fp32`, `--implementation ttlang`, or
`--implementation ttmetal` as needed. The parent enables device profiling
before importing TTNN in each worker. Each implementation runs in a separate
process with a unique profiler directory beside the requested JSON.
Raw logs and individual worker reports are retained; existing logs are not deleted.

The default is global M=64/K=64/N=256, two devices, a 4x2 compute grid per
device, and bias enabled. Activations are K-sharded; weights, bias and output
are N-sharded. All tensors are TILE/interleaved DRAM. The driver discovers
the topology and selects a connected two-device submesh.

Tile/block parameters are exposed by `--help`.
`--k-tiles-per-device 4` exercises repeated transfers and a longer reduction.
Native non-transposed scheduling requires M <= per-device N and at least four
N workers; the driver rejects unsupported configurations.

## Measurement contract

The analysis is the existing TT-Metal
[`device_kernel_duration`](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/tools/tracy/device_post_proc_config.py)
used by its profiler reports. The processor is the same API used in
[TT-Metal's benchmarks](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/tests/ttnn/unit_tests/benchmarks/test_benchmark.py);
the selected metric includes all worker RISCs, not only TRISC1 math time.

- For each device, the metric spans the earliest kernel start to the latest
  kernel end across workers. Each sample reports the maximum device duration.
  Device clocks are never subtracted across chips.
- Three warmups precede five measured invocations by default. Each invocation
  is synchronized and its profiler buffers flushed before validation.
  The newest operation on each participating device is selected; launch IDs
  must advance between samples. Setup programs and warmups are excluded.
- Raw cycles, clock frequency from the profiler header, per-device launch IDs,
  derived microseconds, minimum/median/maximum, and correctness are retained.
  The comparison is the ratio of median durations, not host wall time.
- Compilation, Python dispatch, allocation, semaphore-initialization programs,
  host synchronization, validation, and profiler processing are outside the
  kernel interval. Worker waits within that interval remain included.
- This is not whole-application latency or trace replay. Current TT-Lang
  fabric resource replacement synchronizes between invocations, which prevents
  trace capture; neither implementation is measured through trace replay.
- Inputs, mesh, compute grid, M/K/N blocks, HiFi4, and destination precision
  match: BF16 destinations for BF16 tensors and FP32 destinations for FP32.
- Native uses one linear fabric link, two channel buffers, one worker per M
  block, and additional fabric mux cores. TT-Lang uses direct fabric PipeNets
  on column-zero workers. Both use `FABRIC_1D`.
- TT-Lang writes the complete gathered activation to DRAM. Native scratch holds
  remote shards only; native compute reads its local shard directly. Validation
  checks these respective contracts. Kernel timings include TT-Lang's extra
  local DRAM traffic.
- Native uses packer L1 accumulation and 1x1 subblocks; TT-Lang uses compiler
  selected accumulation. Both outputs must pass identical numerical criteria.

BF16 checks PCC >= 0.99 and rtol/atol 0.05; FP32 checks PCC >= 0.999 and
rtol/atol 0.005. FP32 FPU sources truncate to TF32. This does not claim
full-FP32 arithmetic accuracy. Gather scratch must match exactly.
A failed check exits nonzero without replacing the combined result report.

Reports record UTC time, source revision/status and hashes, binary hashes,
profiler source hashes, topology, device IDs, and compute settings.
Dependency pins do not prove the revision of installed binaries.
Set `BENCHMARK_CONTAINER_IMAGE` to the image digest for publication.

## Host diagnosis

```bash
source build-docker/env/activate
set -o pipefail
timeout 180 python -m benchmarks.all_gather_minimal_matmul.profile \
    --implementation ttlang --runs 30 --output /tmp/ttlang.pstats \
    2>&1 | tee /tmp/device_test.log
```

Leave `TT_METAL_DEVICE_PROFILER` unset for host-only diagnosis.
