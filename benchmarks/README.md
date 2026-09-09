# tt-lang Benchmarks

Each benchmark owns its operation-specific setup. Directory READMEs describe
the drivers, execution commands, and timing conventions.

## Directory index

- [matmul](matmul/README.md): single-device SUMMA and K-split kernels,
  configuration planner, shape sweeps against `ttnn.matmul`, and ratio plots.
- [all_gather_minimal_matmul](all_gather_minimal_matmul/README.md): two-device
  comparison of TT-Lang and native TT-Metal all-gather matmul, with correctness
  checks, repeated timings, and JSON provenance.
- [softmax](softmax/README.md): eight-worker attention reduction and single-core
  online-softmax accumulator benchmarks.

## Shared helpers

[`common.py`](common.py) provides:

- `to_device`: copy a torch tensor to a device tensor.
- `assert_pcc`: imported from `utils.correctness` for correctness checks.
- `measure_pcc`: compute a PCC score for CSV reporting.
- `time_runs`: run warmup iterations, synchronize once, enqueue measured runs
  back-to-back, synchronize once, and report mean wall time. This matches the
  measurement convention from the closed benchmark-harness PR 661.
- `write_csv`: write one benchmark result row.

[`provenance.py`](provenance.py) records source revisions, dependency pins,
source/binary SHA-256 digests, and environment identity for JSON reports.
[`device_timing.py`](device_timing.py) uses TT-Metal's existing profiler
processor and `device_kernel_duration` metric. The matmul sweep and fused
all-gather comparison use this device metric; `time_runs` measures host call
cost and is not interchangeable with device kernel latency.

Add a benchmark by creating a package under `benchmarks/`, defining a
module-level operation or workload, and exposing a `main()` with `argparse`.
Prefer CSV output under `/tmp` unless the caller passes a different path.

Example:

```bash
python -m benchmarks.softmax.flash_chain_8node --runs 10
python -m benchmarks.softmax.online_softmax_accumulators --variant staged
python -m benchmarks.softmax.online_softmax_accumulators --variant ssa
python -m benchmarks.all_gather_minimal_matmul --dtype bf16
```
