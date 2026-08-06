# tt-lang Benchmarks

Benchmarks in this directory are executable Python modules. Each benchmark owns
its operation-specific setup and imports only the small helpers in
`benchmarks.common`.

Shared helpers:

- `to_device`: copy a torch tensor to a device tensor.
- `assert_pcc`: imported from `utils.correctness` for correctness checks.
- `measure_pcc`: compute a PCC score for CSV reporting.
- `time_runs`: run warmup iterations, synchronize once, enqueue measured runs
  back-to-back, synchronize once, and report mean wall time. This matches the
  measurement convention from the closed benchmark-harness PR 661.
- `write_csv`: write one benchmark result row.

Add a benchmark by creating a package under `benchmarks/`, defining a
module-level operation or workload, and exposing a `main()` with `argparse`.
Prefer CSV output under `/tmp` unless the caller passes a different path.

Example:

```bash
python -m benchmarks.softmax.flash_chain_8node --runs 10
python -m benchmarks.softmax.online_softmax_accumulators --variant staged
python -m benchmarks.softmax.online_softmax_accumulators --variant ssa
```
