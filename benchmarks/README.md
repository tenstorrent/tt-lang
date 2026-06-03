# tt-lang benchmarks

Benchmarks for the ops in `python/ttl/ops/`. The op library is the single source
of truth: `test/python/ops/` guards correctness, and the benchmarks here measure
**those exact ops** (no benchmark-local kernel copies).

## Layout

```
benchmarks/
  common/        shared, op-agnostic harness (timing, CSV, plot, sweep/CLI)
  e2e/           end-to-end benchmarks: each op vs a ttnn reference
    matmul/      ttl.ops.matmul ksplit vs ttnn.matmul (+ planner, notes)
  cycles/        single-core cycle counts vs a metal primitive  (planned)
  driver.py      runs every registered benchmark
```

Two kinds of benchmark:

1. **e2e** (`e2e/`) — wall-clock comparison of an op against a ttnn reference
   across a sweep of shapes, reported as a `ttlang / reference` ratio.
2. **cycles** (`cycles/`, planned) — single-core Tracy / device-profiler cycle
   counts against a low-level tt-metal primitive.

## Running

Run from the repo root so `benchmarks` and `ttl` are both importable.

```bash
# one op, full sweep, with a plot
python -m benchmarks.e2e.matmul --plot

# one shape (substring match on the case label)
python -m benchmarks.e2e.matmul --filter 8k

# everything
python -m benchmarks.driver --plot

# everything, one op
python -m benchmarks.driver --only matmul
```

Each benchmark writes a CSV (and, with `--plot`, a PNG) to `/tmp` by default;
`--csv PATH` / `--out-dir DIR` override the location.

## Adding a benchmark

An e2e benchmark declares a `common.BenchSpec` (name, CSV `fields`, `cases`, a
`run_case(device, case, *, warmup, runs) -> dict`, a `label_of`, and an
`open_device`). The shared harness owns the device lifecycle, the timing loop,
failure handling, CSV, plotting, and the CLI. Register the spec in
`driver.py`'s `E2E` list and add a `__main__.py` that calls `common.cli(SPEC)`.
See `e2e/matmul/` for the reference shape.

## Timing

`common.time_runs` enqueues the runs back-to-back and synchronizes the device
**once** after the loop, then averages per run. A per-run sync would measure
single-call dispatch+execute latency (with a host bubble between runs) rather
than steady-state throughput.
```
