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
  cycles/        single-core device-cycle A/Bs vs a metal primitive
    flash_shard/ ttl.ops.flash_mla shard vs metal compute_sdpa_chunk
  driver.py      runs every registered benchmark
```

Two kinds of benchmark:

1. **e2e** (`e2e/`) — wall-clock comparison of an op against a ttnn reference
   across a sweep of shapes, reported as a `ttlang / reference` ratio.
2. **cycles** (`cycles/`) — single-core Tracy / device-profiler cycle counts
   against a low-level tt-metal primitive, reported as a `ttl / metal` ratio.
   One run, no averaging: Tracy is the hardware profiler, so the counts are
   exact.

## Running

Run from the repo root so `benchmarks` and `ttl` are both importable.

```bash
# one op, full sweep, with a plot
python -m benchmarks.e2e.matmul --plot

# one shape (substring match on the case label)
python -m benchmarks.e2e.matmul --filter 8k

# everything (e2e + cycles), one stacked PNG
TT_METAL_DEVICE_PROFILER=1 python -m benchmarks.driver --plot

# everything, one op
python -m benchmarks.driver --only matmul

# just the cycles A/B (needs the profiler)
TT_METAL_DEVICE_PROFILER=1 python -m benchmarks.cycles.flash_shard.sweep
```

Each benchmark writes a CSV (and, with `--plot`, a PNG) to `/tmp` by default;
`--csv PATH` / `--out-dir DIR` override the location.

## Cycles benchmarks

A cycles benchmark runs one kernel once with `TT_METAL_DEVICE_PROFILER=1` and
reads per-zone start/end timestamps from `profile_log_device.csv`. The headline
metric is the device kernel duration (the span of the `*-KERNEL` zones, maxed
over cores); `per_risc` breaks it down so a phase reads as reader- (NCRISC),
compute- (TRISC\*), or writer-bound (BRISC). Cycles are the primary,
frequency-free metric; the microsecond view divides by the chip clock
(`CYCLES_CHIP_FREQ_MHZ`, default 1350 for Blackhole) only for human comparison.

`flash_shard/` compares the `ttl.ops.flash_mla` online-softmax shard against
tt-metal's `compute_sdpa_chunk` (referenced in-place from `third-party/tt-metal`
so we track upstream). Both run a single core over the same per-core MLA decode
slice (PNHt=1, DHt=18, vDHt=16); K streams from DRAM on both sides so the slice
need not be L1-resident. Shapes are keyed by tile-rows and labelled by the decode
seq they represent in a 256-way shard (32k seq = 128 tile-rows on one core):
`1k`/`32k`/`64k`. The driver runs `1k`/`32k`/`64k`.

The metal side pulls `compute_sdpa_chunk` from `sdpa.h` and its header-only
custom-LLK subtree under `third-party/tt-metal/models/demos/deepseek_v3_b1/`.
That path only exists if the **tt-metal submodule is checked out** (at the pin in
`third-party/tt-metal-version`); a checkout is enough, no build is needed (the
LLKs are header-only and compiled into the kernel). The packaged toolchain
tt-metal is a stripped runtime with no `models/`, so it is **not** sufficient --
check out the submodule (`git submodule update --init third-party/tt-metal`, or a
shallow fetch of the pinned commit). Override the include root with
`CYCLES_SDPA_INCLUDE` if it lives elsewhere.

Tracy writes the device CSV only on device **close**, so each variant runs in its
own open/close and `clear_profile_log()` drops the prior CSV first -- every
measurement is isolated, never a stale read. Parse only after `ttnn.close_device`.

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
