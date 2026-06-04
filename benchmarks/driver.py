# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unified benchmark driver.

Runs every registered benchmark and writes each one's CSV + plot. There are two
kinds: e2e wall-clock sweeps (vs a ttnn reference) and cycles A/Bs (single-core
device cycles vs a metal primitive). Adding an e2e op is one line in ``E2E``;
timing, reporting, plotting, and the device lifecycle live in ``benchmarks.common``.

    python -m benchmarks.driver                 # all benchmarks (e2e + cycles)
    python -m benchmarks.driver --only matmul   # one e2e op
    python -m benchmarks.driver --only cycles   # just the cycles A/B
    python -m benchmarks.driver --filter 8k     # one shape, all ops

The cycles A/B reads the Tracy device profiler, so it needs
``TT_METAL_DEVICE_PROFILER=1``; without it the driver runs the e2e sweeps and
skips cycles with a note.
"""

import argparse
import os
from pathlib import Path

from benchmarks.common import run_spec, save_stacked_ratio_plot, write_csv
from benchmarks.cycles.flash_shard import sweep as flash_cycles
from benchmarks.e2e import flash_mla, matmul, rmsnorm, topk

# Each entry is a BenchSpec; order is the run order.
E2E = [matmul.SPEC, rmsnorm.SPEC, topk.SPEC, flash_mla.SPEC]

# Cycles A/Bs: (name, sweep_fn(filter)->rows, panel_fn(rows)->panel, fields).
CYCLES = [("cycles", flash_cycles.sweep, flash_cycles.panel, flash_cycles.FIELDS)]


def _profiler_on():
    return os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"


def main(argv=None):
    ap = argparse.ArgumentParser(description="run the tt-lang op benchmarks")
    ap.add_argument("--only", default=None, help="run a single benchmark by name")
    ap.add_argument("--filter", default=None, help="substring to select cases by label")
    ap.add_argument("--out-dir", default="/tmp", help="directory for CSV/PNG output")
    ap.add_argument("--plot", action="store_true", help="write one stacked ratio PNG")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    e2e_specs = [s for s in E2E if args.only is None or s.name == args.only]
    cycles = [c for c in CYCLES if args.only is None or c[0] == args.only]
    if not e2e_specs and not cycles:
        names = ", ".join([s.name for s in E2E] + [c[0] for c in CYCLES])
        raise SystemExit(f"no benchmark named {args.only!r}; have: {names}")

    panels = []
    for spec in e2e_specs:
        print(f"\n=== {spec.name} ===", flush=True)
        rows = run_spec(
            spec,
            filter=args.filter,
            warmup=args.warmup,
            runs=args.runs,
            csv=str(out_dir / f"{spec.name}_e2e.csv"),
            plot=False,
        )
        panels.append(
            {
                "rows": rows,
                "title": spec.plot_title or spec.name,
                "ylabel": spec.plot_ylabel,
                "ratio_key": spec.ratio_key,
                "label_fn": spec.plot_label_of,
            }
        )

    for name, sweep_fn, panel_fn, fields in cycles:
        print(f"\n=== {name} ===", flush=True)
        if not _profiler_on():
            print("  skipped: cycles needs TT_METAL_DEVICE_PROFILER=1 (Tracy)", flush=True)
            continue
        rows = sweep_fn(filter=args.filter)
        csv_path = out_dir / f"{name}.csv"
        write_csv(csv_path, fields, rows)
        print(f"wrote {len(rows)} rows to {csv_path}", flush=True)
        panels.append(panel_fn(rows))

    if args.plot:
        save_stacked_ratio_plot(panels, path=str(out_dir / "benchmarks.png"))


if __name__ == "__main__":
    main()
