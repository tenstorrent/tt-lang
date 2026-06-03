# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unified benchmark driver.

Sweeps every registered e2e benchmark and writes each op's CSV + plot. Adding a
benchmark is one line in ``E2E`` -- timing, reporting, plotting, and the device
lifecycle all live in ``benchmarks.common``.

    python -m benchmarks.driver                 # all e2e benchmarks
    python -m benchmarks.driver --only matmul   # one op
    python -m benchmarks.driver --filter 8k     # one shape, all ops
"""

import argparse
from pathlib import Path

from benchmarks.common import run_spec
from benchmarks.e2e import matmul, rmsnorm, topk

# Each entry is a BenchSpec; order is the run order.
E2E = [matmul.SPEC, rmsnorm.SPEC, topk.SPEC]


def main(argv=None):
    ap = argparse.ArgumentParser(description="run the tt-lang op benchmarks")
    ap.add_argument("--only", default=None, help="run a single benchmark by name")
    ap.add_argument("--filter", default=None, help="substring to select cases by label")
    ap.add_argument("--out-dir", default="/tmp", help="directory for CSV/PNG output")
    ap.add_argument("--plot", action="store_true", help="write a ratio PNG per benchmark")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    specs = [s for s in E2E if args.only is None or s.name == args.only]
    if not specs:
        names = ", ".join(s.name for s in E2E)
        raise SystemExit(f"no benchmark named {args.only!r}; have: {names}")

    for spec in specs:
        print(f"\n=== {spec.name} ===", flush=True)
        run_spec(
            spec,
            filter=args.filter,
            warmup=args.warmup,
            runs=args.runs,
            csv=str(out_dir / f"{spec.name}_e2e.csv"),
            plot=args.plot,
        )


if __name__ == "__main__":
    main()
