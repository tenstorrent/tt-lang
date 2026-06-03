# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Composable e2e benchmark harness.

A benchmark declares *what* to measure as a ``BenchSpec``; the harness owns
*how* to run it (device lifecycle, per-case timing loop, failure handling,
CSV, plot, CLI). The same spec drives both a standalone run
(``python -m benchmarks.e2e.<op> --filter ...``) and the unified ``driver``.

A spec supplies:
  - ``name``    : short id, used for default output paths.
  - ``fields``  : CSV column order.
  - ``cases``   : the shapes/configs to sweep.
  - ``run_case``: ``(device, case, *, warmup, runs) -> dict`` returning one row.
  - ``label_of``: ``case -> str`` used for ``--filter`` selection.
  - ``open_device``: zero-arg device opener (an op may trim worker L1).
Optional plotting hooks let ``run_spec`` emit a ratio chart.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import ttnn

from .plot import save_ratio_plot
from .report import write_csv


def open_default_device():
    return ttnn.open_device(device_id=0)


@dataclass
class BenchSpec:
    name: str
    fields: Sequence[str]
    cases: Sequence
    run_case: Callable  # (device, case, *, warmup, runs) -> dict
    label_of: Callable = field(default=lambda case: str(case))
    open_device: Callable = field(default=open_default_device)
    format_row: Optional[Callable] = None  # row -> str (console line)
    # plotting
    ratio_key: Optional[str] = "ratio"
    plot_title: Optional[str] = None
    plot_ylabel: str = "ttlang / reference  (lower is better)"
    plot_label_of: Optional[Callable] = None  # row -> str


def _line(spec: BenchSpec, row: dict) -> str:
    if spec.format_row is not None:
        return spec.format_row(row)
    keys = [k for k in row if k != "label"]
    body = "  ".join(f"{k}={row[k]}" for k in keys)
    return f"{str(row.get('label', '')):<32}  {body}"


def sweep(spec: BenchSpec, *, filter: Optional[str] = None, warmup=3, runs=5) -> List[dict]:
    """Open the device, run every (selected) case, return the result rows."""
    device = spec.open_device()
    rows: List[dict] = []
    try:
        for case in spec.cases:
            label = spec.label_of(case)
            if filter and filter not in label:
                continue
            try:
                row = spec.run_case(device, case, warmup=warmup, runs=runs)
            except Exception as e:  # one bad shape should not sink the sweep
                print(f"{label:<32}  FAIL: {e}", flush=True)
                continue
            rows.append(row)
            print(_line(spec, row), flush=True)
    finally:
        ttnn.close_device(device)
    return rows


def run_spec(
    spec: BenchSpec,
    *,
    filter: Optional[str] = None,
    warmup=3,
    runs=5,
    csv: Optional[str] = None,
    plot: bool = False,
) -> List[dict]:
    """Sweep a spec, then write its CSV (and optional plot). Returns the rows."""
    rows = sweep(spec, filter=filter, warmup=warmup, runs=runs)
    csv_path = Path(csv) if csv else Path(f"/tmp/{spec.name}_e2e.csv")
    write_csv(csv_path, spec.fields, rows)
    print(f"wrote {len(rows)} rows to {csv_path}", flush=True)
    if plot and spec.ratio_key:
        save_ratio_plot(
            rows,
            path=str(csv_path.with_suffix(".png")),
            title=spec.plot_title or spec.name,
            ylabel=spec.plot_ylabel,
            ratio_key=spec.ratio_key,
            label_fn=spec.plot_label_of,
        )
    return rows


def cli(spec: BenchSpec, argv=None) -> List[dict]:
    """Standalone entry point: ``python -m benchmarks.e2e.<op> [opts]``."""
    ap = argparse.ArgumentParser(description=f"{spec.name} e2e benchmark")
    ap.add_argument("--filter", default=None, help="substring to select cases by label")
    ap.add_argument("--csv", default=None, help="output CSV path")
    ap.add_argument("--plot", action="store_true", help="write a ratio PNG next to the CSV")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args(argv)
    return run_spec(
        spec,
        filter=args.filter,
        warmup=args.warmup,
        runs=args.runs,
        csv=args.csv,
        plot=args.plot,
    )
