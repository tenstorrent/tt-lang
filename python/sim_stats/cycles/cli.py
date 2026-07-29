# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CLI wiring for the cycle estimator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import build_estimate, resolve_profile
from .parse import build_pipeline
from .report import load_estimate, print_detailed, print_summary, write_json
from .types import CycleEstimate


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tt-lang-sim-cycles",
        description=(
            "Estimate hardware cycles from a tt-lang simulator trace using an "
            "analytical ideal-peak model (work-counts / hardware-profile rates)."
        ),
    )
    parser.add_argument(
        "trace",
        metavar="FILE",
        nargs="?",
        default=None,
        help="JSON Lines trace file produced by tt-lang-sim --trace",
    )
    parser.add_argument(
        "-r",
        "--view-report",
        metavar="REPORT.json",
        default=None,
        help="Render a previously saved JSON report (no trace needed)",
    )
    parser.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        help="Show the full per-kernel table instead of the per-node summary",
    )
    parser.add_argument(
        "-p",
        "--hw-profile",
        default=None,
        help=(
            "Hardware profile: a built-in name or board family (e.g. wormhole), "
            "or a path to a .json profile file (default: wormhole_n300)"
        ),
    )
    parser.add_argument(
        "-o",
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report",
    )
    parser.add_argument(
        "--include-zero-kernels",
        action="store_true",
        help="Summary: also list idle (zero-cycle) nodes",
    )
    args = parser.parse_args()

    def _render(estimate: CycleEstimate) -> None:
        if args.detailed:
            print_detailed(estimate)
        else:
            print_summary(estimate, include_zero=args.include_zero_kernels)

    # View a previously saved report (no trace, no recompute).
    if args.view_report is not None:
        try:
            estimate = load_estimate(args.view_report)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        _render(estimate)
        return

    if args.trace is None:
        parser.error("a trace FILE is required (or use --view-report REPORT.json)")

    trace_path = Path(args.trace).resolve()
    if not trace_path.exists():
        print(f"Error: trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    try:
        hw = resolve_profile(args.hw_profile)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    work = build_pipeline(trace_path)
    if not work:
        print(
            f"Error: no kernel work in {trace_path} "
            "(is it a tt-lang-sim --trace file?)",
            file=sys.stderr,
        )
        sys.exit(1)

    estimate = build_estimate(work, hw)
    _render(estimate)
    if args.json_out is not None:
        out_path = args.json_out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, estimate)
        print(f"\nWrote JSON report: {out_path}")
