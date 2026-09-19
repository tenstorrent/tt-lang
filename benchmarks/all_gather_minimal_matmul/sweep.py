# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned four-device AGMM cases and collect concise summaries."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmarks.all_gather_minimal_matmul.sweep_cases import (
    COMPARABLE_OPERATION_KINDS,
    NATIVE_SUPPORTED_USE_CASES,
    UPSTREAM_AGMM_CASES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-fabric-config",
        choices=("1d-ring", "1d-line"),
        default="1d-ring",
    )
    parser.add_argument("--topology", choices=("ring", "linear"), default="ring")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--case-timeout-seconds", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ttmetal-source-root", type=Path)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument(
        "--all-grid-candidates",
        action="store_true",
        help="measure every source row instead of one production-resolved run per input",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_native_command(
    arguments: argparse.Namespace,
    case_id: str,
    report: Path,
    compute_grid: tuple[int, int] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.all_gather_minimal_matmul",
        "--implementation",
        "ttmetal",
        "--sweep-case",
        case_id,
        "--native-fabric-config",
        arguments.native_fabric_config,
        "--topology",
        arguments.topology,
        "--native-heuristic",
        "--warmup",
        str(arguments.warmup),
        "--samples",
        str(arguments.samples),
        "--json",
        str(report),
    ]
    if arguments.ttmetal_source_root is not None:
        command.extend(["--ttmetal-source-root", str(arguments.ttmetal_source_root)])
    if compute_grid is not None:
        command.extend(
            ["--native-compute-grid", str(compute_grid[0]), str(compute_grid[1])]
        )
    return command


def select_native_cases(case_ids: list[str] | None, all_grid_candidates: bool = False):
    cases = [
        case
        for case in UPSTREAM_AGMM_CASES
        if case.operation_kind in COMPARABLE_OPERATION_KINDS
        and case.use_case in NATIVE_SUPPORTED_USE_CASES
        and (not case_ids or case.case_id in case_ids)
    ]
    if not all_grid_candidates:
        unique_cases = {}
        for case in cases:
            unique_cases.setdefault(case.comparison_id, case)
        cases = list(unique_cases.values())
    unsupported = [
        case.case_id
        for case in UPSTREAM_AGMM_CASES
        if case.operation_kind not in COMPARABLE_OPERATION_KINDS
        or case.use_case not in NATIVE_SUPPORTED_USE_CASES
    ]
    return cases, unsupported


def read_measurement(
    report: Path, case_id: str, comparison_id: str | None = None
) -> dict[str, Any]:
    result = json.loads(report.read_text())
    variant = result["variants"]["ttmetal"]
    if variant["sweep_case"] != case_id:
        raise ValueError(
            f"{report} records {variant['sweep_case']}, expected {case_id}"
        )
    measurement = variant["measurements"]
    return {
        "case_id": case_id,
        "comparison_id": comparison_id or case_id,
        "status": "passed",
        "median_us": measurement["median_us"],
        "min_us": measurement["min_us"],
        "max_us": measurement["max_us"],
        "report": str(report),
    }


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    temporary_path = summary_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(summary, indent=2) + "\n")
    temporary_path.replace(summary_path)


@contextmanager
def exclusive_output_lock(output_dir: Path):
    """Reject concurrent sweeps that would overwrite the same checkpoint."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".sweep.lock"
    lock_file = lock_path.open("w")
    try:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another sweep is writing to {output_dir}; lock: {lock_path}"
            ) from error
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        yield
    finally:
        lock_file.close()


def run_sweep(arguments: argparse.Namespace) -> None:
    cases, unsupported = select_native_cases(
        arguments.case_ids, arguments.all_grid_candidates
    )
    summary = {
        "native_fabric_config": arguments.native_fabric_config,
        "topology": arguments.topology,
        "warmup": arguments.warmup,
        "samples": arguments.samples,
        "case_timeout_seconds": arguments.case_timeout_seconds,
        "ttmetal_source_root": (
            str(arguments.ttmetal_source_root)
            if arguments.ttmetal_source_root is not None
            else None
        ),
        "native_case_ids": [case.comparison_id for case in cases],
        "source_case_ids": [case.case_id for case in cases],
        "all_grid_candidates": arguments.all_grid_candidates,
        "unsupported_case_ids": unsupported,
        "results": [],
    }
    summary_path = arguments.output_dir / f"summary_{arguments.topology}.json"
    write_summary(summary_path, summary)
    for case in cases:
        report_id = (
            case.case_id if arguments.all_grid_candidates else case.comparison_id
        )
        report = arguments.output_dir / f"{report_id}_{arguments.topology}.json"
        command = build_native_command(
            arguments,
            case.case_id,
            report,
            case.compute_grid if arguments.all_grid_candidates else None,
        )
        print(" ".join(command), flush=True)
        if not arguments.dry_run:
            try:
                if not (arguments.resume and report.exists()):
                    subprocess.run(
                        command,
                        check=True,
                        timeout=arguments.case_timeout_seconds,
                    )
                summary["results"].append(
                    read_measurement(report, case.case_id, case.comparison_id)
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                summary["results"].append(
                    {
                        "case_id": case.case_id,
                        "comparison_id": case.comparison_id,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "report": str(report),
                    }
                )
                write_summary(summary_path, summary)
                if arguments.fail_fast:
                    raise
                continue
            write_summary(summary_path, summary)
    print(f"Summary: {summary_path}", flush=True)


def main() -> None:
    arguments = parse_args()
    with exclusive_output_lock(arguments.output_dir):
        run_sweep(arguments)


if __name__ == "__main__":
    main()
