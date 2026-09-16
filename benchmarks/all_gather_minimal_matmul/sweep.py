# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned four-device AGMM cases and collect concise summaries."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.all_gather_minimal_matmul.sweep_cases import (
    COMPARABLE_OPERATION_KINDS,
    COMPARABLE_USE_CASES,
    UPSTREAM_AGMM_CASES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fabric-config", choices=("1d-ring", "1d-line"), default="1d-ring"
    )
    parser.add_argument("--topology", choices=("ring", "linear"), default="ring")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    cases = [
        case
        for case in UPSTREAM_AGMM_CASES
        if case.operation_kind in COMPARABLE_OPERATION_KINDS
        and case.use_case in COMPARABLE_USE_CASES
        and (not arguments.case_ids or case.case_id in arguments.case_ids)
    ]
    unsupported = [
        case.case_id
        for case in UPSTREAM_AGMM_CASES
        if case.operation_kind not in COMPARABLE_OPERATION_KINDS
        or case.use_case not in COMPARABLE_USE_CASES
    ]
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "fabric_config": arguments.fabric_config,
        "topology": arguments.topology,
        "warmup": arguments.warmup,
        "samples": arguments.samples,
        "comparable_case_ids": [case.case_id for case in cases],
        "unsupported_case_ids": unsupported,
        "results": [],
    }
    for case in cases:
        report = arguments.output_dir / f"{case.case_id}_{arguments.topology}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.all_gather_minimal_matmul",
            "--sweep-case",
            case.case_id,
            "--fabric-config",
            arguments.fabric_config,
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
        print(" ".join(command), flush=True)
        if not arguments.dry_run:
            subprocess.run(command, check=True)
            result = json.loads(report.read_text())
            summary["results"].append(
                {
                    "case_id": case.case_id,
                    "ratio": result.get("ttlang_over_ttmetal"),
                    "report": str(report),
                }
            )
    summary_path = arguments.output_dir / f"summary_{arguments.topology}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
