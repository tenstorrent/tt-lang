# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
User-defined signpost profiler for tt-lang kernels.

Enabled via TTLANG_SIGNPOST_PROFILE=1. Parses device profiler CSV for
user-defined signpost zones (created with `with ttl.signpost("name"):`)
and prints a per-region cycle count summary.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# User signpost zones are prefixed with "ttl_" to distinguish from
# tt-metal internal zones.
_USER_PREFIX = "ttl_"


def is_signpost_profile_enabled() -> bool:
    return os.environ.get("TTLANG_SIGNPOST_PROFILE", "0") == "1"


def parse_signpost_zones(
    csv_path: Path,
) -> List[Tuple[str, str, int]]:
    """Parse user-defined signpost zones from the device profile CSV.

    Returns:
        List of (display_name, thread, cycles) tuples, in CSV order.
    """
    results = []
    starts: Dict[str, int] = {}

    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        if header is None:
            return results
        for row in reader:
            if len(row) < 13:
                continue

            try:
                timestamp = int(row[5])
            except (ValueError, IndexError):
                continue

            thread = row[3]
            zone_name = row[10]
            zone_type = row[11]

            if not zone_name or not zone_name.startswith(_USER_PREFIX):
                continue

            key = f"{thread}_{zone_name}"

            if zone_type == "ZONE_START":
                starts[key] = timestamp
            elif zone_type == "ZONE_END" and key in starts:
                duration = timestamp - starts[key]
                display_name = zone_name[len(_USER_PREFIX) :]
                results.append((display_name, thread, duration))
                del starts[key]

    return results


def format_report(zones: List[Tuple[str, str, int]]) -> str:
    """Format the signpost profile report, aggregated by (name, thread)."""
    if not zones:
        return ""

    # Aggregate by (name, thread), preserving first-seen order
    aggregated: Dict[Tuple[str, str], List[int]] = {}
    for name, thread, cycles in zones:
        key = (name, thread)
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(cycles)

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("SIGNPOST PROFILE")
    lines.append("=" * 80)
    lines.append("")

    max_name = max(len(k[0]) for k in aggregated)
    col_w = max(max_name, 4) + 2

    lines.append(
        f"  {'NAME':<{col_w}} {'THREAD':<12} "
        f"{'COUNT':>6} {'TOTAL':>12} {'AVG':>10} {'MIN':>10} {'MAX':>10}"
    )
    lines.append(
        f"  {'-' * col_w} {'-' * 12} "
        f"{'-' * 6} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}"
    )

    for (name, thread), cycles_list in aggregated.items():
        count = len(cycles_list)
        total = sum(cycles_list)
        avg = total // count
        lo = min(cycles_list)
        hi = max(cycles_list)
        lines.append(
            f"  {name:<{col_w}} {thread:<12} "
            f"{count:>6} {total:>12,} {avg:>10,} {lo:>10,} {hi:>10,}"
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    return "\n".join(lines)


def run(logs_path: Path) -> str:
    """Run signpost profiler and return formatted report.

    Args:
        logs_path: Directory containing profile_log_device.csv
    """
    csv_path = logs_path / "profile_log_device.csv"
    if not csv_path.exists():
        return ""

    zones = parse_signpost_zones(csv_path)
    return format_report(zones)
