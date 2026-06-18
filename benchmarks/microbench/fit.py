# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fit the pack/unpack probe sweep to fixed + per-tile cost terms.

Standalone results-processing utility, not imported by the runner: run by hand
on saved MB1 CSVs to derive the cost-model weights for RESULTS.md.

Reads one or more probe CSVs (benchmarks.microbench.sweep output), groups by
(arch, dtype, dst_full_sync_en, fp32_dest_acc_en), and least-squares fits each
per-RISC and the TRISC-max per-iteration time against tiles:

    us_per_iter(tiles) = fixed_us + per_tile_us * tiles

The TRISC-max fit is the round-trip per-tile cost (pipelined throughput basis).
The probe's per-RISC unpack/pack times are pipelined too, not clean engine
isolates -- the per-engine split comes from the LLK microbenchmarks (RESULTS.md).

    python -m benchmarks.microbench.fit "benchmarks/microbench/results/pack_unpack_*.csv"
"""

import csv
import glob
import sys
from collections import defaultdict


def _least_squares(points):
    """Return (intercept, slope, r_squared) for (x, y) points."""
    count = len(points)
    if count < 2:
        return None
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    sum_xy = sum(x * y for x, y in points)
    sum_xx = sum(x * x for x, _ in points)
    denom = count * sum_xx - sum_x * sum_x
    if denom == 0:
        return None
    slope = (count * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / count
    mean_y = sum_y / count
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return intercept, slope, r_squared


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(paths):
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            with open(path) as file:
                rows.extend(csv.DictReader(file))
    return rows


def fit(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if "full_sync" not in row or "fp32_dest_acc" not in row:
            continue  # pre-refactor CSV schema; skip
        key = (
            row["arch"],
            row["dtype"],
            row["full_sync"],
            row["fp32_dest_acc"],
        )
        tiles = _to_float(row.get("tiles"))
        for column in (
            "unpack_us_per_iters",
            "pack_us_per_iters",
            "trisc_max_us_per_iters",
        ):
            value = _to_float(row.get(column))
            if tiles is not None and value is not None:
                groups[key][column].append((tiles, value))
    return groups


def main():
    paths = sys.argv[1:] or ["benchmarks/microbench/results/pack_unpack_*.csv"]
    rows = load_rows(paths)
    if not rows:
        print(f"no rows found in {paths}")
        return
    groups = fit(rows)
    header = (
        f"{'arch':<10} {'dtype':<5} {'fsync':<5} {'fp32':<5} "
        f"{'metric':<14} {'fixed_us':>10} {'per_tile_us':>12} {'r2':>7}"
    )
    print(header)
    print("-" * len(header))
    for key, metrics in groups.items():
        arch, dtype, fsync, fp32 = key
        for metric, points in metrics.items():
            result = _least_squares(sorted(points))
            if result is None:
                continue
            intercept, slope, r_squared = result
            print(
                f"{arch:<10} {dtype:<5} {fsync:<5} {fp32:<5} "
                f"{metric:<14} {intercept:>10.4f} {slope:>12.4f} {r_squared:>7.4f}"
            )


if __name__ == "__main__":
    main()
