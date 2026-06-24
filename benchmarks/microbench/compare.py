# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare per-transfer cost across paired tt-lang and C++ baseline variants.

Reads the variant CSVs written by the PipeNet sweeps (``ttlang_pipes.py`` and
``baseline_pipes.py``), fits ``sender_dm_max_us = fixed_us + per_transfer_us *
N`` per variant over the transfer-count sweep (reusing ``fit.py``'s least
squares), and reports per_transfer_us (the cost of one transfer) and fixed_us
(the one-time setup), each variant's ratio to the ``cpp_bounded_ring`` target,
and the published-minus-computed protocol delta.

Generic over the ``variant`` column, so any future paired benchmark that writes
the same schema is comparable with this tool.

    python -m benchmarks.microbench.compare "benchmarks/microbench/results/pipes_*.csv"
"""

import sys
from collections import defaultdict

from benchmarks.microbench.fit import _least_squares, _to_float, load_rows

# The target is a bounded producer/consumer ring at depth 8. It keeps bounded
# slot reuse and cross-core credit flow control, but it is not a tt-lang PipeNet
# and does not preserve per-transfer DFB reserve/push/wait/pop bookkeeping.
# Deeper rings amortize per-chunk work. cpp_optimized is the bulk ceiling with
# bounded buffering removed.
BASELINE = "cpp_bounded_ring_r8"
METRIC = "sender_dm_max_us"


def regress(rows):
    """Map ``(arch, dtype) -> {variant: (fixed_us, per_transfer_us, r2)}``."""
    points = defaultdict(lambda: defaultdict(list))
    for row in rows:
        variant = row.get("variant")
        n = _to_float(row.get("n"))
        y = _to_float(row.get(METRIC))
        if variant and n is not None and y is not None:
            points[(row.get("arch"), row.get("dtype"))][variant].append((n, y))
    fits = defaultdict(dict)
    for key, by_variant in points.items():
        for variant, pts in by_variant.items():
            result = _least_squares(sorted(pts))
            if result is not None:
                intercept, per_transfer, r2 = result
                fits[key][variant] = (intercept, per_transfer, r2)
    return fits


def main():
    paths = sys.argv[1:] or ["benchmarks/microbench/results/pipes_*.csv"]
    rows = load_rows(paths)
    if not rows:
        print(f"no rows found in {paths}")
        return
    fits = regress(rows)
    header = (
        f"{'arch':<10} {'dtype':<5} {'variant':<20} "
        f"{'per_transfer_us':>16} {'fixed_us':>10} {'r2':>7} {'vs_target':>13}"
    )
    for (arch, dtype), by_variant in fits.items():
        print(header)
        print("-" * len(header))
        base = by_variant.get(BASELINE)
        base_per_transfer = base[1] if base else None
        for variant, (intercept, per_transfer, r2) in sorted(by_variant.items()):
            ratio = (
                f"{per_transfer / base_per_transfer:.2f}x"
                if base_per_transfer not in (None, 0)
                else "-"
            )
            print(
                f"{arch:<10} {dtype:<5} {variant:<20} "
                f"{per_transfer:>16.4f} {intercept:>10.4f} {r2:>7.4f} {ratio:>13}"
            )
        computed = by_variant.get("ttlang_computed")
        published = by_variant.get("ttlang_published")
        if computed and published:
            delta_fixed = published[0] - computed[0]
            delta_per_transfer = published[1] - computed[1]
            print(
                "protocol delta (published - computed): "
                f"fixed_us {delta_fixed:+.4f}  "
                f"per_transfer_us {delta_per_transfer:+.4f}"
            )
        print()


if __name__ == "__main__":
    main()
