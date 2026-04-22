# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Render a per-shape ratio plot (ttlang / ttnn.matmul) from sweep results.

Called by sweep.py after the sweep completes. Stand-alone usage also
supported: `python3 plot.py /tmp/ksplit_sweep.csv` reads the CSV sweep.py
wrote and produces a PNG alongside it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, Mapping


def save_plot(
    results: Iterable[Mapping],
    path: str = "/tmp/ksplit_sweep_ratio.png",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot", flush=True)
        return

    rows = [r for r in results if r.get("ratio") is not None]
    if not rows:
        print("no rows to plot", flush=True)
        return

    labels = [r["label"] for r in rows]
    ratios = [float(r["ratio"]) for r in rows]
    colors = ["#2ca02c" if v <= 1.0 else "#ff7f0e" if v <= 1.2 else "#d62728"
              for v in ratios]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(labels))
    ax.bar(x, ratios, color=colors, alpha=0.85)

    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(y=1.1, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=1.2, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)

    for i, v in enumerate(ratios):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("ttlang / ttnn.matmul  (lower is better)")
    ax.set_title("ttlang matmul vs ttnn.matmul  (bar = ratio, dotted = 1.0)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(ratios) * 1.15 + 0.1)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"plot saved to {path}", flush=True)


def _read_csv(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/ksplit_sweep.csv")
    out = src.with_suffix(".png")
    save_plot(_read_csv(src), path=str(out))
