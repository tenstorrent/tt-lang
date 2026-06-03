# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reusable per-case ratio bar chart (ttlang / reference) for e2e benchmarks.

Any e2e benchmark whose rows carry a ratio column can plot with this; the
matmul sweep was the first user. Lower bars are better; the dotted line marks
parity with the reference.
"""

from typing import Callable, Iterable, Mapping, Optional


def save_ratio_plot(
    rows: Iterable[Mapping],
    *,
    path: str,
    title: str,
    ylabel: str,
    ratio_key: str = "ratio",
    label_fn: Optional[Callable[[Mapping], str]] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot", flush=True)
        return

    rows = [r for r in rows if r.get(ratio_key) is not None]
    if not rows:
        print("no rows to plot", flush=True)
        return

    label_fn = label_fn or (lambda r: str(r.get("label", "")))
    labels = [label_fn(r) for r in rows]
    ratios = [float(r[ratio_key]) for r in rows]
    colors = [
        "#8fbf6e" if v < 1.1 else "#e8b05c" if v < 1.5 else "#d97a7a" for v in ratios
    ]

    fig, ax = plt.subplots(figsize=(max(14, len(rows) * 0.9), 7))
    x = range(len(labels))
    ax.bar(x, ratios, color=colors, alpha=0.85)

    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(y=1.1, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=1.5, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)

    for i, v in enumerate(ratios):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(ratios) * 1.15 + 0.1)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"plot saved to {path}", flush=True)
