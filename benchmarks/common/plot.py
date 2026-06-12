# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-case ratio bar charts (ttlang / reference) for e2e benchmarks.

``save_ratio_plot`` renders one benchmark; ``save_stacked_ratio_plot`` stacks
several into a single figure (one panel per op) for the driver. Lower bars are
better; the dotted line marks parity with the reference.
"""

from typing import Callable, Iterable, List, Mapping, Optional


def _plottable(rows: Iterable[Mapping], ratio_key: str) -> List[Mapping]:
    return [r for r in rows if r.get(ratio_key) is not None]


def _draw_ratio_bars(ax, rows, *, title, ylabel, ratio_key, label_fn):
    """Draw one panel of ratio bars onto ``ax``. ``rows`` must be pre-filtered
    to those carrying a ratio."""
    label_fn = label_fn or (lambda r: str(r.get("label", "")))
    labels = [label_fn(r) for r in rows]
    ratios = [float(r[ratio_key]) for r in rows]
    colors = [
        "#8fbf6e" if v < 1.1 else "#e8b05c" if v < 1.5 else "#d97a7a" for v in ratios
    ]

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

    rows = _plottable(rows, ratio_key)
    if not rows:
        print("no rows to plot", flush=True)
        return

    fig, ax = plt.subplots(figsize=(max(14, len(rows) * 0.9), 7))
    _draw_ratio_bars(
        ax, rows, title=title, ylabel=ylabel, ratio_key=ratio_key, label_fn=label_fn
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"plot saved to {path}", flush=True)


def save_stacked_ratio_plot(panels: List[Mapping], *, path: str) -> None:
    """Stack one ratio panel per benchmark into a single figure.

    Each panel is a mapping with ``rows``, ``title``, ``ylabel``, ``ratio_key``,
    and ``label_fn``. Panels with no rows carrying a ratio are dropped (e.g. an
    op whose ttnn baseline rejected every shape).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot", flush=True)
        return

    drawable = [(p, _plottable(p["rows"], p.get("ratio_key", "ratio"))) for p in panels]
    drawable = [(p, rows) for p, rows in drawable if rows]
    if not drawable:
        print("no rows to plot", flush=True)
        return

    widest = max(len(rows) for _, rows in drawable)
    fig, axes = plt.subplots(
        len(drawable),
        1,
        figsize=(max(14, widest * 0.9), 4.0 * len(drawable)),
        squeeze=False,
    )
    for ax, (p, rows) in zip(axes[:, 0], drawable):
        _draw_ratio_bars(
            ax,
            rows,
            title=p["title"],
            ylabel=p.get("ylabel", "ttlang / reference  (lower is better)"),
            ratio_key=p.get("ratio_key", "ratio"),
            label_fn=p.get("label_fn"),
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"stacked plot saved to {path}", flush=True)
