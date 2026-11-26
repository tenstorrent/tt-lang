# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared trace parsing and kernel-name helpers for sim_stats tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator


def iter_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSON Lines file, skipping blank lines."""
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed line {lineno}: {exc}",
                    file=sys.stderr,
                )


def node_from_kernel(kernel: str | None) -> str:
    """Extract the node identifier from a kernel name like 'node3-read'."""
    if kernel and "-" in kernel:
        return kernel.split("-", 1)[0]
    return kernel or "unknown"


def role_from_kernel(kernel: str) -> str:
    """Classify a kernel as read / compute / write / other from its name suffix."""
    node = node_from_kernel(kernel)
    suffix = kernel.removeprefix(f"{node}-") if node != kernel else ""
    return suffix if suffix in {"compute", "read", "write"} else "other"


def as_int(value: Any) -> int:
    """Convert value to int, handling bool (which is technically an int in Python)."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        return int(value)
    return 0


def abbrev_count(value: float) -> str:
    """Format a magnitude compactly with a K/M/B suffix, always 2 decimals."""
    for suffix, divisor in (("", 1.0), ("K", 1e3), ("M", 1e6), ("B", 1e9)):
        if round(abs(value) / divisor, 2) < 1000.0:
            return f"{value / divisor:.2f}{suffix}"
    return f"{value / 1e9:.2f}B"


def node_sort_key(node: str) -> int:
    """Sort nodes numerically: node0 < node1 < ... < node10 (not node0, node1, node10)."""
    try:
        return int(node.removeprefix("node"))
    except ValueError:
        return 0
