# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The static CB table: logical DFB names to final physical CB ids.

Rows are built from the same DataflowBuffer objects that produce the ttnn CB
descriptors, so the table cannot disagree with what the device is configured
with.

The mapping is not one to one. ``ttl-finalize-dfb-indices`` reuse-colors user
DFBs and then compacts the survivors, so several logical names can share one
physical slot and every index can be renumbered. Rows therefore carry an alias
set, not a single name: "this DFB was not merged, so its id is unchanged" is
false in general.

Every compile appends its table to a fixed file, so after a hang the last block
in that file is the program that hung. Each block is stamped with a UTC time,
pid and program hash so a stale file is recognizable as stale.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .dataflow_buffer import CompilerAllocatedDFBConfig
from .kernel_runner import _ensure_ttnn, cb_geometry

# Fixed destination, overridable; "0" disables the write.
CB_TABLE_PATH = "/tmp/ttlang_cb_table.txt"
PATH_ENV = "TTLANG_CB_TABLE"

COMPILER_ALLOCATED_NAME = "<compiler>"
UNNAMED_NAME = "<unnamed>"

# The first write in a process replaces the file left by an earlier run; later
# compiles in the same process append.
_started = False


@dataclass(frozen=True)
class CBTableRow:
    """One physical CB slot, with every logical name that landed on it."""

    index: int
    names: Tuple[str, ...]
    dtype: str
    shape: Optional[Tuple[int, ...]]
    tile: Optional[Tuple[int, int]]
    block_count: int
    page_size: int
    num_pages: int
    total_size: int


def write_cb_table(
    cb_configs: List[Any],
    names: Dict[int, Tuple[str, ...]],
    program_hash=None,
    source_file: Optional[str] = None,
) -> Optional[str]:
    """Append this program's CB table to the fixed file. Returns the path.

    None when the write is disabled, or when ttnn is unavailable and so page
    sizes cannot be resolved.
    """
    global _started
    path = os.environ.get(PATH_ENV, CB_TABLE_PATH)
    if path == "0":
        return None
    if _ensure_ttnn() is None:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"=== {stamp} pid {os.getpid()} program_hash {program_hash} "
        f"source {source_file} ==="
    )
    body = format_cb_table(build_cb_table(cb_configs, names))
    with open(path, "a" if _started else "w") as fd:
        fd.write(f"{header}\n{body}\n\n")
    _started = True
    return path


def record_dfb_name(dfb, name: str) -> None:
    """Remember the Python identifier a DFB was captured under.

    A buffer can be captured under different names by different threads, so
    names accumulate in first-seen order.
    """
    if name not in dfb.debug_names:
        dfb.debug_names = dfb.debug_names + (name,)


def resolve_cb_names(
    cb_configs: List[Any], index_map: Dict[int, int]
) -> Dict[int, Tuple[str, ...]]:
    """Map final CB id to every logical name that reached it.

    Takes the CB config list *before* ``_apply_dfb_index_map``, because that
    pass keeps only the largest DFB per slot and so discards the names of the
    buffers it merges away.
    """
    names: Dict[int, Tuple[str, ...]] = {}
    for provisional_index, cb in enumerate(cb_configs):
        if cb is None:
            continue
        final_index = index_map.get(provisional_index, provisional_index)
        existing = names.get(final_index, ())
        names[final_index] = existing + tuple(
            n for n in cb.debug_names if n not in existing
        )
    return names


def build_cb_table(
    cb_configs: List[Any], names: Optional[Dict[int, Tuple[str, ...]]] = None
) -> List[CBTableRow]:
    """Build the table for a final (post-index-map, post-merge) CB config list."""
    names = names or {}
    rows = []
    for index, cb in enumerate(cb_configs):
        geometry = cb_geometry(index, cb)
        row_names = names.get(index, ())
        if not row_names:
            row_names = (
                COMPILER_ALLOCATED_NAME
                if isinstance(cb, CompilerAllocatedDFBConfig)
                else UNNAMED_NAME,
            )
        rows.append(
            CBTableRow(
                index=index,
                names=row_names,
                dtype=_dtype_name(geometry.data_format),
                shape=geometry.shape,
                tile=geometry.tile,
                block_count=geometry.block_count,
                page_size=geometry.page_size,
                num_pages=geometry.num_pages,
                total_size=geometry.total_size,
            )
        )
    return rows


def format_cb_table(rows: List[CBTableRow]) -> str:
    """Render the table for a terminal."""
    total_bytes = sum(row.total_size for row in rows)
    summary = f"tt-lang CB table: {len(rows)} CBs, {total_bytes} bytes of L1 backing store"
    if not rows:
        return summary

    header = ("id", "names", "shape", "tile", "blk", "dtype", "page", "pages", "bytes")
    body = [
        (
            str(row.index),
            ", ".join(row.names),
            _shape_str(row.shape),
            _tile_str(row.tile),
            str(row.block_count),
            row.dtype,
            str(row.page_size),
            str(row.num_pages),
            str(row.total_size),
        )
        for row in rows
    ]
    widths = [max(len(c) for c in column) for column in zip(header, *body)]
    lines = [summary]
    for cells in [header] + body:
        row = "  ".join(c.ljust(w) for c, w in zip(cells, widths))
        lines.append("  " + row.rstrip())
    return "\n".join(lines)


def _dtype_name(data_format) -> str:
    return getattr(data_format, "name", str(data_format))


def _shape_str(shape: Optional[Tuple[int, ...]]) -> str:
    if shape is None:
        return "-"
    return "x".join(str(d) for d in shape)


def _tile_str(tile: Optional[Tuple[int, int]]) -> str:
    if tile is None:
        return "-"
    return "x".join(str(d) for d in tile)
