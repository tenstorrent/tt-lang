# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Flash-shard cycle A/B over several decode lengths.

Sweeps the ttl ``flash_mla`` shard against the metal ``compute_sdpa_chunk``
baseline and reports the device-cycle ratio per shape. Each variant runs in its
own device open/close: Tracy only writes the device CSV on close, and every
``run`` clears the log first, so each measurement is isolated (no stale zones).

Shapes are keyed by the per-core slice (tile-rows), labelled by the decode seq
they represent in a 256-way MLA shard (32k seq = 128 tile-rows on one core).
The metal side groups tile-rows into ``chunk_size`` per ``compute_sdpa_chunk``
call; chunk_size divides the tile-row count so both sides cover the same work.

    TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.flash_shard.sweep
"""

import argparse

from . import metal_sdpa, shapes, ttl_shard

# (label, tile_rows, metal_chunk_size). tile_rows == ttl N_CHUNKS == the per-core
# K slice; metal num_chunks = tile_rows // metal_chunk_size.
SHAPES = [
    ("1k", 4, 4),
    ("32k", 128, 8),
    ("64k", 256, 8),
]

FIELDS = (
    "label", "tile_rows", "ttl_cyc", "ttl_us", "metal_cyc", "metal_us",
    "ratio", "ttl_pcc", "metal_pcc",
)


def run_case(label, tile_rows, metal_chunk):
    """One shape: run both variants (clean open/close each) and build a row."""
    ttl = ttl_shard.run(tile_rows)
    metal = metal_sdpa.run(num_chunks=tile_rows // metal_chunk, chunk_size=metal_chunk)
    return {
        "label": label,
        "tile_rows": tile_rows,
        "ttl_cyc": ttl["cycles"],
        "ttl_us": round(ttl["us"], 1),
        "metal_cyc": metal["cycles"],
        "metal_us": round(metal["us"], 1),
        "ratio": round(ttl["cycles"] / metal["cycles"], 2),
        "ttl_pcc": round(ttl["pcc"], 4),
        "metal_pcc": round(metal["pcc"], 4),
    }


def sweep(filter=None):
    """Run every (selected) shape; return result rows."""
    rows = []
    for label, tile_rows, metal_chunk in SHAPES:
        if filter and filter not in label:
            continue
        row = run_case(label, tile_rows, metal_chunk)
        rows.append(row)
        print(
            f"{label:<5} {tile_rows:>4} rows  "
            f"ttl {row['ttl_us']:>9.1f}us / metal {row['metal_us']:>7.1f}us  "
            f"= {row['ratio']:.1f}x  "
            f"(pcc ttl {row['ttl_pcc']:.4f} / metal {row['metal_pcc']:.4f})",
            flush=True,
        )
    return rows


PANEL = {
    "title": "flash_shard: ttl vs metal compute_sdpa_chunk (single-core device cycles)",
    "ylabel": "ttl / metal cycles  (lower is better)",
    "ratio_key": "ratio",
    "label_fn": lambda r: f"{r['label']}\n({r['tile_rows']} rows)",
}


def panel(rows):
    return {**PANEL, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description="flash-shard cycle A/B sweep")
    ap.add_argument("--filter", default=None, help="substring to select a shape by label")
    args = ap.parse_args(argv)
    sweep(filter=args.filter)


if __name__ == "__main__":
    main()
