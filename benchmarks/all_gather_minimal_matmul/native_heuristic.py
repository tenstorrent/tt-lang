# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The TT-Metal AGMM blocking heuristic used by the comparison harness."""

from math import ceil


def choose_agmm_blocking(
    m_tiles: int,
    full_k_tiles: int,
    n_tiles_per_device: int,
    compute_grid: tuple[int, int],
) -> tuple[int, int, int, tuple[int, int]]:
    """Return ``(M_block, K_block, N_block, (sub_h, sub_w))``.

    This mirrors ``_compute_heuristic_blocking`` in TT-Metal's
    ``models/tt_dit/utils/matmul.py``.  AGMM receives the gathered activation,
    so K blocking is based on full K tiles, not the original per-device shard.
    The supplied grid is the matmul worker grid; AGMM's caller has already
    removed the mux row or column when it derives that grid.
    """

    grid_x, grid_y = compute_grid
    if min(m_tiles, full_k_tiles, n_tiles_per_device, grid_x, grid_y) <= 0:
        raise ValueError("tile counts and compute-grid extents must be positive")

    if n_tiles_per_device <= 4:
        subblock = (4, 1)
    elif n_tiles_per_device % 4 == 0:
        subblock = (1, 4)
    elif n_tiles_per_device % 3 == 0 and n_tiles_per_device <= 24:
        subblock = (1, 3)
    elif n_tiles_per_device % 2 == 0:
        subblock = (1, 2)
    else:
        subblock = (1, 1)

    subblock_h, subblock_w = subblock
    m_block = min(
        max(subblock_h, ceil(ceil(m_tiles / grid_x) / subblock_h) * subblock_h), 16
    )
    n_block = min(
        max(subblock_w, ceil((n_tiles_per_device / grid_y) / subblock_w) * subblock_w),
        16,
    )
    while n_tiles_per_device % n_block and n_block > subblock_w:
        n_block -= subblock_w
    n_block = max(subblock_w, n_block)

    k_block = min(8, full_k_tiles)
    while full_k_tiles % k_block and k_block > 1:
        k_block -= 1

    return m_block, k_block, n_block, subblock
