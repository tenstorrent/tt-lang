# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-kind geometry + kernel wiring for the single-block subblock benchmark.

All kinds share the same single-core / fixed-block / vary-subblock harness:

  add           : y = a + b   FPU add, full DST capacity.
  matmul        : y = a @ b   K reduces in DST; compiler forces fp32 dest acc.
  bcast_add     : out = bcast_col(b) + a  (b is a tile-column)
  adversarial   : 4-in/4-out SFPU chains  (test_adversarial_multinode)
  comprehensive : 3-in/3-out 20-op fused  (test_comprehensive_multinode)
  silu          : out = y*sigmoid(y)      copy_dst (test_copy_dst, #443)
  rsqrt_abs     : out = x*rsqrt(abs(x))   copy_dst (test_copy_dst, #384)
  axby          : out = a*x + b*y         two live intermediates (test_axby)

Elementwise-style kinds (N same-size blocks) register via ``simple_kind``;
matmul / bcast_add keep custom geometry.

Both force the subblock with --ttl-force-subblock and run compute-isolated
(no-DRAM). The sweep enumerates *every* subblock (each divisor pair of the block
dims) and lets the compiler decide validity: an over-budget subblock fails to
compile and is reported as invalid -- we no longer model the DST budget here.
``KINDS`` maps a name to a ``Kind``; ``sweep.py`` / ``emit.py`` are kind-agnostic
and just take a kind name.

Env overrides: SUBBLOCK_FULL_SYNC, SUBBLOCK_ROW_TILES/COL_TILES (add),
MM_M_TILES/MM_N_TILES/MM_K_TILES (matmul).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

import ttnn

from .single_block_add import make_single_block_add_no_dram
from .single_block_adversarial import make_single_block_adversarial_no_dram
from .single_block_axby import make_single_block_axby_no_dram
from .single_block_bcast_add import make_single_block_bcast_add_no_dram
from .single_block_comprehensive import make_single_block_comprehensive_no_dram
from .single_block_matmul import make_single_block_matmul_no_dram
from .single_block_multi_consumer import (
    make_single_block_mc_branch_no_dram,
    make_single_block_mc_silu_no_dram,
    make_single_block_mc_square_no_dram,
    make_single_block_mc_three_no_dram,
    make_single_block_mc_unary_binary_no_dram,
)
from .single_block_rsqrt_abs import make_single_block_rsqrt_abs_no_dram
from .single_block_silu import make_single_block_silu_no_dram

TILE = 32  # ttnn.TILE_SIZE

DST_FULL_SYNC_EN = bool(int(os.environ.get("SUBBLOCK_FULL_SYNC", "0")))


def _divisors(n):
    return [d for d in (1, 2, 4, 8, 16, 32) if n % d == 0]


def _dram(device, h_tiles, w_tiles):
    return ttnn.from_torch(
        torch.zeros((h_tiles * TILE, w_tiles * TILE), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@dataclass
class Kind:
    name: str
    out_rows: int        # output-block rows in tiles (sR divides this)
    out_cols: int        # output-block cols in tiles (sC divides this)
    make_op: Callable    # (sR, sC, dst_full_sync_en) -> compiled ttl op
    make_tensors: Callable   # (device) -> (a, b, y)
    block_label: str     # e.g. "8x8" or "8x8 k=4"

    def subblocks(self) -> List[Tuple[int, int]]:
        """Every subblock candidate: each divisor pair of the block dims. No
        DST-budget filter -- the compiler rejects over-budget ones at compile
        time and the sweep reports those as invalid."""
        return [
            (sR, sC)
            for sR in _divisors(self.out_rows)
            for sC in _divisors(self.out_cols)
        ]


def simple_kind(name, maker, n_tensors, env_prefix):
    """Kind for an elementwise-style kernel: n same-size ROWxCOL blocks, maker
    taking row/col_tiles_per_block. ROW/COL come from {env_prefix}_ROW/COL_TILES
    (default 8)."""
    row = int(os.environ.get(f"{env_prefix}_ROW_TILES", "8"))
    col = int(os.environ.get(f"{env_prefix}_COL_TILES", "8"))

    def make_op(sR, sC, fs):
        return maker(
            row_tiles_per_block=row, col_tiles_per_block=col,
            grid=(1, 1), dst_full_sync_en=fs,
            compiler_options=f"--ttl-force-subblock={sR},{sC}",
        )

    def make_tensors(device):
        return tuple(_dram(device, row, col) for _ in range(n_tensors))

    return Kind(name, row, col, make_op, make_tensors, f"{row}x{col}")


# ---- matmul: y[M x N] = a[M x K] @ b[K x N], K reduced in DST ----
_MM_M = int(os.environ.get("MM_M_TILES", "8"))
_MM_N = int(os.environ.get("MM_N_TILES", "8"))
_MM_K = int(os.environ.get("MM_K_TILES", "8"))


def _mm_make_op(sM, sN, fs):
    return make_single_block_matmul_no_dram(
        m_tiles=_MM_M, n_tiles=_MM_N, k_tiles=_MM_K,
        grid=(1, 1), dst_full_sync_en=fs,
        compiler_options=f"--ttl-force-subblock={sM},{sN}",
    )


def _mm_tensors(device):
    return (
        _dram(device, _MM_M, _MM_K),
        _dram(device, _MM_K, _MM_N),
        _dram(device, _MM_M, _MM_N),
    )


# ---- bcast_add: out = bcast_col(b) + a; b is a (ROW x 1) tile-column ----
_BCA_ROW = int(os.environ.get("BCA_ROW_TILES", "8"))
_BCA_COL = int(os.environ.get("BCA_COL_TILES", "8"))


def _bca_make_op(sR, sC, fs):
    return make_single_block_bcast_add_no_dram(
        row_tiles_per_block=_BCA_ROW, col_tiles_per_block=_BCA_COL,
        grid=(1, 1), dst_full_sync_en=fs,
        compiler_options=f"--ttl-force-subblock={sR},{sC}",
    )


def _bca_tensors(device):
    # a (ROW x COL), b (ROW x 1) column source, out (ROW x COL).
    return (
        _dram(device, _BCA_ROW, _BCA_COL),
        _dram(device, _BCA_ROW, 1),
        _dram(device, _BCA_ROW, _BCA_COL),
    )


KINDS = {
    "add": simple_kind("add", make_single_block_add_no_dram, 3, "SUBBLOCK"),
    "matmul": Kind("matmul", _MM_M, _MM_N, _mm_make_op, _mm_tensors,
                   f"{_MM_M}x{_MM_N} k={_MM_K}"),
    "bcast_add": Kind("bcast_add", _BCA_ROW, _BCA_COL, _bca_make_op, _bca_tensors,
                      f"{_BCA_ROW}x{_BCA_COL} (b col)"),
    "adversarial": simple_kind("adversarial", make_single_block_adversarial_no_dram, 8, "ADV"),
    "comprehensive": simple_kind("comprehensive", make_single_block_comprehensive_no_dram, 6, "CMP"),
    "silu": simple_kind("silu", make_single_block_silu_no_dram, 3, "SILU"),
    "rsqrt_abs": simple_kind("rsqrt_abs", make_single_block_rsqrt_abs_no_dram, 3, "SILU"),
    "axby": simple_kind("axby", make_single_block_axby_no_dram, 5, "AXBY"),
    # test_dst_multi_consumer patterns (copy_tile insertion stressors)
    "mc_silu": simple_kind("mc_silu", make_single_block_mc_silu_no_dram, 2, "MC"),
    "mc_unary_binary": simple_kind("mc_unary_binary", make_single_block_mc_unary_binary_no_dram, 3, "MC"),
    "mc_three": simple_kind("mc_three", make_single_block_mc_three_no_dram, 5, "MC"),
    "mc_square": simple_kind("mc_square", make_single_block_mc_square_no_dram, 2, "MC"),
    "mc_branch": simple_kind("mc_branch", make_single_block_mc_branch_no_dram, 4, "MC"),
}
