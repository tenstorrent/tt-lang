# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from benchmarks.all_gather_minimal_matmul.native_heuristic import choose_agmm_blocking
from benchmarks.all_gather_minimal_matmul.sweep_cases import UPSTREAM_AGMM_CASES


def test_wide_output_uses_full_k_tiles():
    assert choose_agmm_blocking(296, 160, 120, (12, 9)) == (16, 8, 12, (1, 4))


def test_narrow_output_uses_m_direction_subblock():
    assert choose_agmm_blocking(32, 24, 4, (12, 9)) == (4, 8, 1, (4, 1))


def test_non_multiple_n_uses_even_subblock_when_available():
    assert choose_agmm_blocking(4, 192, 18, (12, 9)) == (1, 8, 3, (1, 3))


def test_upstream_manifest_is_pinned_and_tile_aligned():
    assert len(UPSTREAM_AGMM_CASES) == 155
    assert all(case.m_elements % 32 == 0 for case in UPSTREAM_AGMM_CASES)
    assert all(case.full_k_elements % 128 == 0 for case in UPSTREAM_AGMM_CASES)
    assert all(case.n_elements_per_device % 32 == 0 for case in UPSTREAM_AGMM_CASES)
