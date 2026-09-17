# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

from benchmarks.all_gather_minimal_matmul.native_heuristic import choose_agmm_blocking
from benchmarks.all_gather_minimal_matmul.sweep import build_native_command
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


def test_sweep_command_runs_only_native_with_native_fabric_configuration():
    arguments = argparse.Namespace(
        native_fabric_config="1d-ring", topology="ring", warmup=3, samples=10
    )
    command = build_native_command(
        arguments,
        "3072x5120x3840_8x8_agmm_plain",
        Path("/tmp/native-ring.json"),
    )

    assert command[:3] == [
        sys.executable,
        "-m",
        "benchmarks.all_gather_minimal_matmul",
    ]
    assert command[command.index("--implementation") + 1] == "ttmetal"
    assert command[command.index("--native-fabric-config") + 1] == "1d-ring"
    assert "--ttlang-fabric-config" not in command
