# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from benchmarks.all_gather_minimal_matmul import native_heuristic
from benchmarks.all_gather_minimal_matmul.sweep import (
    build_native_command,
    select_native_cases,
)
from benchmarks.all_gather_minimal_matmul.sweep_cases import (
    NATIVE_SUPPORTED_USE_CASES,
    TT_METAL_SWEEP_REVISION,
    UPSTREAM_AGMM_CASES,
)


def test_upstream_manifest_is_pinned_and_tile_aligned():
    assert TT_METAL_SWEEP_REVISION == "0e9d200db976120c129ab0deb13aa3f6d972b723"
    assert len(UPSTREAM_AGMM_CASES) == 156
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


def test_native_case_selection_includes_agmm_epilogues_and_excludes_sagmm():
    cases, unsupported = select_native_cases(None)

    assert len(cases) == 150
    assert len(unsupported) == 6
    assert {case.operation_kind for case in cases} == {"agmm"}
    assert {case.use_case for case in cases} == NATIVE_SUPPORTED_USE_CASES
    assert all(case.operation_kind == "sagmm" for case in UPSTREAM_AGMM_CASES[-6:])
    assert unsupported == [case.case_id for case in UPSTREAM_AGMM_CASES[-6:]]


def test_native_resolver_reproduces_model_configuration(monkeypatch):
    calls = {}

    def get_agmm_config(*args, **kwargs):
        calls.update(kwargs)
        return (
            SimpleNamespace(x=12, y=9),
            SimpleNamespace(
                M_block_size=8,
                K_block_size=3,
                N_block_size=14,
                subblock_h=2,
                subblock_w=2,
            ),
            6,
        )

    def load_symbol(module_name, symbol_name, *_args):
        if symbol_name == "get_agmm_config":
            return get_agmm_config
        assert symbol_name == "agmm_block_size"
        return lambda k_elements, n_elements: (8, 3, 14)

    monkeypatch.setattr(native_heuristic, "load_ttmetal_symbol", load_symbol)
    fake_ttnn = SimpleNamespace(
        CoreCoord=lambda grid_x, grid_y: SimpleNamespace(x=grid_x, y=grid_y)
    )

    result = native_heuristic.resolve_agmm_config(
        ttnn_module=fake_ttnn,
        m_elements=4768,
        full_k_elements=5376,
        n_elements_per_device=7168,
        full_grid=SimpleNamespace(x=12, y=10),
        device_count=4,
        num_links=2,
        compute_grid=(12, 9),
        source_root=Path("/tmp/tt-metal"),
        expected_revision=TT_METAL_SWEEP_REVISION,
        fuse_swiglu=True,
        use_addcmul=False,
    )

    assert result == ((12, 9), 8, 3, 14, (2, 2), 6)
    assert calls["default_block_size"] == (8, 3, 14)
    assert calls["use_heuristic"] is False
    assert calls["force_transpose"] is True


def test_native_resolver_uses_heuristic_without_model_blocking(monkeypatch):
    calls = {}

    def get_agmm_config(*args, **kwargs):
        calls.update(kwargs)
        return (
            SimpleNamespace(x=12, y=9),
            SimpleNamespace(
                M_block_size=4,
                K_block_size=3,
                N_block_size=8,
                subblock_h=2,
                subblock_w=2,
            ),
            6,
        )

    def load_symbol(module_name, symbol_name, *_args):
        if symbol_name == "get_agmm_config":
            return get_agmm_config
        assert symbol_name == "agmm_block_size"
        return lambda k_elements, n_elements: None

    monkeypatch.setattr(native_heuristic, "load_ttmetal_symbol", load_symbol)

    native_heuristic.resolve_agmm_config(
        ttnn_module=SimpleNamespace(
            CoreCoord=lambda grid_x, grid_y: SimpleNamespace(x=grid_x, y=grid_y)
        ),
        m_elements=1024,
        full_k_elements=768,
        n_elements_per_device=4608,
        full_grid=SimpleNamespace(x=13, y=10),
        device_count=4,
        num_links=2,
        compute_grid=(12, 9),
        source_root=Path("/tmp/tt-metal"),
        expected_revision=TT_METAL_SWEEP_REVISION,
        fuse_swiglu=False,
        use_addcmul=False,
    )

    assert calls["default_block_size"] is None
    assert calls["use_heuristic"] is True


def test_native_resolver_rejects_invalid_ring_k_block(monkeypatch):
    def get_agmm_config(*_args, **_kwargs):
        return (
            SimpleNamespace(x=12, y=9),
            SimpleNamespace(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=2,
                subblock_w=2,
            ),
            6,
        )

    def load_symbol(_module_name, symbol_name, *_args):
        if symbol_name == "get_agmm_config":
            return get_agmm_config
        return lambda _k_elements, _n_elements: None

    monkeypatch.setattr(native_heuristic, "load_ttmetal_symbol", load_symbol)

    with pytest.raises(ValueError, match="does not divide 42 K tiles"):
        native_heuristic.resolve_agmm_config(
            ttnn_module=SimpleNamespace(
                CoreCoord=lambda grid_x, grid_y: SimpleNamespace(x=grid_x, y=grid_y)
            ),
            m_elements=4768,
            full_k_elements=5376,
            n_elements_per_device=7168,
            full_grid=SimpleNamespace(x=12, y=10),
            device_count=4,
            num_links=2,
            compute_grid=(12, 9),
            source_root=Path("/tmp/tt-metal"),
            expected_revision=TT_METAL_SWEEP_REVISION,
            fuse_swiglu=True,
            use_addcmul=False,
        )
