# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test lit configuration without a compiler, TTNN, or a device."""

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EMULE_SETTINGS = (
    "TT_EMULE_JIT_CACHE_DIR",
    "TT_METAL_ALLOCATOR_MODE_HYBRID",
    "EMULE_FABRIC8",
)


@pytest.fixture
def load_lit_config(monkeypatch, tmp_path):
    lit = ModuleType("lit")
    lit.__path__ = []
    lit.formats = ModuleType("lit.formats")
    lit.formats.ShTest = lambda **kwargs: SimpleNamespace(**kwargs)
    lit.util = ModuleType("lit.util")
    lit.llvm = ModuleType("lit.llvm")
    lit.llvm.llvm_config = None
    for module in (lit, lit.formats, lit.util, lit.llvm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    monkeypatch.setattr(os, "listdir", lambda path: [])

    def load(environment):
        monkeypatch.setattr(os, "environ", environment.copy())
        config = SimpleNamespace(
            ttlang_obj_root=str(tmp_path / "build"),
            ttlang_source_dir=str(REPO_ROOT),
            ttlang_build_type="Release",
            python_executable=sys.executable,
            environment={"PRESERVED_LIT_SETTING": "keep"},
            substitutions=[],
            available_features=set(),
        )
        runpy.run_path(
            str(REPO_ROOT / "test" / "lit.cfg.py"), init_globals={"config": config}
        )
        return config

    return load


@pytest.mark.parametrize("mode", ["1", "0"])
def test_lit_preserves_emule_runtime_settings(load_lit_config, tmp_path, mode):
    environment = {
        "TT_METAL_EMULE_MODE": "1",
        "TT_EMULE_JIT_CACHE_DIR": str(tmp_path / "custom jit cache"),
        "TT_METAL_ALLOCATOR_MODE_HYBRID": mode,
        "EMULE_FABRIC8": mode,
        "TT_METAL_CACHE": str(tmp_path / "metal-cache"),
        "MESH_DEVICE": "P150",
    }

    config = load_lit_config(environment)

    for name, value in environment.items():
        assert config.environment[name] == value
    assert config.environment["PRESERVED_LIT_SETTING"] == "keep"
    assert "tt-device" in config.available_features


def test_lit_does_not_default_absent_emule_settings(load_lit_config):
    environment = {"TT_METAL_EMULE_MODE": "1", "MESH_DEVICE": "P150"}

    config = load_lit_config(environment)

    for name in EMULE_SETTINGS:
        assert name not in config.environment
    for name, value in environment.items():
        assert config.environment[name] == value


def test_lit_preserves_non_emule_runtime_settings(load_lit_config):
    environment = {
        "TT_METAL_SIMULATOR": "/runtime/simulator",
        "TT_METAL_HOME": "/runtime/metal",
        "TT_VISIBLE_DEVICES": "0,1",
        "MESH_DEVICE": "N300",
    }

    config = load_lit_config(environment)

    for name, value in environment.items():
        assert config.environment[name] == value
    for name in EMULE_SETTINGS:
        assert name not in config.environment
    assert "TT_METAL_EMULE_MODE" not in config.environment
    assert config.environment["PRESERVED_LIT_SETTING"] == "keep"
    assert "tt-device" in config.available_features
