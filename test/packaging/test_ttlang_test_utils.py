# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared test utility behavior used by packaging workflows."""

from __future__ import annotations

import glob
import importlib.util
import sys
import types
from pathlib import Path

from conftest import REPO_ROOT


def _load_ttlang_test_utils_with_device_glob(monkeypatch, device_nodes: list[str]):
    fake_ttl = types.ModuleType("ttl")
    fake_ttl.__path__ = []
    fake_config = types.ModuleType("ttl.config")
    fake_config.HAS_TT_DEVICE = False
    monkeypatch.setitem(sys.modules, "ttl", fake_ttl)
    monkeypatch.setitem(sys.modules, "ttl.config", fake_config)

    real_glob = glob.glob

    def fake_glob(pattern: str):
        if pattern == "/dev/tenstorrent/*":
            return device_nodes
        if pattern == "/dev/tenstorrent[0-9]*":
            return []
        return real_glob(pattern)

    monkeypatch.setattr(glob, "glob", fake_glob)

    module_path = REPO_ROOT / "test" / "ttlang_test_utils.py"
    module_name = "ttlang_test_utils_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_runtime_device_nodes_override_no_device_build_config(monkeypatch) -> None:
    module = _load_ttlang_test_utils_with_device_glob(
        monkeypatch, ["/dev/tenstorrent/0"]
    )

    assert module.is_hardware_available() is True
