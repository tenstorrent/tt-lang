# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared test utility behavior used by packaging workflows."""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import types
from pathlib import Path

from conftest import REPO_ROOT


def _load_ttlang_test_utils(
    monkeypatch,
    *,
    device_nodes: list[str] | None = None,
    flat_nodes: list[str] | None = None,
    has_tt_device: bool = False,
    ttl_importable: bool = True,
):
    """Import ttlang_test_utils under controlled device/config conditions.

    device_nodes feeds the /dev/tenstorrent/* glob, flat_nodes the legacy
    /dev/tenstorrent[0-9]* glob, has_tt_device the wheel's build-time
    ttl.config.HAS_TT_DEVICE, and ttl_importable whether ttl imports at all.
    """
    device_nodes = device_nodes or []
    flat_nodes = flat_nodes or []

    # Env vars checked before the device-node probe must not leak in from the
    # CI runner. TTLANG_COMPILE_ONLY is set by the module as an import side
    # effect; delenv here so monkeypatch restores its absence on teardown.
    monkeypatch.delenv("TT_METAL_SIMULATOR", raising=False)
    monkeypatch.delenv("TTLANG_HAS_DEVICE", raising=False)
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)

    if ttl_importable:
        fake_ttl = types.ModuleType("ttl")
        fake_ttl.__path__ = []
        fake_config = types.ModuleType("ttl.config")
        fake_config.HAS_TT_DEVICE = has_tt_device
        monkeypatch.setitem(sys.modules, "ttl", fake_ttl)
        monkeypatch.setitem(sys.modules, "ttl.config", fake_config)
    else:
        # None in sys.modules makes `from ttl.config import ...` raise ImportError.
        monkeypatch.setitem(sys.modules, "ttl", None)

    real_glob = glob.glob

    def fake_glob(pattern: str):
        if pattern == "/dev/tenstorrent/*":
            return device_nodes
        if pattern == "/dev/tenstorrent[0-9]*":
            return flat_nodes
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
    # A directory-style node is present but the wheel's build config reports no
    # device; the runtime node must win. This is the core fix.
    module = _load_ttlang_test_utils(
        monkeypatch, device_nodes=["/dev/tenstorrent/0"], has_tt_device=False
    )
    assert module.is_hardware_available() is True


def test_flat_style_device_node_is_detected(monkeypatch) -> None:
    # The legacy flat node (/dev/tenstorrentN) is detected via the second glob.
    module = _load_ttlang_test_utils(
        monkeypatch, flat_nodes=["/dev/tenstorrent0"], has_tt_device=False
    )
    assert module.is_hardware_available() is True


def test_no_nodes_and_no_build_config_is_compile_only(monkeypatch) -> None:
    # No node and HAS_TT_DEVICE=False means not available; guards against a
    # regression (e.g. a bad glob) that would report a device where there is
    # none, and confirms compile-only mode is set.
    module = _load_ttlang_test_utils(monkeypatch, has_tt_device=False)
    assert module.is_hardware_available() is False
    assert os.environ.get("TTLANG_COMPILE_ONLY") == "1"


def test_build_config_true_without_nodes_is_available(monkeypatch) -> None:
    # No runtime node, but the CMake build config reports a device; the
    # ttl.config fallback below the node check still applies.
    module = _load_ttlang_test_utils(monkeypatch, has_tt_device=True)
    assert module.is_hardware_available() is True


def test_no_nodes_and_ttl_unimportable_is_unavailable(monkeypatch) -> None:
    # No node and ttl.config not importable: the fallback is False. The PR
    # changed this from globbing /dev/tenstorrent* (which matched an empty
    # driver directory as available).
    module = _load_ttlang_test_utils(monkeypatch, ttl_importable=False)
    assert module.is_hardware_available() is False
