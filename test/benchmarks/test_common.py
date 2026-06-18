# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types

import torch


def _load_common(monkeypatch):
    ttnn = types.ModuleType("ttnn")
    ttnn.bfloat16 = object()
    ttnn.float32 = object()
    ttnn.TILE_LAYOUT = object()
    ttnn.DRAM_MEMORY_CONFIG = object()
    ttnn.from_torch = lambda *args, **kwargs: None
    ttnn.synchronize_device = lambda *args, **kwargs: None

    utils = types.ModuleType("utils")
    correctness = types.ModuleType("utils.correctness")
    correctness.assert_pcc = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setitem(sys.modules, "utils.correctness", correctness)
    sys.modules.pop("benchmarks.common", None)
    common = importlib.import_module("benchmarks.common")
    monkeypatch.setitem(sys.modules, "benchmarks.common", common)
    return common


def test_measure_pcc_rejects_wrong_constant_actual(monkeypatch):
    common = _load_common(monkeypatch)

    golden = torch.tensor([1.0, 2.0, 3.0])
    actual = torch.ones(3)

    assert common.measure_pcc(golden, actual) == 0.0


def test_measure_pcc_rejects_nonfinite_mismatch(monkeypatch):
    common = _load_common(monkeypatch)

    golden = torch.tensor([1.0, 2.0, 3.0])
    actual = torch.tensor([1.0, float("nan"), 3.0])

    assert common.measure_pcc(golden, actual) == 0.0


def test_measure_pcc_rejects_matching_nonfinite_values(monkeypatch):
    common = _load_common(monkeypatch)

    golden = torch.tensor([1.0, float("inf"), 3.0])
    actual = torch.tensor([1.0, float("inf"), 3.0])

    assert common.measure_pcc(golden, actual) == 0.0


def test_measure_pcc_accepts_exact_constant_match(monkeypatch):
    common = _load_common(monkeypatch)

    golden = torch.ones(3)
    actual = torch.ones(3)

    assert common.measure_pcc(golden, actual) == 1.0
