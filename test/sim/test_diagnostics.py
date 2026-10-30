# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for how the simulator finds the compiler's diagnostics module.

The simulator formats kernel errors, deadlocks and unsupported copy patterns
with ``ttl.diagnostics``, which it loads by path. If the path is wrong, the
lookup itself fails and the user sees a FileNotFoundError instead of the
error in their kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sim import diagnostics as sim_diagnostics
from sim.diagnostics import lazy_import_diagnostics


def _place_simulator_at(monkeypatch: pytest.MonkeyPatch, sim_dir: Path) -> None:
    sim_dir.mkdir(parents=True)
    monkeypatch.setattr(sim_diagnostics, "__file__", str(sim_dir / "diagnostics.py"))
    # The loader registers what it loads; put back whatever was there before.
    monkeypatch.setitem(
        sys.modules, "ttl.diagnostics", sys.modules.get("ttl.diagnostics")
    )


def test_diagnostics_load_in_the_installed_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both wheels install the simulator as ``ttl/sim/`` next to ``ttl/diagnostics.py``.

    Looking only for the source-tree layout resolves to ``ttl/ttl/diagnostics.py``
    there, so every kernel error turns into a FileNotFoundError.
    """
    root = tmp_path.resolve()
    _place_simulator_at(monkeypatch, root / "ttl" / "sim")
    (root / "ttl" / "diagnostics.py").write_text("LAYOUT = 'installed'\n")

    assert lazy_import_diagnostics().LAYOUT == "installed"


def test_diagnostics_load_in_the_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    _place_simulator_at(monkeypatch, root / "python" / "sim")
    (root / "python" / "ttl").mkdir()
    (root / "python" / "ttl" / "diagnostics.py").write_text("LAYOUT = 'source'\n")

    assert lazy_import_diagnostics().LAYOUT == "source"


def test_missing_diagnostics_names_the_places_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A packaging change that drops the file should be easy to trace."""
    root = tmp_path.resolve()
    _place_simulator_at(monkeypatch, root / "ttl" / "sim")

    with pytest.raises(ImportError) as excinfo:
        lazy_import_diagnostics()

    reason = str(excinfo.value)
    for path in (
        root / "ttl" / "ttl" / "diagnostics.py",
        root / "ttl" / "diagnostics.py",
    ):
        assert str(path) in reason, f"error does not name {path}:\n{reason}"
