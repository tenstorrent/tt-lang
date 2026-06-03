# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared, op-agnostic benchmark infrastructure (timing, reporting, plotting,
and the spec-driven sweep/CLI harness)."""

from .harness import BenchSpec, cli, open_default_device, run_spec, sweep
from .plot import save_ratio_plot
from .report import pcc, write_csv
from .timing import time_runs

__all__ = [
    "BenchSpec",
    "cli",
    "open_default_device",
    "run_spec",
    "sweep",
    "save_ratio_plot",
    "pcc",
    "write_csv",
    "time_runs",
]
