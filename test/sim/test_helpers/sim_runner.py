# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helper for running example/fixture scripts under the simulator.

Executing a script through the simulator requires temporarily replacing
``sys.modules["ttl"]`` / ``["ttnn"]`` with the simulator implementations -- the
same shadowing ``ttlang_sim.setup_simulator_imports`` does for the CLI. This
helper centralizes that dance so individual tests do not each reimplement it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sim
from sim.context import set_dry_run
from sim.greenlet_scheduler import set_scheduler_algorithm
from sim.program import set_max_l1_bytes
from sim.ttlang_sim import execute_script_with_simulator


def run_script_in_process(
    script_path: Path,
    scheduler: str = "fair",
    max_l1_bytes: int | None = None,
    no_float32_promotion: bool = False,
    dry_run: bool = False,
) -> tuple[int, str]:
    """Run a script in-process with the simulator backend.

    Args:
        script_path: Path to the Python file to execute.
        scheduler: Scheduler algorithm ('greedy' or 'fair').
        max_l1_bytes: Optional per-node L1 limit override in bytes; uses the
            simulator default when None.
        no_float32_promotion: If True, disable the default float32 promotion so
            the script runs with its declared dtypes (e.g. bfloat16 as bfloat16).
        dry_run: If True, skip math/data operations and run only structural
            checks (also compiles with ``optimize=1`` to strip asserts).

    Returns:
        ``(exit_code, output)`` where exit_code is 0 on success, 1 on error.
    """
    set_scheduler_algorithm(scheduler)
    if max_l1_bytes is not None:
        set_max_l1_bytes(max_l1_bytes)
    if no_float32_promotion:
        sim.ttnn.set_disable_float32_promotion(True)
    if dry_run:
        set_dry_run(True)

    # Shadow sys.modules locally (same as ttlang_sim.setup_simulator_imports()).
    # Done here, and restored in finally, so parallel tests do not interfere.
    original_modules = {"ttl": sys.modules.get("ttl"), "ttnn": sys.modules.get("ttnn")}
    sys.modules["ttl"] = sim.ttl  # type: ignore[assignment]
    sys.modules["ttnn"] = sim.ttnn  # type: ignore[assignment]

    try:
        return execute_script_with_simulator(
            script_path, capture_output=True, optimize=dry_run
        )
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        if no_float32_promotion:
            sim.ttnn.set_disable_float32_promotion(False)
