# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s
# Failed allocation must not emit a report that implies a valid placement.

import importlib.util
from pathlib import Path
import subprocess

source = Path(__file__).with_name("compiler_l1_stress.py")
spec = importlib.util.spec_from_file_location("allocation_stress", source)
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)
result = subprocess.run(
    [
        "ttlang-opt",
        "-pass-pipeline=builtin.module(ttl-finalize-dfb-indices{"
        "memory-model=compiler-l1 sram-allocation-report=true l1-budget-override=64})",
    ],
    input=stress.make_control_prefix_module("blackhole", 1),
    text=True,
    capture_output=True,
    timeout=30,
)
assert result.returncode != 0
assert "budget" in result.stderr, result.stderr
assert "ttlang-sram-report:" not in result.stderr
