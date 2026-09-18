# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 %python %s | FileCheck %s

"""Import-only frontend coverage for control-flow stored-value kernels."""

import os
import sys
from pathlib import Path

os.environ["TTLANG_COMPILE_ONLY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Inputs.control_flow_stored_values_kernels import (  # noqa: E402
    CONTROL_FLOW_CASES,
)

for case_name, _kernel, _grid_width, _output_count in CONTROL_FLOW_CASES:
    print(f"DEFINED {case_name}")

# CHECK: DEFINED then_only
# CHECK: DEFINED else_only
# CHECK: DEFINED if_else
# CHECK: DEFINED released_input
# CHECK: DEFINED elif_chain
# CHECK: DEFINED elif_gap
# CHECK: DEFINED nested_if
# CHECK: DEFINED sibling_ifs
# CHECK: DEFINED nested_def
# CHECK: DEFINED loop_wrapped
# CHECK: DEFINED external_use
# CHECK: DEFINED parent_and_branch
# CHECK: DEFINED attached_input
