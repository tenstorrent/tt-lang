# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s | FileCheck %s

"""Compile-only frontend coverage for control-flow store fanout."""

import os
import sys
from pathlib import Path

os.environ["TTLANG_COMPILE_ONLY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Inputs.control_flow_store_fanout_kernels import (  # noqa: E402
    CONTROL_FLOW_CASES,
    host_tensor,
)
import ttnn  # noqa: E402
from ttlang_test_utils import to_dram  # noqa: E402

device = ttnn.open_device(device_id=0)
try:
    for case_name, kernel, grid_width, output_count in CONTROL_FLOW_CASES:
        input_tensor = to_dram(host_tensor((32, grid_width * 32)), device)
        output_tensors = [
            to_dram(host_tensor((32, 32)), device)
            for _output_index in range(output_count)
        ]

        kernel(input_tensor, *output_tensors)
        print(f"COMPILED {case_name}")
finally:
    ttnn.close_device(device)

# CHECK: COMPILED then_only
# CHECK: COMPILED else_only
# CHECK: COMPILED if_else
# CHECK: COMPILED elif_chain
# CHECK: COMPILED elif_gap
# CHECK: COMPILED nested_if
# CHECK: COMPILED sibling_ifs
# CHECK: COMPILED nested_def
# CHECK: COMPILED loop_wrapped
# CHECK: COMPILED external_use
# CHECK: COMPILED parent_and_branch
# CHECK: COMPILED attached_input
