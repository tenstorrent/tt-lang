# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s | FileCheck %s

"""Compile-only coverage for values stored from multiple control-flow blocks."""

import os
import sys
from pathlib import Path

import torch
import ttnn

os.environ["TTLANG_COMPILE_ONLY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Inputs.control_flow_stored_values_kernels import (  # noqa: E402
    CONTROL_FLOW_CASES,
    host_tensor,
)


def to_device_tensor(torch_tensor, device):
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


device = ttnn.open_device(device_id=0)
try:
    for case_name, kernel, grid_width, output_count in CONTROL_FLOW_CASES:
        input_tensor = to_device_tensor(host_tensor((32, grid_width * 32)), device)
        output_tensors = [
            to_device_tensor(torch.zeros((32, 32), dtype=torch.bfloat16), device)
            for _output_index in range(output_count)
        ]

        kernel(input_tensor, *output_tensors)
        print(f"COMPILED {case_name}")
finally:
    ttnn.close_device(device)

# CHECK: COMPILED then_only
# CHECK: COMPILED else_only
# CHECK: COMPILED if_else
# CHECK: COMPILED released_input
# CHECK: COMPILED elif_chain
# CHECK: COMPILED elif_gap
# CHECK: COMPILED nested_if
# CHECK: COMPILED sibling_ifs
# CHECK: COMPILED nested_def
# CHECK: COMPILED loop_wrapped
# CHECK: COMPILED external_use
# CHECK: COMPILED parent_and_branch
# CHECK: COMPILED attached_input
