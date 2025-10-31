# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

# Test MetalLayoutConfig and device shape computation.

from ttmlir.ir import *
from ttmlir.dialects import ttcore

from ttlang.layouts import MetalLayoutConfig, create_metal_layout, compute_device_shape

ctx = Context()

# Test: Create MetalLayoutConfig with L1 memory space
config = MetalLayoutConfig(
    logical_shape=[128, 128], grid=[2, 2], tiled=True, memory_space="L1"
)

layout = create_metal_layout(ctx, config)
print(layout)
# CHECK: #ttcore.metal_layout<logical_shape = 128x128

# Test: Compute device shape for grid [2,2] with shape [128,128]
# Expected: [2, 2, 2, 2] (grid_y, grid_x, shard_y, shard_x)
device_shape = compute_device_shape(layout, [2, 2], [128, 128], [32, 32])
print(f"Device shape: {device_shape}")
# CHECK: Device shape: [2, 2, 2, 2]

# Test: DRAM memory space
config_dram = MetalLayoutConfig(
    logical_shape=[64, 64], grid=[1, 1], tiled=True, memory_space="DRAM"
)
layout_dram = create_metal_layout(ctx, config_dram)
print(layout_dram)
# CHECK: dram>
