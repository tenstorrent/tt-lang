# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Simple add kernel with float32 data type.

Tests that float32 tensors are properly handled through the layout derivation
path (TTNNLayoutAttr -> page size calculation).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

from ttlang import make_circular_buffer_like, ttl
from ttlang.operators import copy
from ttlang.ttl_api import Program

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.kernel(grid=(1, 1))
def add_kernel_f32(lhs, rhs, out):
    lhs_cb = make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
            result = l + r
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with lhs_cb.reserve():
            tx_lhs = copy(lhs[0, 0], lhs_cb)
            tx_lhs.wait()

        with rhs_cb.reserve():
            tx_rhs = copy(rhs[0, 0], rhs_cb)
            tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait():
            tx = copy(out_cb, out[0, 0])
            tx.wait()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# =============================================================================
# Initial IR Checks - Verify float32 layout attributes
# =============================================================================

# CHECK: #ttnn.buffer_type<l1>
# CHECK: #ttnn_layout = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, f32>{{.*}}>

# CHECK-LABEL: func.func @add_compute
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: %arg0: tensor<{{[^>]+}}!ttcore.tile<32x32, f32>, #ttnn_layout>
# CHECK-SAME: %arg1: tensor<{{[^>]+}}!ttcore.tile<32x32, f32>, #ttnn_layout>
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>}

# CHECK-LABEL: func.func @dm_write
# CHECK-SAME: %arg0: tensor<{{[^>]+}}!ttcore.tile<32x32, f32>, #ttnn_layout>
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [2], ttl.kernel_thread = #ttkernel.thread<noc>}


if __name__ == "__main__":
    import torch

    print("=== Float32 Add Kernel Test ===")

    device = ttnn.open_device(device_id=0)

    try:
        lhs_torch = torch.full((32, 32), 2.0, dtype=torch.float32)
        rhs_torch = torch.full((32, 32), 3.0, dtype=torch.float32)
        out_torch = torch.zeros((32, 32), dtype=torch.float32)

        lhs = ttnn.from_torch(
            lhs_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rhs = ttnn.from_torch(
            rhs_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        rhs = ttnn.to_memory_config(rhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling float32 add kernel...")
        add_kernel_f32(lhs, rhs, out)

        print("=== Float32 Add Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
