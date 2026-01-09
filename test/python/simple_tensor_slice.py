# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Tensor slice test - verifies tensor[row, col] creates ttl.tensor_slice ops.

Uses 64x64 tensors (2x2 tiles of 32x32) and accesses specific tiles
via indices to test the tensor slice infrastructure.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

from ttlang import ttl, make_circular_buffer_like
from ttlang.ttl_api import Program
from ttlang.operators import copy

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.kernel(grid=(1, 1))
def tile_index_kernel(lhs, rhs, out):
    """Kernel that accesses specific tiles via indices."""
    lhs_cb = make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        # Access tile at [0, 1]
        lhs_cb.reserve()
        tx_lhs = copy(lhs[0, 1], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        # Access tile at [0, 2]
        rhs_cb.reserve()
        tx_rhs = copy(rhs[0, 2], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        # Write to tile at [0, 3]
        out_cb.wait()
        tx = copy(out_cb, out[0, 3])
        tx.wait()
        out_cb.pop()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# =============================================================================
# Initial IR Checks - Verify tensor_slice ops are generated
# =============================================================================

# CHECK-LABEL: func.func @dm_read

# First tensor slice at [0, 1]
# CHECK: %[[C1:.+]] = arith.constant 1 : index
# CHECK: ttl.tensor_slice %arg0[%{{.+}}, %[[C1]]]
# CHECK: ttl.copy %{{.+}}, %{{.+}} : (tensor<{{.*}}>) -> !ttl.transfer_handle<read>

# Second tensor slice at [0, 2]
# CHECK: %[[C2:.+]] = arith.constant 2 : index
# CHECK: ttl.tensor_slice %arg1[%{{.+}}, %[[C2]]]
# CHECK: ttl.copy %{{.+}}, %{{.+}} : (tensor<{{.*}}>) -> !ttl.transfer_handle<read>

# CHECK-LABEL: func.func @dm_write

# Output tensor slice at [0, 3]
# CHECK: %[[C3:.+]] = arith.constant 3 : index
# CHECK: ttl.tensor_slice %arg0[%{{.+}}, %[[C3]]]
# CHECK: ttl.copy %{{.+}}, %{{.+}} : ({{.*}}tensor<{{.*}}>) -> !ttl.transfer_handle<write>

# =============================================================================
# C++ Kernel Checks - Verify correct tile offsets in NOC ops
# =============================================================================

# CHECK-CPP: // dm_read
# CHECK-CPP: void kernel_main()

# Constants are hoisted to the top of the function
# Second read tile index (2) is declared before first read tile index (1)
# CHECK-CPP: int32_t [[V2:[a-z0-9]+]] = 2;
# CHECK-CPP: int32_t [[V1:[a-z0-9]+]] = 1;

# First read at tile [0, 1]: offset = 1
# CHECK-CPP: noc_async_read_tile([[V1]],

# Second read at tile [0, 2]: offset = 2
# CHECK-CPP: noc_async_read_tile([[V2]],

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()

# Write at tile [0, 3]: offset = 3
# CHECK-CPP: int32_t [[V3:[a-z0-9]+]] = 3;
# CHECK-CPP: noc_async_write_tile([[V3]],


if __name__ == "__main__":
    import torch

    print("=== Tile Index Kernel Test ===")

    device = ttnn.open_device(device_id=0)

    try:
        # 64x64 = 2x2 tiles of 32x32
        lhs_torch = torch.full((64, 64), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((64, 64), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        lhs = ttnn.from_torch(
            lhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rhs = ttnn.from_torch(
            rhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        rhs = ttnn.to_memory_config(rhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling tile index kernel (64x64 = 2x2 tiles)...")
        tile_index_kernel(lhs, rhs, out)

        print("=== Tile Index Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
