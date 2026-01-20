# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: Large DRAM tensors can be streamed through small CBs.
# This tests that CB size is computed from block shape, not tensor volume.
# Total data: ~800MB (4 tensors * 200MB each)
# CB size: 8 tiles * 2KB = 16KB per CB

import torch
import ttnn
import ttl
from ttlang_test_utils import to_dram


@ttl.kernel(grid=(1, 1))
def fused_mul_add_streaming(a, b, c, y):
    """
    Compute y = a * b + c by streaming large DRAM tensors through small CBs.
    """
    row_tiles = 2
    col_tiles = 2

    rows = a.shape[0] // 32 // row_tiles
    cols = a.shape[1] // 32 // col_tiles

    a_cb = ttl.make_circular_buffer_like(
        a, shape=(row_tiles, col_tiles), buffer_factor=2
    )
    b_cb = ttl.make_circular_buffer_like(
        b, shape=(row_tiles, col_tiles), buffer_factor=2
    )
    c_cb = ttl.make_circular_buffer_like(
        c, shape=(row_tiles, col_tiles), buffer_factor=2
    )
    y_cb = ttl.make_circular_buffer_like(
        y, shape=(row_tiles, col_tiles), buffer_factor=2
    )

    @ttl.compute()
    def compute_kernel():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    a_cb.wait() as a_block,
                    b_cb.wait() as b_block,
                    c_cb.wait() as c_block,
                    y_cb.reserve() as y_block,
                ):
                    y_block.store(a_block * b_block + c_block)

    @ttl.datamovement()
    def read_kernel():
        for row in range(rows):
            for col in range(cols):
                with (
                    a_cb.reserve() as a_blk,
                    b_cb.reserve() as b_blk,
                    c_cb.reserve() as c_blk,
                ):
                    tx_a = ttl.copy(
                        a[
                            row * row_tiles : (row + 1) * row_tiles,
                            col * col_tiles : (col + 1) * col_tiles,
                        ],
                        a_blk,
                    )
                    tx_b = ttl.copy(
                        b[
                            row * row_tiles : (row + 1) * row_tiles,
                            col * col_tiles : (col + 1) * col_tiles,
                        ],
                        b_blk,
                    )
                    tx_c = ttl.copy(
                        c[
                            row * row_tiles : (row + 1) * row_tiles,
                            col * col_tiles : (col + 1) * col_tiles,
                        ],
                        c_blk,
                    )
                    tx_a.wait()
                    tx_b.wait()
                    tx_c.wait()

    @ttl.datamovement()
    def write_kernel():
        for row in range(rows):
            for col in range(cols):
                with y_cb.wait() as y_blk:
                    tx = ttl.copy(
                        y_blk,
                        y[
                            row * row_tiles : (row + 1) * row_tiles,
                            col * col_tiles : (col + 1) * col_tiles,
                        ],
                    )
                    tx.wait()

    return ttl.Program(compute_kernel, read_kernel, write_kernel)(a, b, c, y)


# CHECK: Testing Large DRAM Streaming
print("=== Testing Large DRAM Streaming ===")

device = ttnn.open_device(device_id=0)

try:
    # 10240 x 10240 = 104,857,600 elements * 2 bytes = ~200MB per tensor
    shape = (10240, 10240)
    tensor_size_mb = shape[0] * shape[1] * 2 / (1024 * 1024)
    print(f"Size per tensor: {tensor_size_mb:.1f} MB")
    # CHECK: Size per tensor: 200.0 MB

    a_torch = torch.rand(shape, dtype=torch.bfloat16)
    b_torch = torch.rand(shape, dtype=torch.bfloat16)
    c_torch = torch.rand(shape, dtype=torch.bfloat16)
    y_torch = torch.zeros(shape, dtype=torch.bfloat16)
    expected = a_torch * b_torch + c_torch

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    c = to_dram(c_torch, device)
    y = to_dram(y_torch, device)

    fused_mul_add_streaming(a, b, c, y)
    result = ttnn.to_torch(y)

    if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
        print("\nPASS: Large DRAM streaming works!")
        # CHECK: PASS
    else:
        max_diff = (result - expected).abs().max().item()
        print(f"\nFAIL: Max difference = {max_diff}")

finally:
    ttnn.close_device(device)

print("\n=== Large DRAM Streaming Test Complete ===")
# CHECK: Test Complete
