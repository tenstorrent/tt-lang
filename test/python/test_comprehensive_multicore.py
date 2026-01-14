# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive multicore test combining multiple features:
- 2MB DRAM inputs (a, b) + L1 input (c)
- 2MB DRAM outputs (out1, out2) + L1 output (out3)
- 8x8 multicore grid with dynamic indexing via core(dims=2)
- 4x4 CB shape with buffer_factor=2
- 20 fused ops using bounded operations
- Random inputs
"""

import pytest
import torch

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")

from ttlang import ttl

TILE_SIZE = 32
GRID_ROWS = 8
GRID_COLS = 8
CB_ROWS = 4
CB_COLS = 4

# 8x8 grid * 4x4 CB = 32x32 tiles = 1024x1024 elements = 2MB per tensor
TENSOR_SHAPE = (GRID_ROWS * CB_ROWS * TILE_SIZE, GRID_COLS * CB_COLS * TILE_SIZE)


@ttl.kernel(grid=(GRID_ROWS, GRID_COLS))
def comprehensive_kernel(a, b, c, out1, out2, out3):
    """
    Multicore kernel with 20 fused ops across 3 outputs.
    - out1 = f(a, b): 7 ops
    - out2 = g(b, c): 7 ops
    - out3 = h(a, c): 6 ops
    """
    # Circular buffers: 4x4 shape, buffer_factor=2
    a_cb = ttl.make_circular_buffer_like(a, shape=(CB_ROWS, CB_COLS), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(CB_ROWS, CB_COLS), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(CB_ROWS, CB_COLS), buffer_factor=2)
    out1_cb = ttl.make_circular_buffer_like(
        out1, shape=(CB_ROWS, CB_COLS), buffer_factor=2
    )
    out2_cb = ttl.make_circular_buffer_like(
        out2, shape=(CB_ROWS, CB_COLS), buffer_factor=2
    )
    out3_cb = ttl.make_circular_buffer_like(
        out3, shape=(CB_ROWS, CB_COLS), buffer_factor=2
    )

    @ttl.compute()
    def fused_compute():
        with (
            a_cb.wait() as av,
            b_cb.wait() as bv,
            c_cb.wait() as cv,
            out1_cb.reserve() as o1,
            out2_cb.reserve() as o2,
            out3_cb.reserve() as o3,
        ):
            # out1 = f(a, b): 7 ops
            v1 = ttl.math.sigmoid(av)  # 1
            v1 = ttl.math.tanh(v1)  # 2
            v1 = v1 + bv  # 3
            v1 = ttl.math.sigmoid(v1)  # 4
            v1 = ttl.math.tanh(v1)  # 5
            v1 = ttl.math.abs(v1)  # 6
            v1 = ttl.math.relu(v1)  # 7
            o1.store(v1)

            # out2 = g(b, c): 7 ops
            v2 = ttl.math.tanh(bv)  # 8
            v2 = ttl.math.sigmoid(v2)  # 9
            v2 = v2 + cv  # 10
            v2 = ttl.math.tanh(v2)  # 11
            v2 = ttl.math.neg(v2)  # 12
            v2 = ttl.math.abs(v2)  # 13
            v2 = ttl.math.sigmoid(v2)  # 14
            o2.store(v2)

            # out3 = h(a, c): 6 ops
            v3 = ttl.math.relu(av)  # 15
            v3 = ttl.math.sigmoid(v3)  # 16
            v3 = v3 + cv  # 17
            v3 = ttl.math.tanh(v3)  # 18
            v3 = ttl.math.abs(v3)  # 19
            v3 = ttl.math.sigmoid(v3)  # 20
            o3.store(v3)

    @ttl.datamovement()
    def dm_read():
        with (
            a_cb.reserve() as a_blk,
            b_cb.reserve() as b_blk,
            c_cb.reserve() as c_blk,
        ):
            x, y = ttl.core(dims=2)
            row = y * CB_ROWS
            col = x * CB_COLS
            tx_a = ttl.copy(a[row : row + CB_ROWS, col : col + CB_COLS], a_blk)
            tx_b = ttl.copy(b[row : row + CB_ROWS, col : col + CB_COLS], b_blk)
            tx_c = ttl.copy(c[row : row + CB_ROWS, col : col + CB_COLS], c_blk)
            tx_a.wait()
            tx_b.wait()
            tx_c.wait()

    @ttl.datamovement()
    def dm_write():
        with (
            out1_cb.wait() as o1_blk,
            out2_cb.wait() as o2_blk,
            out3_cb.wait() as o3_blk,
        ):
            x, y = ttl.core(dims=2)
            row = y * CB_ROWS
            col = x * CB_COLS
            tx1 = ttl.copy(o1_blk, out1[row : row + CB_ROWS, col : col + CB_COLS])
            tx2 = ttl.copy(o2_blk, out2[row : row + CB_ROWS, col : col + CB_COLS])
            tx3 = ttl.copy(o3_blk, out3[row : row + CB_ROWS, col : col + CB_COLS])
            tx1.wait()
            tx2.wait()
            tx3.wait()

    return ttl.Program(fused_compute, dm_read, dm_write)(a, b, c, out1, out2, out3)


def compute_expected(a, b, c):
    """Compute expected outputs using torch (matching kernel ops)."""
    a_f, b_f, c_f = a.float(), b.float(), c.float()

    # out1 = f(a, b): 7 ops
    v1 = torch.sigmoid(a_f)
    v1 = torch.tanh(v1)
    v1 = v1 + b_f
    v1 = torch.sigmoid(v1)
    v1 = torch.tanh(v1)
    v1 = torch.abs(v1)
    exp1 = torch.relu(v1)

    # out2 = g(b, c): 7 ops
    v2 = torch.tanh(b_f)
    v2 = torch.sigmoid(v2)
    v2 = v2 + c_f
    v2 = torch.tanh(v2)
    v2 = -v2
    v2 = torch.abs(v2)
    exp2 = torch.sigmoid(v2)

    # out3 = h(a, c): 6 ops
    v3 = torch.relu(a_f)
    v3 = torch.sigmoid(v3)
    v3 = v3 + c_f
    v3 = torch.tanh(v3)
    v3 = torch.abs(v3)
    exp3 = torch.sigmoid(v3)

    return exp1, exp2, exp3


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_comprehensive_multicore(device):
    """Test comprehensive multicore kernel with mixed DRAM/L1 tensors."""
    height, width = TENSOR_SHAPE
    tensor_size_mb = height * width * 2 / (1024 * 1024)

    # Random inputs (small values for bounded ops)
    a_torch = torch.rand(TENSOR_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    b_torch = torch.rand(TENSOR_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    c_torch = torch.rand(TENSOR_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    out1_torch = torch.zeros(TENSOR_SHAPE, dtype=torch.bfloat16)
    out2_torch = torch.zeros(TENSOR_SHAPE, dtype=torch.bfloat16)
    out3_torch = torch.zeros(TENSOR_SHAPE, dtype=torch.bfloat16)

    # Compute expected results
    exp1, exp2, exp3 = compute_expected(a_torch, b_torch, c_torch)

    # DRAM inputs: a, b
    a = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        b_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # L1 input: c
    c = ttnn.from_torch(
        c_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c = ttnn.to_memory_config(c, memory_config=ttnn.L1_MEMORY_CONFIG)

    # DRAM outputs: out1, out2
    out1 = ttnn.from_torch(
        out1_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out2 = ttnn.from_torch(
        out2_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # L1 output: out3
    out3 = ttnn.from_torch(
        out3_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out3 = ttnn.to_memory_config(out3, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel
    comprehensive_kernel(a, b, c, out1, out2, out3)

    # Verify grid_size
    x_size, y_size = ttl.grid_size(dims=2)
    assert (x_size, y_size) == (
        GRID_COLS,
        GRID_ROWS,
    ), f"grid_size mismatch: got ({x_size}, {y_size}), expected ({GRID_COLS}, {GRID_ROWS})"

    # Verify results
    result1 = ttnn.to_torch(out1)
    result2 = ttnn.to_torch(out2)
    result3 = ttnn.to_torch(out3)

    assert torch.allclose(
        result1.float(), exp1, rtol=0.05, atol=0.1
    ), f"out1 mismatch: max diff = {(result1.float() - exp1).abs().max().item()}"
    assert torch.allclose(
        result2.float(), exp2, rtol=0.05, atol=0.1
    ), f"out2 mismatch: max diff = {(result2.float() - exp2).abs().max().item()}"
    assert torch.allclose(
        result3.float(), exp3, rtol=0.05, atol=0.1
    ), f"out3 mismatch: max diff = {(result3.float() - exp3).abs().max().item()}"


if __name__ == "__main__":
    import sys

    if not TTNN_AVAILABLE:
        print("TTNN not available - skipping tests")
        sys.exit(0)

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
