# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Adversarial multicore test designed to stress compiler optimizations.

Evil features:
- Non-square grid (8x6) and tensors (512x384)
- 2x2 CB shape with multi-tile blocks
- Variable reuse/shadowing
- Interleaved operations in non-obvious order
- buffer_factor=1 (no double buffering) for some CBs
- Binary ops with operands in different orders
- Deep nesting of with statements (8 levels)
- 4 inputs, 4 outputs
"""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram

import ttl

TILE_SIZE = 32

# Non-square grid
GRID_ROWS = 8
GRID_COLS = 6

# All tensors same size: 2x2 tiles per core = 512x384 total
CB_ROWS = 2
CB_COLS = 2
TENSOR_SHAPE = (
    GRID_ROWS * CB_ROWS * TILE_SIZE,  # 512
    GRID_COLS * CB_COLS * TILE_SIZE,  # 384
)


@ttl.kernel(grid=(8, 6))
def adversarial_kernel(a, b, c, d, out1, out2, out3, out4):
    """
    Adversarial kernel with 4 inputs and 4 outputs.
    Designed to stress compiler with weird patterns.
    All tensors same size, 2x2 CB blocks.
    """
    # Mix of buffer_factor=1 and buffer_factor=2
    a_cb = ttl.make_circular_buffer_like(a, shape=(2, 2), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(
        b, shape=(2, 2), buffer_factor=1
    )  # No double buffer!
    c_cb = ttl.make_circular_buffer_like(c, shape=(2, 2), buffer_factor=2)
    d_cb = ttl.make_circular_buffer_like(
        d, shape=(2, 2), buffer_factor=1
    )  # No double buffer!

    out1_cb = ttl.make_circular_buffer_like(out1, shape=(2, 2), buffer_factor=2)
    out2_cb = ttl.make_circular_buffer_like(out2, shape=(2, 2), buffer_factor=1)
    out3_cb = ttl.make_circular_buffer_like(out3, shape=(2, 2), buffer_factor=2)
    out4_cb = ttl.make_circular_buffer_like(out4, shape=(2, 2), buffer_factor=1)

    @ttl.compute()
    def evil_compute():
        # Deep nesting with all inputs
        with a_cb.wait() as av:
            with b_cb.wait() as bv:
                with c_cb.wait() as cv:
                    with d_cb.wait() as dv:
                        with out1_cb.reserve() as o1:
                            with out2_cb.reserve() as o2:
                                # out1: uses sigmoid(c) + a + b
                                # Variable shadowing: reuse 'v' name
                                v = ttl.math.tanh(av)
                                v = v + bv  # a op b
                                v = v + ttl.math.sigmoid(cv)  # inline sigmoid
                                v = ttl.math.sigmoid(v)
                                v = ttl.math.abs(v)
                                o1.store(v)

                                # out2: operands in reverse order
                                v = ttl.math.tanh(bv)
                                v = ttl.math.sigmoid(cv) + v  # reverse order
                                v = ttl.math.relu(v)
                                v = ttl.math.neg(v)
                                v = ttl.math.abs(v)
                                o2.store(v)

                        with out3_cb.reserve() as o3:
                            with out4_cb.reserve() as o4:
                                # out3: chain using sigmoid(c) + d
                                v = ttl.math.sigmoid(cv) + dv
                                v = ttl.math.tanh(v)
                                v = ttl.math.sigmoid(v)
                                v = ttl.math.relu(v)
                                o3.store(v)

                                # out4: different combination
                                v = ttl.math.tanh(cv)
                                v = v + dv
                                v = ttl.math.abs(v)
                                v = ttl.math.sigmoid(v)
                                o4.store(v)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        row = y * 2
        col = x * 2

        with a_cb.reserve() as a_blk:
            tx = ttl.copy(a[row : row + 2, col : col + 2], a_blk)
            tx.wait()

        with b_cb.reserve() as b_blk:
            tx = ttl.copy(b[row : row + 2, col : col + 2], b_blk)
            tx.wait()

        with c_cb.reserve() as c_blk:
            tx = ttl.copy(c[row : row + 2, col : col + 2], c_blk)
            tx.wait()

        with d_cb.reserve() as d_blk:
            tx = ttl.copy(d[row : row + 2, col : col + 2], d_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)
        row = y * 2
        col = x * 2

        with out1_cb.wait() as o1_blk:
            tx = ttl.copy(o1_blk, out1[row : row + 2, col : col + 2])
            tx.wait()

        with out2_cb.wait() as o2_blk:
            tx = ttl.copy(o2_blk, out2[row : row + 2, col : col + 2])
            tx.wait()

        with out3_cb.wait() as o3_blk:
            tx = ttl.copy(o3_blk, out3[row : row + 2, col : col + 2])
            tx.wait()

        with out4_cb.wait() as o4_blk:
            tx = ttl.copy(o4_blk, out4[row : row + 2, col : col + 2])
            tx.wait()


def compute_expected(a, b, c, d):
    """Compute expected outputs matching the evil kernel."""
    a_f, b_f = a.float(), b.float()
    c_f, d_f = c.float(), d.float()

    # shared = sigmoid(c)
    shared = torch.sigmoid(c_f)

    # out1 = abs(sigmoid(tanh(a) + b + shared))
    v1 = torch.tanh(a_f)
    v1 = v1 + b_f
    v1 = v1 + shared
    v1 = torch.sigmoid(v1)
    exp1 = torch.abs(v1)

    # out2 = abs(neg(relu(shared + tanh(b))))
    v2 = torch.tanh(b_f)
    v2 = shared + v2
    v2 = torch.relu(v2)
    v2 = -v2
    exp2 = torch.abs(v2)

    # out3 = relu(sigmoid(tanh(shared + d)))
    v3 = shared + d_f
    v3 = torch.tanh(v3)
    v3 = torch.sigmoid(v3)
    exp3 = torch.relu(v3)

    # out4 = sigmoid(abs(tanh(c) + d))
    v4 = torch.tanh(c_f)
    v4 = v4 + d_f
    v4 = torch.abs(v4)
    exp4 = torch.sigmoid(v4)

    return exp1, exp2, exp3, exp4


def test_adversarial_multicore(device):
    """Test adversarial kernel designed to break compiler optimizations."""
    # Random inputs
    a_torch = torch.rand(TENSOR_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    b_torch = torch.rand(TENSOR_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    c_torch = torch.rand(TENSOR_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    d_torch = torch.rand(TENSOR_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    out1_torch = torch.zeros(TENSOR_SHAPE, dtype=torch.bfloat16)
    out2_torch = torch.zeros(TENSOR_SHAPE, dtype=torch.bfloat16)
    out3_torch = torch.zeros(TENSOR_SHAPE, dtype=torch.bfloat16)
    out4_torch = torch.zeros(TENSOR_SHAPE, dtype=torch.bfloat16)

    exp1, exp2, exp3, exp4 = compute_expected(a_torch, b_torch, c_torch, d_torch)

    # All tensors in DRAM
    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    c = to_dram(c_torch, device)
    d = to_dram(d_torch, device)
    out1 = to_dram(out1_torch, device)
    out2 = to_dram(out2_torch, device)
    out3 = to_dram(out3_torch, device)
    out4 = to_dram(out4_torch, device)

    adversarial_kernel(a, b, c, d, out1, out2, out3, out4)

    # Verify grid_size
    x_size, y_size = ttl.grid_size(dims=2)
    assert (x_size, y_size) == (GRID_COLS, GRID_ROWS)

    # Verify results
    result1 = ttnn.to_torch(out1)
    result2 = ttnn.to_torch(out2)
    result3 = ttnn.to_torch(out3)
    result4 = ttnn.to_torch(out4)

    assert torch.allclose(result1.float(), exp1, rtol=0.05, atol=0.1)
    assert torch.allclose(result2.float(), exp2, rtol=0.05, atol=0.1)
    assert torch.allclose(result3.float(), exp3, rtol=0.05, atol=0.1)
    assert torch.allclose(result4.float(), exp4, rtol=0.05, atol=0.1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
