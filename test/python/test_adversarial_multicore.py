# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Adversarial multicore test designed to stress compiler optimizations.

Evil features:
- Non-square tensors and grid (8x6 grid, 1024x768 DRAM, 256x192 L1)
- Prime-ish CB dimensions (4x3 for DRAM)
- Shared subexpressions across outputs (tests CSE)
- Variable reuse/shadowing
- Interleaved operations in non-obvious order
- buffer_factor=1 (no double buffering) for some CBs
- Single-iteration vs multi-iteration loops mixed
- Binary ops with operands in different orders
- Deep nesting of with statements
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

# Non-square grid
GRID_ROWS = 8
GRID_COLS = 6

# DRAM: 4x3 tiles per core = 32x18 tiles total = 1024x576 elements
DRAM_CB_ROWS = 4
DRAM_CB_COLS = 3
DRAM_SHAPE = (
    GRID_ROWS * DRAM_CB_ROWS * TILE_SIZE,  # 1024
    GRID_COLS * DRAM_CB_COLS * TILE_SIZE,  # 576
)

# L1: 1x1 tile per core = 8x6 tiles total = 256x192 elements
L1_CB_ROWS = 1
L1_CB_COLS = 1
L1_SHAPE = (
    GRID_ROWS * L1_CB_ROWS * TILE_SIZE,  # 256
    GRID_COLS * L1_CB_COLS * TILE_SIZE,  # 192
)


@ttl.kernel(grid=(8, 6))
def adversarial_kernel(a, b, c, d, out1, out2, out3, out4):
    """
    Adversarial kernel with 4 inputs and 4 outputs.
    Designed to stress compiler with weird patterns.
    """
    # Mix of buffer_factor=1 and buffer_factor=2
    a_cb = ttl.make_circular_buffer_like(a, shape=(4, 3), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(4, 3), buffer_factor=1)  # No double buffer!
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    d_cb = ttl.make_circular_buffer_like(d, shape=(1, 1), buffer_factor=1)  # No double buffer!

    out1_cb = ttl.make_circular_buffer_like(out1, shape=(4, 3), buffer_factor=2)
    out2_cb = ttl.make_circular_buffer_like(out2, shape=(4, 3), buffer_factor=1)
    out3_cb = ttl.make_circular_buffer_like(out3, shape=(1, 1), buffer_factor=2)
    out4_cb = ttl.make_circular_buffer_like(out4, shape=(1, 1), buffer_factor=1)

    @ttl.compute()
    def evil_compute():
        # Load L1 tiles first (broadcast pattern)
        with c_cb.wait() as cv:
            with d_cb.wait() as dv:
                # Shared subexpression: sigmoid(cv) used in multiple places
                shared = ttl.math.sigmoid(cv)

                # Process DRAM in weird nested loop (4 rows x 3 cols = 12 iterations)
                for row_iter in range(4):
                    for col_iter in range(3):
                        with a_cb.wait() as av:
                            with b_cb.wait() as bv:
                                with out1_cb.reserve() as o1:
                                    with out2_cb.reserve() as o2:
                                        # out1: uses shared subexpr + a + b
                                        # Variable shadowing: reuse 'v' name
                                        v = ttl.math.tanh(av)
                                        v = v + bv  # a op b
                                        v = v + shared  # + broadcast L1
                                        v = ttl.math.sigmoid(v)
                                        v = ttl.math.abs(v)
                                        o1.store(v)

                                        # out2: operands in reverse order
                                        v = ttl.math.tanh(bv)
                                        v = shared + v  # reverse order: L1 + DRAM
                                        v = ttl.math.relu(v)
                                        v = ttl.math.neg(v)
                                        v = ttl.math.abs(v)
                                        o2.store(v)

                # L1 outputs: use both cv and dv
                with out3_cb.reserve() as o3:
                    # out3: chain using shared + dv
                    v = shared + dv  # L1 + L1
                    v = ttl.math.tanh(v)
                    v = ttl.math.sigmoid(v)
                    v = ttl.math.relu(v)
                    o3.store(v)

                with out4_cb.reserve() as o4:
                    # out4: different combination
                    v = ttl.math.tanh(cv)
                    v = v + dv
                    v = ttl.math.abs(v)
                    v = ttl.math.sigmoid(v)
                    o4.store(v)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)

        # Load L1 tiles first
        with c_cb.reserve() as c_blk:
            tx = ttl.copy(c[y, x], c_blk)
            tx.wait()

        with d_cb.reserve() as d_blk:
            tx = ttl.copy(d[y, x], d_blk)
            tx.wait()

        # Load DRAM tiles in nested loop
        for local_r in range(4):
            for local_c in range(3):
                with a_cb.reserve() as a_blk:
                    dram_row = y * 4 + local_r
                    dram_col = x * 3 + local_c
                    tx = ttl.copy(a[dram_row, dram_col], a_blk)
                    tx.wait()

                with b_cb.reserve() as b_blk:
                    dram_row = y * 4 + local_r
                    dram_col = x * 3 + local_c
                    tx = ttl.copy(b[dram_row, dram_col], b_blk)
                    tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        # Write DRAM tiles
        for local_r in range(4):
            for local_c in range(3):
                with out1_cb.wait() as o1_blk:
                    dram_row = y * 4 + local_r
                    dram_col = x * 3 + local_c
                    tx = ttl.copy(o1_blk, out1[dram_row, dram_col])
                    tx.wait()

                with out2_cb.wait() as o2_blk:
                    dram_row = y * 4 + local_r
                    dram_col = x * 3 + local_c
                    tx = ttl.copy(o2_blk, out2[dram_row, dram_col])
                    tx.wait()

        # Write L1 tiles
        with out3_cb.wait() as o3_blk:
            tx = ttl.copy(o3_blk, out3[y, x])
            tx.wait()

        with out4_cb.wait() as o4_blk:
            tx = ttl.copy(o4_blk, out4[y, x])
            tx.wait()

    return ttl.Program(evil_compute, dm_read, dm_write)(a, b, c, d, out1, out2, out3, out4)


def compute_expected(a, b, c, d):
    """Compute expected outputs matching the evil kernel."""
    a_f, b_f = a.float(), b.float()
    c_f, d_f = c.float(), d.float()

    # Broadcast c to DRAM size (256x192 -> 1024x576, ratio 4x3)
    c_broadcast = c_f.repeat_interleave(4, dim=0).repeat_interleave(3, dim=1)

    # shared = sigmoid(c), broadcast to DRAM size
    shared_broadcast = torch.sigmoid(c_broadcast)

    # out1 = abs(sigmoid(tanh(a) + b + shared))
    v1 = torch.tanh(a_f)
    v1 = v1 + b_f
    v1 = v1 + shared_broadcast
    v1 = torch.sigmoid(v1)
    exp1 = torch.abs(v1)

    # out2 = abs(neg(relu(shared + tanh(b))))
    v2 = torch.tanh(b_f)
    v2 = shared_broadcast + v2
    v2 = torch.relu(v2)
    v2 = -v2
    exp2 = torch.abs(v2)

    # shared = sigmoid(c), L1 size
    shared_l1 = torch.sigmoid(c_f)

    # out3 = relu(sigmoid(tanh(shared + d)))
    v3 = shared_l1 + d_f
    v3 = torch.tanh(v3)
    v3 = torch.sigmoid(v3)
    exp3 = torch.relu(v3)

    # out4 = sigmoid(abs(tanh(c) + d))
    v4 = torch.tanh(c_f)
    v4 = v4 + d_f
    v4 = torch.abs(v4)
    exp4 = torch.sigmoid(v4)

    return exp1, exp2, exp3, exp4


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_adversarial_multicore(device):
    """Test adversarial kernel designed to break compiler optimizations."""
    print(f"DRAM shape: {DRAM_SHAPE} (non-square)")
    print(f"L1 shape: {L1_SHAPE} (non-square)")
    print(f"Grid: {GRID_ROWS}x{GRID_COLS} (non-square)")

    # Random inputs
    a_torch = torch.rand(DRAM_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    b_torch = torch.rand(DRAM_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    c_torch = torch.rand(L1_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0
    d_torch = torch.rand(L1_SHAPE, dtype=torch.bfloat16) * 2.0 - 1.0

    out1_torch = torch.zeros(DRAM_SHAPE, dtype=torch.bfloat16)
    out2_torch = torch.zeros(DRAM_SHAPE, dtype=torch.bfloat16)
    out3_torch = torch.zeros(L1_SHAPE, dtype=torch.bfloat16)
    out4_torch = torch.zeros(L1_SHAPE, dtype=torch.bfloat16)

    # Expected
    exp1, exp2, exp3, exp4 = compute_expected(a_torch, b_torch, c_torch, d_torch)

    # DRAM tensors
    a = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out1 = ttnn.from_torch(out1_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out2 = ttnn.from_torch(out2_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # L1 tensors
    c = ttnn.from_torch(c_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    c = ttnn.to_memory_config(c, memory_config=ttnn.L1_MEMORY_CONFIG)

    d = ttnn.from_torch(d_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    d = ttnn.to_memory_config(d, memory_config=ttnn.L1_MEMORY_CONFIG)

    out3 = ttnn.from_torch(out3_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out3 = ttnn.to_memory_config(out3, memory_config=ttnn.L1_MEMORY_CONFIG)

    out4 = ttnn.from_torch(out4_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out4 = ttnn.to_memory_config(out4, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel
    adversarial_kernel(a, b, c, d, out1, out2, out3, out4)

    # Verify grid_size
    x_size, y_size = ttl.grid_size(dims=2)
    assert (x_size, y_size) == (GRID_COLS, GRID_ROWS), (
        f"grid_size mismatch: got ({x_size}, {y_size}), expected ({GRID_COLS}, {GRID_ROWS})"
    )

    # Verify results
    result1 = ttnn.to_torch(out1)
    result2 = ttnn.to_torch(out2)
    result3 = ttnn.to_torch(out3)
    result4 = ttnn.to_torch(out4)

    assert torch.allclose(result1.float(), exp1, rtol=0.05, atol=0.1), (
        f"out1 mismatch: max diff = {(result1.float() - exp1).abs().max().item()}"
    )
    assert torch.allclose(result2.float(), exp2, rtol=0.05, atol=0.1), (
        f"out2 mismatch: max diff = {(result2.float() - exp2).abs().max().item()}"
    )
    assert torch.allclose(result3.float(), exp3, rtol=0.05, atol=0.1), (
        f"out3 mismatch: max diff = {(result3.float() - exp3).abs().max().item()}"
    )
    assert torch.allclose(result4.float(), exp4, rtol=0.05, atol=0.1), (
        f"out4 mismatch: max diff = {(result4.float() - exp4).abs().max().item()}"
    )


if __name__ == "__main__":
    import sys

    if not TTNN_AVAILABLE:
        print("TTNN not available - skipping tests")
        sys.exit(0)

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
