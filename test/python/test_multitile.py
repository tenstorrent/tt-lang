# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest for multi-tile fused kernel - verifies correct tile indexing across 2x2 tile grid.

Uses 64x64 tensors (2x2 tiles of 32x32) to test that:
1. The two-loop structure is generated correctly (compute loop, then pack loop)
2. tile_regs_commit/wait are placed OUTSIDE the math tile op loops
3. Linearized tile indices are computed correctly for CB operations
4. Fused binary + unary ops work correctly across multiple tiles
"""

import pytest
import torch
import ttnn
from ttlang import ttl


@ttl.kernel(grid=(1, 1))
def add_exp_multitile_kernel(lhs, rhs, out):
    """Fused add + exp kernel processing 2x2 tile grid (4 tiles total).

    Computes: out = exp(lhs + rhs)

    This tests the multitile fix with a fused binary (add) + unary (exp) operation,
    ensuring that both operations are executed in the compute loop before
    tile_regs_commit/wait, and pack_tile is in a separate loop after.
    """
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(2, 2), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def add_exp_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        sum_result = l + r
        exp_result = ttl.exp(sum_result)
        o.store(exp_result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[0:2, 0:2], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[0:2, 0:2], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_cb.pop()

    return ttl.Program(add_exp_compute, dm_read, dm_write)(lhs, rhs, out)


@pytest.mark.requires_ttnn
@pytest.mark.requires_device
def test_multitile_add_exp_correctness(ttnn_device):
    """
    Test that multi-tile fused add+exp kernel produces correct results.

    This test verifies:
    1. The kernel compiles successfully with 2x2 tile grid
    2. Fused binary (add) + unary (exp) operations work correctly
    3. The output matches expected exp(lhs + rhs)
    """
    device = ttnn_device

    # 64x64 = 2x2 tiles of 32x32
    # Use small values to avoid exp overflow
    lhs_torch = torch.full((64, 64), 0.5, dtype=torch.bfloat16)
    rhs_torch = torch.full((64, 64), 0.5, dtype=torch.bfloat16)
    out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
    expected = torch.exp(lhs_torch + rhs_torch)  # exp(1.0) = 2.718...

    # Create TTNN tensors on device
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

    # Move to L1 for kernel execution
    lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
    rhs = ttnn.to_memory_config(rhs, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run the kernel
    add_exp_multitile_kernel(lhs, rhs, out)

    # Verify results
    out_result = ttnn.to_torch(out)
    assert torch.allclose(
        out_result.float(), expected.float(), rtol=1e-2, atol=1e-2
    ), f"Output mismatch! Max error: {(out_result.float() - expected.float()).abs().max().item()}"


@pytest.mark.requires_ttnn
@pytest.mark.requires_device
def test_multitile_add_exp_varying_values(ttnn_device):
    """
    Test multi-tile add+exp with varying input values to ensure all tiles are processed.

    Uses different values in each quadrant to verify tile indexing is correct.
    """
    device = ttnn_device

    # Create tensors with different values in each 32x32 quadrant
    # Use small values to avoid exp overflow
    lhs_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
    lhs_torch[0:32, 0:32] = 0.1  # Top-left
    lhs_torch[0:32, 32:64] = 0.2  # Top-right
    lhs_torch[32:64, 0:32] = 0.3  # Bottom-left
    lhs_torch[32:64, 32:64] = 0.4  # Bottom-right

    rhs_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
    rhs_torch[0:32, 0:32] = 0.1
    rhs_torch[0:32, 32:64] = 0.2
    rhs_torch[32:64, 0:32] = 0.3
    rhs_torch[32:64, 32:64] = 0.4

    out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
    expected = torch.exp(lhs_torch + rhs_torch)

    # Create TTNN tensors
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

    add_exp_multitile_kernel(lhs, rhs, out)

    out_result = ttnn.to_torch(out)

    # Check each quadrant separately to identify which tile failed
    assert torch.allclose(
        out_result[0:32, 0:32].float(),
        expected[0:32, 0:32].float(),
        rtol=1e-2,
        atol=1e-2,
    ), f"Top-left quadrant mismatch: expected exp(0.2)={expected[0,0].item():.4f}, got {out_result[0,0].item():.4f}"
    assert torch.allclose(
        out_result[0:32, 32:64].float(),
        expected[0:32, 32:64].float(),
        rtol=1e-2,
        atol=1e-2,
    ), f"Top-right quadrant mismatch: expected exp(0.4)={expected[0,32].item():.4f}, got {out_result[0,32].item():.4f}"
    assert torch.allclose(
        out_result[32:64, 0:32].float(),
        expected[32:64, 0:32].float(),
        rtol=1e-2,
        atol=1e-2,
    ), f"Bottom-left quadrant mismatch: expected exp(0.6)={expected[32,0].item():.4f}, got {out_result[32,0].item():.4f}"
    assert torch.allclose(
        out_result[32:64, 32:64].float(),
        expected[32:64, 32:64].float(),
        rtol=1e-2,
        atol=1e-2,
    ), f"Bottom-right quadrant mismatch: expected exp(0.8)={expected[32,32].item():.4f}, got {out_result[32,32].item():.4f}"
