# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests loop unrolling correctness across different CB shapes and operation patterns.

Verifies numerical correctness with unrolling enabled for:
- Different CB grid shapes (1D, 2D, with and without remainder iterations)
- Different DST footprints (binary=3 regs, unary=2 regs, fused chains=3 regs)
- Edge cases: exact divisibility vs remainder loops

Key test cases:
- 1x4 tiles, binary op, unroll_factor=2 (evenly divisible)
- 1x5 tiles, binary op, unroll_factor=2 (remainder loop with 1 iteration)
- 2x2 tiles, unary op, unroll_factor=4 (high unroll, evenly divisible)
- 1x4 tiles, fused chain (add+mul+exp), unroll_factor=2 (3 DST regs)
"""

# REQUIRES: ttnn, tt-device
# RUN: %python %s

import torch
import ttnn
from ttlang import ttl


# =============================================================================
# Kernel generators for each operation type
# =============================================================================


def make_binary_kernel(cb_shape):
    """Binary addition kernel."""
    rows, cols = cb_shape

    @ttl.kernel(grid=(1, 1))
    def kernel(lhs, rhs, out):
        lhs_cb = ttl.make_circular_buffer_like(lhs, shape=cb_shape, buffer_factor=2)
        rhs_cb = ttl.make_circular_buffer_like(rhs, shape=cb_shape, buffer_factor=2)
        out_cb = ttl.make_circular_buffer_like(out, shape=cb_shape, buffer_factor=2)

        @ttl.compute()
        def compute_fn():
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
            lhs_blk = lhs_cb.reserve()
            tx_lhs = ttl.copy(lhs[0:rows, 0:cols], lhs_blk)
            tx_lhs.wait()
            lhs_cb.push()

            rhs_blk = rhs_cb.reserve()
            tx_rhs = ttl.copy(rhs[0:rows, 0:cols], rhs_blk)
            tx_rhs.wait()
            rhs_cb.push()

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            tx = ttl.copy(blk, out[0:rows, 0:cols])
            tx.wait()
            out_cb.pop()

        return ttl.Program(compute_fn, dm_read, dm_write)(lhs, rhs, out)

    return kernel


def make_unary_kernel(cb_shape):
    """Unary exp kernel."""
    rows, cols = cb_shape

    @ttl.kernel(grid=(1, 1))
    def kernel(inp, out):
        inp_cb = ttl.make_circular_buffer_like(inp, shape=cb_shape, buffer_factor=2)
        out_cb = ttl.make_circular_buffer_like(out, shape=cb_shape, buffer_factor=2)

        @ttl.compute()
        def compute_fn():
            x = inp_cb.wait()
            o = out_cb.reserve()
            result = ttl.math.exp(x)
            o.store(result)
            inp_cb.pop()
            out_cb.push()

        @ttl.datamovement()
        def dm_read():
            blk = inp_cb.reserve()
            tx = ttl.copy(inp[0:rows, 0:cols], blk)
            tx.wait()
            inp_cb.push()

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            tx = ttl.copy(blk, out[0:rows, 0:cols])
            tx.wait()
            out_cb.pop()

        return ttl.Program(compute_fn, dm_read, dm_write)(inp, out)

    return kernel


def make_fused_kernel(cb_shape):
    """Fused chain kernel: exp(l + r * l) - uses 3 DST regs."""
    rows, cols = cb_shape

    @ttl.kernel(grid=(1, 1))
    def kernel(lhs, rhs, out):
        lhs_cb = ttl.make_circular_buffer_like(lhs, shape=cb_shape, buffer_factor=2)
        rhs_cb = ttl.make_circular_buffer_like(rhs, shape=cb_shape, buffer_factor=2)
        out_cb = ttl.make_circular_buffer_like(out, shape=cb_shape, buffer_factor=2)

        @ttl.compute()
        def compute_fn():
            l = lhs_cb.wait()
            r = rhs_cb.wait()
            o = out_cb.reserve()
            # Fused chain: mul, add, exp - reuses DST registers, footprint=3
            result = ttl.math.exp(l + r * l)
            o.store(result)
            lhs_cb.pop()
            rhs_cb.pop()
            out_cb.push()

        @ttl.datamovement()
        def dm_read():
            lhs_blk = lhs_cb.reserve()
            tx_lhs = ttl.copy(lhs[0:rows, 0:cols], lhs_blk)
            tx_lhs.wait()
            lhs_cb.push()

            rhs_blk = rhs_cb.reserve()
            tx_rhs = ttl.copy(rhs[0:rows, 0:cols], rhs_blk)
            tx_rhs.wait()
            rhs_cb.push()

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            tx = ttl.copy(blk, out[0:rows, 0:cols])
            tx.wait()
            out_cb.pop()

        return ttl.Program(compute_fn, dm_read, dm_write)(lhs, rhs, out)

    return kernel


# =============================================================================
# Test configurations
# =============================================================================

TEST_CONFIGS = [
    # (cb_shape, kernel_factory, torch_fn, op_type, expected_unroll)

    # Binary ops - footprint=3, capacity=8, unroll=floor(8/3)=2
    ((1, 4), make_binary_kernel, lambda l, r: l + r, "binary", 2),  # 4 tiles (evenly divisible)
    ((1, 5), make_binary_kernel, lambda l, r: l + r, "binary", 2),  # 5 tiles (remainder: 1 iter)
    ((2, 2), make_binary_kernel, lambda l, r: l + r, "binary", 2),  # 4 tiles (2D grid)

    # Unary ops - footprint=2, capacity=8, unroll=floor(8/2)=4
    ((2, 2), make_unary_kernel, lambda x: torch.exp(x), "unary", 4),  # 4 tiles (evenly divisible)
    ((1, 5), make_unary_kernel, lambda x: torch.exp(x), "unary", 4),  # 5 tiles (remainder: 1 iter)

    # Fused chains - footprint=3 (operations reuse DST regs), unroll=2
    ((1, 4), make_fused_kernel, lambda l, r: torch.exp(l + r * l), "fused", 2),  # evenly divisible
    ((1, 3), make_fused_kernel, lambda l, r: torch.exp(l + r * l), "fused", 2),  # remainder: 1 iter
]


# =============================================================================
# Test runner
# =============================================================================


def run_test(config, device):
    """Run a single unroll correctness test."""
    cb_shape, kernel_factory, torch_fn, op_type, expected_unroll = config
    rows, cols = cb_shape
    tensor_shape = (rows * 32, cols * 32)

    # Create kernel
    kernel = kernel_factory(cb_shape)

    # Generate inputs and run
    if op_type == "unary":
        inp_torch = torch.full(tensor_shape, 0.5, dtype=torch.bfloat16)
        expected = torch_fn(inp_torch)

        inp = ttnn.from_torch(inp_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.from_torch(torch.zeros(tensor_shape, dtype=torch.bfloat16),
                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        kernel(inp, out)
        result = ttnn.to_torch(out)
    else:  # binary or fused
        lhs_torch = torch.rand(tensor_shape, dtype=torch.bfloat16) * 2.0 - 1.0
        rhs_torch = torch.rand(tensor_shape, dtype=torch.bfloat16) * 2.0 - 1.0
        expected = torch_fn(lhs_torch, rhs_torch)

        lhs = ttnn.from_torch(lhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rhs = ttnn.from_torch(rhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.from_torch(torch.zeros(tensor_shape, dtype=torch.bfloat16),
                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        kernel(lhs, rhs, out)
        result = ttnn.to_torch(out)

    # Verify numerical correctness (fused ops need higher tolerance due to accumulated rounding)
    rtol = 2e-2 if op_type == "fused" else 1e-2
    atol = 2e-2 if op_type == "fused" else 1e-2
    torch.testing.assert_close(result.float(), expected.float(), rtol=rtol, atol=atol)

    # Report
    num_tiles = rows * cols
    remainder = num_tiles % expected_unroll
    status = "✓" if remainder == 0 else f"✓ (remainder: {remainder} iter)"
    print(f"{status} {cb_shape[0]}x{cb_shape[1]} {op_type:6s} unroll={expected_unroll} - numerically correct")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=== Loop Unrolling Correctness Tests ===")
    print()

    device = ttnn.open_device(device_id=0)

    try:
        for i, config in enumerate(TEST_CONFIGS, 1):
            cb_shape, _, _, op_type, expected_unroll = config
            print(f"Test {i}/{len(TEST_CONFIGS)}: ", end="")
            run_test(config, device)
    finally:
        ttnn.close_device(device)

    print()
    print("=== All Unroll Correctness Tests Passed ===")
