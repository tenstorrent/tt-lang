# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test loop unrolling for DST register utilization optimization.

Verifies that tile compute loops are unrolled to maximize DST register usage:
- 4 tiles with 2 inputs (footprint=3) and capacity=8 → unroll_factor=2
- DST indices are correctly updated per unrolled iteration
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import importlib.util
import os
import re
import tempfile

import pytest

# Skip all tests if ttnn or device not available
pytestmark = [pytest.mark.requires_ttnn, pytest.mark.requires_device]


# =============================================================================
# Kernel Templates
# =============================================================================

MULTITILE_BINARY_KERNEL_TEMPLATE = '''
from ttlang import ttl

@ttl.kernel(grid=(1, 1))
def {name}_kernel(lhs, rhs, out):
    """Binary kernel with {shape} tiles to test loop unrolling."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape={shape}, buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape={shape}, buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape={shape}, buffer_factor=2)

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
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[{slice}], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[{slice}], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[{slice}])
        tx.wait()
        out_cb.pop()

    return ttl.Program(add_compute, dm_read, dm_write)(lhs, rhs, out)
'''


def make_multitile_kernel(name: str, shape: tuple):
    """Generate a multi-tile binary kernel."""
    if len(shape) == 1:
        slice_str = f"0:{shape[0]}"
    else:
        slice_str = ", ".join(f"0:{dim}" for dim in shape)

    code = MULTITILE_BINARY_KERNEL_TEMPLATE.format(
        name=name,
        shape=shape,
        slice=slice_str,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"kernel_{name}_"
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location(f"{name}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, f"{name}_kernel")


# =============================================================================
# Tests
# =============================================================================


def test_unroll_factor_4_tiles():
    """Test that 4-tile binary add is unrolled by factor 2.

    With 2 inputs and 1 output, footprint_per_iteration = 3.
    With DST capacity = 8, unroll_factor = floor(8/3) = 2.
    """
    import ttnn

    # Capture MLIR output
    mlir_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", delete=False, prefix="unroll_test_"
    )
    mlir_path = mlir_file.name
    mlir_file.close()

    os.environ["TTLANG_FINAL_MLIR"] = mlir_path
    os.environ["TTLANG_COMPILE_ONLY"] = "1"

    try:
        device = ttnn.open_device(device_id=0)

        try:
            import torch

            # 1x4 tiles = 32x128 tensor
            kernel = make_multitile_kernel("unroll_4tiles", (1, 4))

            lhs_torch = torch.full((32, 128), 2.0, dtype=torch.bfloat16)
            rhs_torch = torch.full((32, 128), 3.0, dtype=torch.bfloat16)
            out_torch = torch.zeros((32, 128), dtype=torch.bfloat16)

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

            kernel(lhs, rhs, out)

        finally:
            ttnn.close_device(device)

        # Verify MLIR output
        with open(mlir_path, "r") as f:
            mlir_content = f.read()

        # Check for step %c2 (unroll factor 2)
        assert re.search(r"scf\.for.*step\s+%c2", mlir_content), \
            "Expected scf.for loop with step %c2 (unroll factor 2)"

        # Check for unroll_factor attribute
        assert "ttl.unroll_factor = 2" in mlir_content, \
            "Expected ttl.unroll_factor = 2 attribute"

    finally:
        del os.environ["TTLANG_FINAL_MLIR"]
        del os.environ["TTLANG_COMPILE_ONLY"]
        if os.path.exists(mlir_path):
            os.unlink(mlir_path)


def test_no_unroll_single_tile():
    """Test that single-tile kernel is not unrolled.

    With only 1 tile, unroll_factor should not be set.
    """
    import ttnn

    mlir_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", delete=False, prefix="no_unroll_test_"
    )
    mlir_path = mlir_file.name
    mlir_file.close()

    os.environ["TTLANG_FINAL_MLIR"] = mlir_path
    os.environ["TTLANG_COMPILE_ONLY"] = "1"

    try:
        device = ttnn.open_device(device_id=0)

        try:
            import torch

            # 1x1 tiles = 32x32 tensor (single tile)
            kernel = make_multitile_kernel("single_tile", (1, 1))

            lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
            rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
            out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

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

            kernel(lhs, rhs, out)

        finally:
            ttnn.close_device(device)

        # Verify MLIR output
        with open(mlir_path, "r") as f:
            mlir_content = f.read()

        # Check that unroll_factor is NOT present
        assert "ttl.unroll_factor" not in mlir_content, \
            "Single-tile kernel should not have ttl.unroll_factor attribute"

    finally:
        del os.environ["TTLANG_FINAL_MLIR"]
        del os.environ["TTLANG_COMPILE_ONLY"]
        if os.path.exists(mlir_path):
            os.unlink(mlir_path)


def test_dst_indices_updated_per_iteration():
    """Test that DST indices are correctly updated for each unrolled iteration.

    For a binary add with 4 tiles and unroll_factor=2:
    - Iteration 0: DST indices 0, 1 (copy_tile), dst_idx=0 (tile_add)
    - Iteration 1: DST indices 2, 3 (copy_tile), dst_idx=2 (tile_add)
    """
    import ttnn

    mlir_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", delete=False, prefix="dst_indices_test_"
    )
    mlir_path = mlir_file.name
    mlir_file.close()

    os.environ["TTLANG_FINAL_MLIR"] = mlir_path
    os.environ["TTLANG_COMPILE_ONLY"] = "1"

    try:
        device = ttnn.open_device(device_id=0)

        try:
            import torch

            kernel = make_multitile_kernel("dst_indices", (1, 4))

            lhs_torch = torch.full((32, 128), 2.0, dtype=torch.bfloat16)
            rhs_torch = torch.full((32, 128), 3.0, dtype=torch.bfloat16)
            out_torch = torch.zeros((32, 128), dtype=torch.bfloat16)

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

            kernel(lhs, rhs, out)

        finally:
            ttnn.close_device(device)

        with open(mlir_path, "r") as f:
            mlir_content = f.read()

        # Find the add_compute function section
        compute_match = re.search(
            r"func\.func @add_compute.*?(?=func\.func|$)",
            mlir_content,
            re.DOTALL
        )
        assert compute_match, "Could not find add_compute function"
        compute_section = compute_match.group(0)

        # Check for DST index 0 in first iteration (dst_idx = 0)
        assert "dst_idx = 0" in compute_section, \
            "Expected dst_idx = 0 for first iteration tile_add"

        # Check for DST index 2 in second iteration (base=0, unrollIdx=1, slots=2 -> 0+1*2=2)
        # Note: The actual index depends on slotsPerIteration calculation
        assert "dst_idx = 2" in compute_section, \
            "Expected dst_idx = 2 for second iteration tile_add"

        # Check that we have constants for DST indices 0, 1, 2, 3
        assert re.search(r"arith\.constant\s+0\s*:", compute_section), \
            "Expected constant 0 for DST index"
        assert re.search(r"arith\.constant\s+1\s*:", compute_section), \
            "Expected constant 1 for DST index"
        assert re.search(r"arith\.constant\s+2\s*:", compute_section), \
            "Expected constant 2 for DST index"
        assert re.search(r"arith\.constant\s+3\s*:", compute_section), \
            "Expected constant 3 for DST index"

    finally:
        del os.environ["TTLANG_FINAL_MLIR"]
        del os.environ["TTLANG_COMPILE_ONLY"]
        if os.path.exists(mlir_path):
            os.unlink(mlir_path)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    if importlib.util.find_spec("ttnn") is None:
        print("TTNN not available - skipping tests")
        sys.exit(0)

    print("=== Loop Unroll Compute Tests ===\n")
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
