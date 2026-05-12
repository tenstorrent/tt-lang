# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Element access kernel - verifies element_read and element_write lower correctly.

Tests reading a single element from a CB block and writing it to another CB block
in a datamovement thread.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def element_copy_kernel(inp, out):
    """Read element [0,0] from input tile, write to output tile [0,0]."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as blk:
            pass
        for _ in range(1):
            with out_dfb.reserve() as oblk:
                pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            val = ttl.element_read(rblk, 0, 5)
            with out_dfb.reserve() as wblk:
                ttl.element_write(wblk, 0, 0, val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks - Verify TTL element access ops appear
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.element_read
# CHECK: ttl.element_write

# =============================================================================
# C++ Kernel Checks - Verify generated C++ contains helper functions
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()

# Helper functions should be emitted
# CHECK-CPP: _ttl_elem_read_bf16
# CHECK-CPP: _ttl_elem_write_bf16


# =============================================================================
# Second kernel: loop variables, if conditionals, scalar arithmetic
#
# WARNING: This kernel compares bf16 element values as raw i32 bit patterns.
# Equality (==) works correctly, but magnitude comparisons (>, <) on raw i32
# bit patterns are NOT correct for bf16 because bf16 uses sign-magnitude
# representation. See https://github.com/tenstorrent/tt-lang/issues/572
# =============================================================================


@ttl.operation(grid=(1, 1))
def element_scan_kernel(inp, out):
    """Scan a tile column-by-column, compare elements, write computed index."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as blk:
            pass
        with out_dfb.reserve() as oblk:
            pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            max_val = ttl.element_read(rblk, 0, 0)
            with out_dfb.reserve() as wblk:
                for c in range(32):
                    val = ttl.element_read(rblk, 0, c)
                    if val == max_val:
                        ttl.element_write(wblk, 0, 0, c * 32 + 1)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# Second kernel C++ checks: loop var, if, scalar arithmetic
# CHECK-CPP: // dm_write
# CHECK-CPP: _ttl_elem_read_bf16
# CHECK-CPP: if
# CHECK-CPP: _ttl_elem_write_bf16


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Element Access Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
        out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

        inp = ttnn.from_torch(
            inp_torch,
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

        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling element access kernel...")
        element_copy_kernel(inp, out)

        print("Compiling element scan kernel...")
        element_scan_kernel(inp, out)

        print("=== Element Access Test Complete ===")

    finally:
        ttnn.close_device(device)
