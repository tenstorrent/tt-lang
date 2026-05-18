# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Element access kernel - verifies unsafe_element_read and unsafe_element_write
lower correctly.

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
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            val = ttl.unsafe.element_read(rblk, 0, 5)
            with out_dfb.reserve() as wblk:
                ttl.unsafe.element_write(wblk, 0, 0, val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks - Verify TTL element access ops appear
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.unsafe_element_read
# CHECK: ttl.unsafe_element_write

# =============================================================================
# C++ Kernel Checks - Verify generated C++ contains TTKernel lowering output
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()

# Lowering should produce get_read_ptr/get_write_ptr and load/store
# CHECK-CPP: get_read_ptr
# CHECK-CPP: get_write_ptr


# =============================================================================
# Second kernel: loop variables, if conditionals, scalar arithmetic
#
# This test validates compile-time lowering only (TTLANG_COMPILE_ONLY=1).
# It does NOT test numerical correctness on hardware.
#
# WARNING: unsafe_element_read returns raw element bits as i32. Equality (==) on
# these bit patterns is correct, but magnitude comparisons (>, <) on raw i32
# are NOT correct for f32 because f32 uses sign-magnitude representation.
# A runtime argmax using this pattern would produce wrong results for inputs
# containing negative or mixed-sign values.
# See https://github.com/tenstorrent/tt-lang/issues/572
# =============================================================================


@ttl.operation(grid=(1, 1))
def element_scan_kernel(inp, out):
    """Scan a tile column-by-column, compare elements, write computed index.

    Compile-only test: verifies the lowering of loop variables, if-blocks,
    and scalar arithmetic with unsafe_element_read/unsafe_element_write. Does not
    validate numerical correctness. See issue #572 for f32 comparison limitations.
    """
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            max_val = ttl.unsafe.element_read(rblk, 0, 0)
            with out_dfb.reserve() as wblk:
                for c in range(32):
                    val = ttl.unsafe.element_read(rblk, 0, c)
                    if val == max_val:
                        ttl.unsafe.element_write(wblk, 0, 0, c * 32 + 1)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# Second kernel C++ checks: loop var, if, scalar arithmetic
# CHECK-CPP: // dm_write
# CHECK-CPP: if


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Element Access Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.randn(32, 32, dtype=torch.float32)
        out_torch = torch.zeros(32, 32, dtype=torch.float32)

        inp = ttnn.from_torch(
            inp_torch,
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

        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling element access kernel...")
        element_copy_kernel(inp, out)

        print("Compiling element scan kernel...")
        element_scan_kernel(inp, out)

        print("=== Element Access Test Complete ===")

    finally:
        ttnn.close_device(device)
