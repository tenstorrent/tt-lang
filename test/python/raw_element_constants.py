# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Element access kernel -- verifies raw_element_write with a constant value
lowers correctly.

Tests writing a constant float to a CB block in a datamovement thread.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def element_constant_write_kernel(out):
    """Write a constant value pi to the output tile [0,0]."""
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        val = 3.14
        with out_dfb.reserve() as wblk:
            ttl.raw_element_write(wblk, 0, 0, val)
            tx = ttl.copy(wblk, out[0, 0])
            tx.wait()


# =============================================================================
# Initial IR Checks - Verify TTL element access ops appear
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.raw_element_write

# =============================================================================
# C++ Kernel Checks - Verify generated C++ contains lowering output
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()

# Lowering should produce reinterpret_cast and array subscript access
# CHECK-CPP: reinterpret_cast<tt_l1_ptr uint32_t*>


device = ttnn.open_device(device_id=0)

out = ttnn.from_torch(
    __import__("torch").zeros(32, 32, dtype=__import__("torch").float32),
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)

element_constant_write_kernel(out)

ttnn.close_device(device)
