# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
BFloat16 constant write kernel -- verifies that writing an f32 constant
to a bf16 block inserts arith.truncf and lowers to uint16_t pointer access.

The Python DSL auto-inserts arith.truncf when writing an f32 literal to a
bf16 block. The lowering extracts the upper 16 bits of the f32 encoding
via shift+trunc.

Compile-only test.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def bf16_constant_write_kernel(out):
    """Write a constant value (3.14) to a bf16 output tile [0,0].

    The f32 literal is implicitly truncated to bf16 by the DSL.
    """
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
# Initial IR Checks -- arith.truncf from f32 to bf16 before the write
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: arith.truncf
# CHECK: ttl.raw_element_write

# =============================================================================
# C++ Checks -- bf16 pointer width (uint16_t)
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()
# CHECK-CPP: reinterpret_cast<tt_l1_ptr uint16_t*>


device = ttnn.open_device(device_id=0)

out = ttnn.from_torch(
    __import__("torch").zeros(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

bf16_constant_write_kernel(out)
ttnn.close_device(device)
