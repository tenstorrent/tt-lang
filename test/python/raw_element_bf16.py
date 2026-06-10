# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
BFloat16 element access kernel -- verifies raw_element_read and
raw_element_write lower correctly for bf16 tensors.

Mirrors the pairwise compare-and-swap pattern from raw_element_topk.py
but with bf16 input/output tensors. The lowering must produce 16-bit L1
pointer access (uint16_t) instead of 32-bit (uint32_t), and comparisons
must route through bfloat16_greater() rather than float32_greater().

Compile-only test.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def bf16_element_sort_kernel(inp, out):
    """Pairwise compare-and-swap at distance 16 on bf16 elements.

    Reads pairs of bf16 elements at distance 16 and sorts each pair
    so that the smaller value is at the lower index. Exercises
    bfloat16_greater() for the comparison.
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
            with out_dfb.reserve() as wblk:
                for i in range(16):
                    a = ttl.raw_element_read(rblk, 0, i)
                    b = ttl.raw_element_read(rblk, 0, i + 16)
                    ttl.raw_element_write(wblk, 0, i, a)
                    ttl.raw_element_write(wblk, 0, i + 16, b)
                    if a > b:
                        ttl.raw_element_write(wblk, 0, i, b)
                        ttl.raw_element_write(wblk, 0, i + 16, a)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks -- bf16 element access ops and comparison
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_write
# CHECK: ttl.raw_element_write
# CHECK: arith.cmpf ogt
# CHECK: ttl.raw_element_write
# CHECK: ttl.raw_element_write

# =============================================================================
# C++ Checks -- bf16 pointer width (uint16_t) and bfloat16_greater helper
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()
# CHECK-CPP: reinterpret_cast<tt_l1_ptr uint16_t*>
# CHECK-CPP: bfloat16_greater(


device = ttnn.open_device(device_id=0)

inp = ttnn.from_torch(
    __import__("torch").randn(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
out = ttnn.from_torch(
    __import__("torch").zeros(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

bf16_element_sort_kernel(inp, out)
ttnn.close_device(device)
