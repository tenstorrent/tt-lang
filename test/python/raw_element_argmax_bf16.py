# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
BFloat16 argmax element access kernel -- verifies the argmax reduction
pattern compiles correctly with bf16 tensors.

Mirrors raw_element_argmax.py but with bf16 input/output tensors. The
lowering must produce 16-bit L1 pointer access (uint16_t) and
comparisons must route through bfloat16_greater().

Compile-only test.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def bf16_argmax_element_kernel(inp, out):
    """Argmax-style kernel on bf16 tensors: multi-row reduction.

    Scans two tile rows element-by-element. For each row, tracks the
    "best" value via a cross-scope variable update. Writes the max
    value to the output.
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
                max_val = ttl.raw_element_read(rblk, 0, 0)
                for c in range(32):
                    val = ttl.raw_element_read(rblk, 0, c)
                    if val > max_val:
                        max_val = val
                ttl.raw_element_write(wblk, 0, 0, max_val)

                best = ttl.raw_element_read(rblk, 1, 0)
                for c in range(32):
                    val = ttl.raw_element_read(rblk, 1, c)
                    if val > best:
                        best = val
                        ttl.raw_element_write(wblk, 1, 0, val)
                ttl.raw_element_write(wblk, 1, 1, best)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks -- bf16 element access ops and ogt comparison
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_read
# CHECK: arith.cmpf ogt
# CHECK: ttl.raw_element_write
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_read
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

bf16_argmax_element_kernel(inp, out)
ttnn.close_device(device)
