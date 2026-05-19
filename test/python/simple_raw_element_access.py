# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s
# RUN: FileCheck %s < %t.initial.mlir

"""
Raw element access kernel -- verifies ttl.raw_element_read and
ttl.raw_element_write lower correctly from the Python DSL into the
expected TTL dialect ops.

Tests reading a single element from a CB block and writing it to another
CB block in a datamovement thread. Compile-only; does not run on hardware.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def raw_element_copy_kernel(inp, out):
    """Read element [0,5] from input tile, write to output tile [0,0]."""
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
            val = ttl.raw_element_read(rblk, 0, 5)
            with out_dfb.reserve() as wblk:
                ttl.raw_element_write(wblk, 0, 0, val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks -- verify TTL raw element ops appear in dm_write
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# CHECK: ttl.raw_element_read
# CHECK-SAME: -> f32

# CHECK: ttl.raw_element_write


device = ttnn.open_device(device_id=0)

try:
    inp = ttnn.from_torch(
        torch.randn(32, 32, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    raw_element_copy_kernel(inp, out)

finally:
    ttnn.close_device(device)
