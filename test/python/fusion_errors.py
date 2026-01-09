# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s 2>&1 | FileCheck %s

"""
Negative tests for fusion errors.

Tests that the compiler produces clear error messages when fusion fails due to:
- Multiple uses of an intermediate value
- Non-elementwise operations in the chain
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

from ttlang import ttl, make_circular_buffer_like, exp, sqrt
from ttlang.ttl_api import Program
from ttlang.operators import copy

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


# CHECK: fusion failed: intermediate value has multiple uses
@ttl.kernel(grid=(1, 1))
def multiple_uses_kernel(inp, out):
    """Kernel where intermediate value is used twice - should fail fusion."""
    inp_cb = make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        i = inp_cb.wait()
        o = out_cb.reserve()
        # exp_result is used twice - this should fail fusion
        exp_result = exp(i)
        result = exp_result + exp_result
        o.store(result)
        inp_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        inp_cb.reserve()
        tx = copy(inp[0, 0], inp_cb)
        tx.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute, dm_read, dm_write)(inp, out)


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

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

        # This should fail with fusion error
        multiple_uses_kernel(inp, out)

    finally:
        ttnn.close_device(device)
