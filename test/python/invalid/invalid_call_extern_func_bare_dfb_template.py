# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: ttl.call_extern_func rejects ambiguous bare DFB template args.

The diagnostic names both explicit supported representations.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


FAKE_HEADER = "/dev/null/fake_shim.hpp"


# CHECK: TTLangCompileError: error: bare DFB template arguments are ambiguous; use ttl.dfb_descriptor(dfb) for allocation metadata or ttl.get_dfb_id(dfb) for an integer index
@ttl.operation(grid=(1, 1))
def invalid_bare_dfb_template(inp):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        ttl.call_extern_func(FAKE_HEADER, "my_shim", template_args=[in_dfb])

    @ttl.datamovement()
    def dm_read():
        blk = in_dfb.reserve()
        tx = ttl.copy(inp[0, 0], blk)
        tx.wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        blk = in_dfb.wait()
        blk.pop()


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)

        invalid_bare_dfb_template(inp)
    finally:
        ttnn.close_device(device)
