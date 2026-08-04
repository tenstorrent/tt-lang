# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a thread-unified @ttl.operation body
# whose dataflow buffer is built through an aliased API name, while every op the
# split depends on is spelled `ttl.<op>`.
#
# This is the spelling the alias guard must let through. Construction is
# recognized by name without its receiver, so the statement is hoisted out of the
# body and the three kernels share the one buffer; the copies that decide the
# split are spelled the way thread assignment reads them. The run has to produce
# the copied tile, not just survive decoration.

import ttl
import ttl as T
import ttnn
import torch

device = ttnn.open_device(device_id=0)

try:

    @ttl.operation(grid=(1, 1))
    def copy_through(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
        dfb = T.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        blk = dfb.reserve()
        ttl.copy(a[0:1, 0:1], blk).wait()
        blk.push()
        out_blk = dfb.wait()
        ttl.copy(out_blk, out[0:1, 0:1]).wait()
        out_blk.pop()

    x = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    z = ttnn.zeros(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    copy_through(x, z)

    assert torch.equal(
        ttnn.to_torch(z), ttnn.to_torch(x)
    ), "the aliased-factory body did not copy the tile"

finally:
    ttnn.close_device(device)
