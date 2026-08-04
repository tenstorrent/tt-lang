# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a thread-unified @ttl.operation body that
# builds its own dataflow buffer under a name a module-level buffer already uses.
#
# The body captures nothing -- `dfb` is its own local -- so the guard against
# captured buffers must leave it alone. Reading the guard as "a name the body
# mentions that the enclosing scope also binds" turns this program away, and the
# program is a normal one: a file with one buffer at module scope and another
# operation that happens to name its buffer the same thing.

import ttl
import ttnn
import torch

device = ttnn.open_device(device_id=0)

try:
    template = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    # Shares the body's spelling for its buffer, and is nothing to do with it.
    dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.operation(grid=(1, 1))
    def copy_through(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
        dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
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
    ), "the body's own buffer did not carry the tile"

finally:
    ttnn.close_device(device)
