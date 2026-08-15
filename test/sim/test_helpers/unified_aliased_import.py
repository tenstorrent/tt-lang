# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a thread-unified @ttl.operation body in
# a module that imports the API as `T` instead of `ttl`.
#
# Thread assignment resolves calls by the receiver name `ttl`, so this body
# cannot be split correctly. The simulator must reject it with an explanation
# rather than mis-split it.

import ttl as T
import ttnn

device = ttnn.open_device(device_id=0)

try:

    @T.operation(grid=(1, 1))
    def copy_through(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
        dfb = T.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        blk = dfb.reserve()
        T.copy(a[0:1, 0:1], blk).wait()
        blk.push()
        out_blk = dfb.wait()
        T.copy(out_blk, out[0:1, 0:1]).wait()
        out_blk.pop()

    x = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    z = ttnn.zeros(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    copy_through(x, z)

finally:
    ttnn.close_device(device)
