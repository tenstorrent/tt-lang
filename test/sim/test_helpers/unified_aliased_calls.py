# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a thread-unified @ttl.operation body
# whose calls go through an alias, in a module that ALSO binds `ttl`.
#
# Binding `ttl` is what makes this distinct from unified_aliased_import.py: a
# guard that only looks at how the module imported the API sees nothing wrong
# here. Thread assignment still resolves calls by receiver name, so these calls
# anchor nothing and the body must be turned away rather than mis-split.

import ttl
import ttl as T
import ttnn

device = ttnn.open_device(device_id=0)

try:

    @ttl.operation(grid=(1, 1))
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
