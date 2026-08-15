# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a thread-unified @ttl.operation body that
# constructs its dataflow buffer inside an `if`.
#
# Only top-level construction is hoisted, and hoisting is what makes the three
# kernels share one buffer. Left in the body, the construction is duplicated into
# every kernel, each reserves and waits on a buffer of its own, and the run fails
# with a dataflow state error against the wrong buffer that points nowhere near
# the cause. The operation must be turned away at decoration instead.

import ttl
import ttnn

USE_LARGE_BUFFER = True

device = ttnn.open_device(device_id=0)

try:

    @ttl.operation(grid=(1, 1))
    def copy_through(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
        if USE_LARGE_BUFFER:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=4)
        else:
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

finally:
    ttnn.close_device(device)
