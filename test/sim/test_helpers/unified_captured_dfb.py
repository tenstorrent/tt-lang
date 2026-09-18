# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a thread-unified @ttl.operation body that
# uses a dataflow buffer built outside the operation, the way a user would try to
# share one between operations.
#
# The specification constructs a dataflow buffer in the scope of the operation
# function that uses it, and the compiler refuses a captured one at decoration
# time. Left to run, the simulator would not recognize the captured name as a
# buffer it constructed, so the reserve and wait against it would anchor no
# thread, be replicated onto all three kernels, and fail as a dataflow state
# error inside a kernel the user never wrote.

import ttl
import ttnn

device = ttnn.open_device(device_id=0)

try:
    template = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    shared_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.operation(grid=(1, 1))
    def copy_through(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
        blk = shared_dfb.reserve()
        ttl.copy(a[0:1, 0:1], blk).wait()
        blk.push()
        out_blk = shared_dfb.wait()
        ttl.copy(out_blk, out[0:1, 0:1]).wait()
        out_blk.pop()

    x = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    z = ttnn.zeros(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    copy_through(x, z)

finally:
    ttnn.close_device(device)
