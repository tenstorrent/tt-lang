# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a hand-written multi-kernel
# @ttl.operation whose kernel decorators come from an aliased import while the
# calls inside are spelled `ttl.<op>`.
#
# This is the spelling that reads most like a supported body, so a classifier
# keyed on decorator text mistakes it for a thread-unified body, splits it, and
# returns a wrong answer instead of failing. Compute only forwards the block, so
# `out` must come back equal to `a`.
#
# The kernels also use bare reserve/push/wait/pop rather than `with` blocks, and
# an empty compute kernel, which no other multi-kernel fixture covers.

import torch

import ttl
import ttl as T
import ttnn

device = ttnn.open_device(device_id=0)

try:

    @ttl.operation(grid=(1, 1))
    def forward(a: ttnn.Tensor, out: ttnn.Tensor) -> None:
        dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

        @T.datamovement()
        def reader() -> None:
            blk = dfb.reserve()
            ttl.copy(a[0:1, 0:1], blk).wait()
            blk.push()

        @T.compute()
        def comp() -> None:
            pass

        @T.datamovement()
        def writer() -> None:
            blk = dfb.wait()
            ttl.copy(blk, out[0:1, 0:1]).wait()
            blk.pop()

    x = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    z = ttnn.zeros(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    forward(x, z)

    assert torch.allclose(
        ttnn.to_torch(x), ttnn.to_torch(z), rtol=1e-1, atol=1e-1
    ), "multi-kernel operation with aliased decorators did not forward its input"

finally:
    ttnn.close_device(device)
