# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a hand-written multi-kernel
# @ttl.operation whose kernel decorators are reached through an aliased import
# (`import ttl as T`) and a direct one (`from ttl import compute`).
#
# The kernels must be recognized as hand-written whatever the spelling. If they
# are mistaken for a thread-unified body the operation gets split and silently
# computes the wrong answer, so the result is checked against torch.

import torch

import ttl as T
import ttnn
from ttl import compute

device = ttnn.open_device(device_id=0)

try:

    @T.operation(grid=(1, 1))
    def add(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
        a_dfb = T.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = T.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        out_dfb = T.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @T.datamovement()
        def reader() -> None:
            with a_dfb.reserve() as a_blk:
                T.copy(a[0:1, 0:1], a_blk).wait()
            with b_dfb.reserve() as b_blk:
                T.copy(b[0:1, 0:1], b_blk).wait()

        @compute()
        def comp() -> None:
            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                with out_dfb.reserve() as out_blk:
                    out_blk.store(a_blk + b_blk)

        @T.datamovement()
        def writer() -> None:
            with out_dfb.wait() as out_blk:
                T.copy(out_blk, out[0:1, 0:1]).wait()

    x = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    z = ttnn.zeros(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    add(x, y, z)

    expected = ttnn.to_torch(x) + ttnn.to_torch(y)
    assert torch.allclose(
        expected, ttnn.to_torch(z), rtol=1e-1, atol=1e-1
    ), "aliased multi-kernel operation did not match torch reference"

finally:
    ttnn.close_device(device)
