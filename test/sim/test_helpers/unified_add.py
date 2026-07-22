# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Fixture for test_unified_operation.py: a thread-unified @ttl.operation body
# (no hand-written @ttl.compute / @ttl.datamovement kernels). The simulator
# splits it into compute/dm0/dm1 kernels by reusing the compiler's splitter.
#
# Uses the splitter-accepted single-statement copy form `ttl.copy(...).wait()`
# (an assigned handle `tx = ttl.copy(...)` is rejected by atom_split).

import torch

import ttl
import ttnn

device = ttnn.open_device(device_id=0)

try:

    @ttl.operation(grid=(1, 1))
    def add(
        a: ttnn.Tensor,  # input tensor
        b: ttnn.Tensor,  # input tensor
        out: ttnn.Tensor,  # output tensor
    ) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        a_dst = a_dfb.reserve()
        ttl.copy(a[0:1, 0:1], a_dst).wait()
        a_dst.push()
        b_dst = b_dfb.reserve()
        ttl.copy(b[0:1, 0:1], b_dst).wait()
        b_dst.push()

        out_blk = out_dfb.reserve()
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        out_blk.store(a_blk + b_blk)
        a_blk.pop()
        b_blk.pop()

        ttl.copy(out_dfb.wait(), out[0:1, 0:1]).wait()
        out_blk.push()

    def add_op(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        out = ttnn.zeros(a.shape, layout=ttnn.TILE_LAYOUT, device=device)
        add(a, b, out)
        return out

    x = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.rand(ttnn.Shape([32, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    z = add_op(x, y)

    expected = ttnn.to_torch(x) + ttnn.to_torch(y)
    assert torch.allclose(
        expected, ttnn.to_torch(z), rtol=1e-1, atol=1e-1
    ), "unified @ttl.operation add did not match torch reference"

finally:
    ttnn.close_device(device)
