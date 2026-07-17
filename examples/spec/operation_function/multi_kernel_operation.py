# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Illustrative example source for docs/sphinx/specs/TTLangSpecification.md.
#
# The lines between the "spec:begin" and "spec:end" markers below are included
# verbatim in the specification. Regenerate the specification after editing:
#
#     python docs/sphinx/specs/build_spec.py

import torch

import ttl
import ttnn


# spec:begin
@ttl.operation(grid=(1, 1))
def __foo(
    x: ttnn.Tensor,  # input tensor
    y: ttnn.Tensor,  # output tensor
) -> None:
    # ...
    # spec:end
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(4, 4), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(4, 4), block_count=2)
    # spec:begin

    @ttl.compute()
    def some_compute():
        # ...
        # spec:end
        with x_dfb.wait() as x_blk, y_dfb.reserve() as y_blk:
            y_blk.store(x_blk)
        # spec:begin

    @ttl.datamovement()
    def some_dm0():
        # ...
        # spec:end
        with x_dfb.reserve() as x_blk:
            x_xf = ttl.copy(x[0:4, 0:4], x_blk)
            x_xf.wait()
        # spec:begin

    @ttl.datamovement()
    def some_dm1():
        # ...
        # spec:end
        with y_dfb.wait() as y_blk:
            y_xf = ttl.copy(y_blk, y[0:4, 0:4])
            y_xf.wait()


torch.manual_seed(42)
# spec:begin

device = ttnn.open_device(device_id=0)

try:
    # Simple wrapper to allow returning output tensor in TT-NN style
    def foo(x: ttnn.Tensor) -> ttnn.Tensor:
        y = ttnn.zeros(x.shape, layout=ttnn.TILE_LAYOUT, device=device)
        __foo(x, y)
        return y

    shape = ttnn.Shape([128, 128])

    x = ttnn.rand(shape, layout=ttnn.TILE_LAYOUT, device=device)

    y = ttnn.exp(foo(ttnn.abs(x)), fast_and_approximate_mode=True)
    # spec:end
    assert torch.allclose(
        torch.exp(torch.abs(ttnn.to_torch(x))), ttnn.to_torch(y), rtol=1e-1, atol=1e-1
    ), "Tensors do not match"
    # spec:begin

finally:
    ttnn.close_device(device)
    # spec:end
