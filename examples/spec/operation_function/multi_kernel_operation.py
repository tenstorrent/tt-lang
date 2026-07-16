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

# spec:begin
@ttl.operation()
def __foo(
    x: ttnn.Tensor, # input tensor
    y: ttnn.Tensor, # output tensor
) -> None:
    # ...
# spec:end
    x_dfb = ttl.make_dataflow_buffer_like(x,
        shape = (4, 4),
        block_count = 2)
    y_dfb = ttl.make_dataflow_buffer_like(y,
        shape = (4, 4),
        block_count = 2)
# spec:begin

    @ttl.compute()
    def some_compute():
        # ...
# spec:end
        with x_blk = x_dfb.wait(), y_blk = y_dfb.reserve():
            y_blk.store(x_blk)
            y_blk.push() 
            x_blk.pop()
# spec:begin

    @ttl.datamovement()
    def some_dm0():
        # ...
# spec:end
        with x_dfb.reserve() as x_blk:
            x_xf = ttl.copy(x[:, :], x_blk)
            x_xf.wait()
# spec:begin

    @ttl.datamovement()
    def some_dm1():
        # ...
# spec:end
        with y_dfb.wait() as y_blk:

            y_xf = ttl.copy(y_blk, y[:, :])
            y_xf.wait()
# spec:begin

# Simple wrapper to allow returning output tensor in TT-NN style
def foo(x: ttnn.Tensor) -> ttnn.Tensor:
    y = ttnn.zeros(x.shape, layout=ttnn.TILE_LAYOUT)
    __foo(x, y)
    return y

shape = ttnn.Shape([128, 128])

x = ttnn.rand(shape, layout=ttnn.TILE_LAYOUT)

y = ttnn.exp(foo(ttnn.abs(x)), fast_and_approximate_mode=True)
# spec:end
