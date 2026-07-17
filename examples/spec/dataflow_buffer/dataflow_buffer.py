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


def from_torch(tensor: torch.Tensor):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.operation(grid=(1, 1))
def dataflow_buffer_example(
    x: ttnn.Tensor,  # input tensor
    y: ttnn.Tensor,  # output tensor
) -> None:

    # spec:begin
    x_dfb = ttl.make_dataflow_buffer_like(
        x, shape=(2, 2), block_count=2
    )  # This can be omitted since block_count defaults to 2
    # spec:end
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(2, 2), block_count=2)
    # spec:begin

    @ttl.datamovement()
    def some_read():
        # Reserve x_blk from x_dfb
        with x_dfb.reserve() as x_blk:

            # Load data into x_blk
            # ...
            # spec:end
            x_xf = ttl.copy(x[0:2, 0:2], x_blk)
            x_xf.wait()
            # spec:begin

            # Push x_blk implicitly at the end of the "with" scope

    @ttl.compute()
    def some_compute():
        # Wait for x_blk from x_dfb
        x_blk = x_dfb.wait()

        # Consume data in x_blk
        # ...
        # spec:end
        y_blk = y_dfb.reserve()
        y_blk.store(x_blk)
        y_blk.push()
        # spec:begin

        x_blk.pop()  # Pop x_blk explicitly
        # spec:end

    @ttl.datamovement()
    def some_write():
        with y_dfb.wait() as y_blk:

            y_xf = ttl.copy(y_blk, y[0:2, 0:2])
            y_xf.wait()


torch.manual_seed(42)

device = ttnn.open_device(device_id=0)

try:
    shape = (64, 64)

    x = torch.rand(shape, dtype=torch.bfloat16)
    y = torch.rand(shape, dtype=torch.bfloat16)

    x = from_torch(x)
    y = from_torch(y)

    dataflow_buffer_example(x, y)

    x = ttnn.to_torch(x)
    y = ttnn.to_torch(y)

    assert torch.allclose(x, y, rtol=1e-1, atol=1e-1), "Tensors do not match"

finally:
    ttnn.close_device(device)
