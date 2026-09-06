# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Add two tensors while calling user-provided C++ from a compiled kernel.

The external marker is intentionally a no-op: the tensor golden checks the
normal kernel path while the generated C++ proves that ``ttl.call_extern_func``
was lowered. The Python simulator intentionally rejects that compiler-only API.
"""

import os

import torch
import ttl
import ttnn

TILE_SIZE = 32
EXTERNAL_HEADER = os.path.join(os.path.dirname(__file__), "compiler_only_marker.hpp")


@ttl.operation(grid=(1, 1))
def compiler_only_external_call(a_in, b_in, out):
    row_tiles = a_in.shape[0] // TILE_SIZE
    col_tiles = a_in.shape[1] // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        ttl.call_extern_func(EXTERNAL_HEADER, "compiler_only_marker")

        for row in range(row_tiles):
            for col in range(col_tiles):
                with (
                    a_dfb.wait() as a_block,
                    b_dfb.wait() as b_block,
                    out_dfb.reserve() as out_block,
                ):
                    out_block.store(a_block + b_block)

    @ttl.datamovement()
    def read():
        for row in range(row_tiles):
            for col in range(col_tiles):
                with a_dfb.reserve() as a_block, b_dfb.reserve() as b_block:
                    a_copy = ttl.copy(a_in[row : row + 1, col : col + 1], a_block)
                    b_copy = ttl.copy(b_in[row : row + 1, col : col + 1], b_block)
                    a_copy.wait()
                    b_copy.wait()

    @ttl.datamovement()
    def write():
        for row in range(row_tiles):
            for col in range(col_tiles):
                with out_dfb.wait() as out_block:
                    ttl.copy(out_block, out[row : row + 1, col : col + 1]).wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        shape = (64, 64)
        a_host = torch.rand(shape, dtype=torch.bfloat16)
        b_host = torch.rand(shape, dtype=torch.bfloat16)

        a = ttnn.from_torch(
            a_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        b = ttnn.from_torch(
            b_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        out = ttnn.from_torch(
            torch.zeros_like(a_host),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        compiler_only_external_call(a, b, out)

        torch.testing.assert_close(
            ttnn.to_torch(out), a_host + b_host, rtol=1e-2, atol=1e-2
        )
        print("PASSED: compiler external C++ call executed")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
