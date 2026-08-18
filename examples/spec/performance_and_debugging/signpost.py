# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Example source for docs/sphinx/specs/TTLangSpecification.md.
#
# The lines between the "spec:begin" and "spec:end" markers below are included
# verbatim in the specification. Regenerate the specification after editing:
#
#     python docs/sphinx/specs/build_spec.py
#
# Everything outside the markers (imports, scaffolding, the @ttl.operation
# wrapper, the kernels that consume what matmul_read loads, device setup, and
# the checks) exists so the file can run standalone; it is not copied into the
# specification. The marked lines are nested inside @ttl.operation and dedented
# on render, so these mechanics add nothing to the rendered text.


import torch

import ttl
import ttnn

# Concrete compile-time sizes for a single-block run.
TILE_SIZE = 32
I, M, N, K = 1, 32, 32, 32


@ttl.operation(grid=(1, 1))
def signpost_example(
    a: ttnn.Tensor,  # input tensor a (I, M, K)
    b: ttnn.Tensor,  # input tensor b (K, N)
    c: ttnn.Tensor,  # input bias tensor c (M, N)
    y: ttnn.Tensor,  # output tensor y (I, M, N)
) -> None:
    I_TILES = I
    M_TILES = M // TILE_SIZE
    N_TILES = N // TILE_SIZE
    K_TILES = K // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1, 1))
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1))
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1))
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1, 1))

    # The marked region is kept out of the formatter's reach: nesting it inside
    # the operation adds four columns, and reflowing a line for that indentation
    # would change the text the specification shows.
    # fmt: off
    # spec:begin
    @ttl.datamovement()
    def matmul_read():
        for i_tile in range(I_TILES):
            for m_tile in range(M_TILES):
                for n_tile in range(N_TILES):

                    # Measure the entire iteration

                    with ttl.signpost("i_m_n iteration"):

                        # Measure from reserve to push

                        with ttl.signpost("push c"):
                            with c_dfb.reserve() as c_blk:

                                # Measure only copy

                                with ttl.signpost("read c"):
                                    c_xf = ttl.copy(c[m_tile, n_tile], c_blk)
                                    c_xf.wait()

                        for k_tile in range(K_TILES):
                            with ttl.signpost("push a and b"):

                                # Measure from reserve to push

                                with (
                                    a_dfb.reserve() as a_blk,
                                    b_dfb.reserve() as b_blk,
                                ):

                                    # Measure only copy

                                    with ttl.signpost("read a and b"):
                                        a_xf = ttl.copy(a[i_tile, m_tile, k_tile], a_blk)
                                        b_xf = ttl.copy(b[k_tile, n_tile], b_blk)

                                        a_xf.wait()
                                        b_xf.wait()

    # spec:end
    # fmt: on

    @ttl.compute()
    def matmul_compute():
        for _ in range(I_TILES):
            for _ in range(M_TILES):
                for _ in range(N_TILES):
                    with y_dfb.reserve() as y_blk:
                        y_final = ttl.block.fill(0, shape=(1, 1, 1))

                        for _ in range(K_TILES):
                            with (
                                a_dfb.wait() as a_blk,
                                b_dfb.wait() as b_blk,
                            ):
                                b_bcast = ttl.block.broadcast(
                                    ttl.block.unsqueeze(b_blk, dims=[0]),
                                    dims=[0],
                                    shape=(1, 1, 1),
                                )
                                y_final += a_blk @ b_bcast

                        with c_dfb.wait() as c_blk:
                            c_bcast = ttl.block.broadcast(
                                ttl.block.unsqueeze(c_blk, dims=[0]),
                                dims=[0],
                                shape=(1, 1, 1),
                            )
                            y_final = y_final + c_bcast

                        y_blk.store(y_final)

    @ttl.datamovement()
    def matmul_write():
        for i_tile in range(I_TILES):
            for m_tile in range(M_TILES):
                for n_tile in range(N_TILES):
                    with y_dfb.wait() as y_blk:
                        y_xf = ttl.copy(y_blk, y[i_tile, m_tile, n_tile])
                        y_xf.wait()


device = ttnn.open_device(device_id=0)

try:
    a_t = ttnn.rand(ttnn.Shape([I, M, K]), layout=ttnn.TILE_LAYOUT, device=device)
    b_t = ttnn.rand(ttnn.Shape([K, N]), layout=ttnn.TILE_LAYOUT, device=device)
    c_t = ttnn.rand(ttnn.Shape([M, N]), layout=ttnn.TILE_LAYOUT, device=device)
    y_t = ttnn.zeros(ttnn.Shape([I, M, N]), layout=ttnn.TILE_LAYOUT, device=device)

    signpost_example(a_t, b_t, c_t, y_t)

    golden = torch.matmul(ttnn.to_torch(a_t), ttnn.to_torch(b_t)) + ttnn.to_torch(c_t)
    assert torch.allclose(
        golden, ttnn.to_torch(y_t), rtol=1e-1, atol=1e-1
    ), "matmul + bias did not match torch reference"

finally:
    ttnn.close_device(device)
