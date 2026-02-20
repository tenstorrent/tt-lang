# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from utils import assert_with_ulp
import ttl


@ttl.kernel(grid="auto")
def matmul_1d(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    out: ttnn.Tensor,
    granularity_m: int,
    granularity_n: int,
    granularity_k: int,
):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    assert b.shape[1] == out.shape[1], "Output matrix has incorrect number of columns."

    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]

    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    Mb = Mt // granularity_m
    Kb = Kt // granularity_k
    Nb = Nt // granularity_n

    nb_per_core = -(-Nb // ttl.grid_size(dims=1))  # divceil
    num_working_cores = -(-Nb // nb_per_core)  # divceil

    read_tiles = (
        (granularity_k * granularity_n + granularity_m * granularity_k) * Mb * Kb * Nb
    )
    write_tiles = (granularity_n * granularity_m) * Mb * Nb

    bf = 2
    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(granularity_m, granularity_k), buffer_factor=bf
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(granularity_k, granularity_n), buffer_factor=bf
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(granularity_m, granularity_n), buffer_factor=bf
    )

    def block_slice(block_offset, block_size):
        return slice(block_offset * block_size, (block_offset + 1) * block_size)

    @ttl.compute()
    def compute():
        core_index = ttl.core(dims=1)
        if core_index < num_working_cores:
            for mb in range(Mb):
                for local_nb in range(nb_per_core):
                    nb = local_nb + core_index * nb_per_core
                    if nb < Nb:
                        with out_dfb.reserve() as out_blk:
                            for kb in range(Kb):
                                with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                                    out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def a_reader_a_b_reader():
        core_index = ttl.core(dims=1)
        if core_index < num_working_cores:
            for mb in range(Mb):
                for local_nb in range(nb_per_core):
                    nb = local_nb + core_index * nb_per_core
                    if nb < Nb:
                        for kb in range(Kb):
                            with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:

                                ttl.copy(
                                    b[
                                        block_slice(kb, granularity_k),
                                        block_slice(nb, granularity_n),
                                    ],
                                    b_blk,
                                ).wait()

                                ttl.copy(
                                    a[
                                        block_slice(mb, granularity_m),
                                        block_slice(kb, granularity_k),
                                    ],
                                    a_blk,
                                ).wait()

    @ttl.datamovement()
    def out_writer():
        core_index = ttl.core(dims=1)
        if core_index < num_working_cores:
            for mb in range(Mb):
                for block_n in range(nb_per_core):
                    nb = block_n + core_index * nb_per_core
                    if nb < Nb:

                        with out_dfb.wait() as out_blk:
                            ttl.copy(
                                out_blk,
                                out[
                                    block_slice(mb, granularity_m),
                                    block_slice(nb, granularity_n),
                                ],
                            ).wait()


def test_matmul_1d(Mt, Nt, Kt, granularity_m, granularity_n, granularity_k):
    M = Mt * ttnn.TILE_SIZE
    N = Nt * ttnn.TILE_SIZE
    K = Kt * ttnn.TILE_SIZE

    print(
        "Testing a[M={0}, K={1}] @ b[K={1}, N={2}] = out[M={0}, N={2}]: ".format(
            M, K, N
        ),
        end="",
    )

    a = ttnn.rand(
        (M, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    b = ttnn.rand(
        (K, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    out = ttnn.empty(
        (M, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    matmul_1d(a, b, out, granularity_m, granularity_n, granularity_k)

    golden_out = a.to_torch() @ b.to_torch()
    assert_with_ulp(out.to_torch(), golden_out)

    print("PASSED!")


test_matmul_1d(1, 1, 1, 1, 1, 1)

test_matmul_1d(1, 64, 1, 1, 1, 1)
test_matmul_1d(1, 113, 1, 1, 1, 1)

test_matmul_1d(4, 256, 4, 1, 1, 1)
test_matmul_1d(4, 256, 4, 2, 1, 1)
test_matmul_1d(4, 256, 4, 1, 2, 1)
test_matmul_1d(4, 256, 4, 1, 1, 2)
test_matmul_1d(4, 256, 4, 2, 2, 2)
test_matmul_1d(4, 256, 4, 4, 4, 4)
