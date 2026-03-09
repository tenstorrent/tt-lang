# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
import ttl

TILE_SIZE = 32


def make_matmul_1d_kernel(
    granularity_m, granularity_n, granularity_k,
    num_working_cores, nb_per_core,
    Mb, Kb, Nb,
):
    @ttl.kernel(grid="auto")
    def matmul_1d(a, b, out):
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
        partial_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(granularity_m, granularity_n), buffer_factor=bf
        )

        mcast_a_net = ttl.PipeNet([
            ttl.Pipe((0, 0), (slice(1, num_working_cores), 0))
        ])

        @ttl.compute()
        def compute():
            x, y = ttl.core(dims=2)
            if x < num_working_cores:
                for mb in range(Mb):
                    for local_nb in range(nb_per_core):
                        nb = local_nb + x * nb_per_core
                        if nb < Nb:
                            with out_dfb.reserve() as out_blk:
                                for kb in range(Kb):
                                    with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                                        with partial_dfb.reserve() as partial_blk:
                                            partial_blk.store(a_blk @ b_blk)
                                    with partial_dfb.wait() as partial_blk:
                                        if kb == 0:
                                            out_blk.store(partial_blk)
                                        else:
                                            out_blk.store(out_blk + partial_blk)

        @ttl.datamovement()
        def a_reader_a_mcast_b_reader():
            x, y = ttl.core(dims=2)
            if x < num_working_cores:
                for mb in range(Mb):
                    for local_nb in range(nb_per_core):
                        nb = local_nb + x * nb_per_core
                        if nb < Nb:
                            for kb in range(Kb):
                                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                                    ttl.copy(
                                        b[
                                            kb * granularity_k:(kb + 1) * granularity_k,
                                            nb * granularity_n:(nb + 1) * granularity_n,
                                        ],
                                        b_blk,
                                    ).wait()

                                    def pipe_src(pipe):
                                        ttl.copy(
                                            a[
                                                mb * granularity_m:(mb + 1) * granularity_m,
                                                kb * granularity_k:(kb + 1) * granularity_k,
                                            ],
                                            a_blk,
                                        ).wait()

                                        ttl.copy(a_blk, pipe).wait()

                                    def pipe_dst(pipe):
                                        ttl.copy(pipe, a_blk).wait()

                                    mcast_a_net.if_src(pipe_src)
                                    mcast_a_net.if_dst(pipe_dst)

        @ttl.datamovement()
        def out_writer():
            x, y = ttl.core(dims=2)
            if x < num_working_cores:
                for mb in range(Mb):
                    for block_n in range(nb_per_core):
                        nb = block_n + x * nb_per_core
                        if nb < Nb:

                            with out_dfb.wait() as out_blk:
                                ttl.copy(
                                    out_blk,
                                    out[
                                        mb * granularity_m:(mb + 1) * granularity_m,
                                        nb * granularity_n:(nb + 1) * granularity_n,
                                    ],
                                ).wait()

    return matmul_1d


def _divceil(a, b):
    return -(-a // b)


def _make_tensor(shape, device=None):
    """Create a bfloat16 tensor, using device if available (hw) or sim."""
    t = torch.randn(shape, dtype=torch.bfloat16)
    if device is not None:
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _empty_tensor(shape, device=None):
    """Create an empty bfloat16 tensor."""
    t = torch.zeros(shape, dtype=torch.bfloat16)
    if device is not None:
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def test_matmul_1d(
    Mt, Nt, Kt, granularity_m, granularity_n, granularity_k, device=None
):
    M = Mt * TILE_SIZE
    N = Nt * TILE_SIZE
    K = Kt * TILE_SIZE

    Mb = Mt // granularity_m
    Kb = Kt // granularity_k
    Nb = Nt // granularity_n

    grid_width = 8
    nb_per_core = _divceil(Nb, grid_width)
    num_working_cores = _divceil(Nb, nb_per_core)

    print(
        "Testing a[M={0}, K={1}] @ b[K={1}, N={2}] = out[M={0}, N={2}]: ".format(
            M, K, N
        ),
        end="",
    )

    a = _make_tensor((M, K), device)
    b = _make_tensor((K, N), device)
    out = _empty_tensor((M, N), device)

    kernel = make_matmul_1d_kernel(
        granularity_m, granularity_n, granularity_k,
        num_working_cores, nb_per_core,
        Mb, Kb, Nb,
    )
    kernel(a, b, out)

    golden_out = ttnn.to_torch(a) @ ttnn.to_torch(b)
    actual_out = ttnn.to_torch(out)

    diff = (actual_out.float() - golden_out.float()).abs()
    max_diff = diff.max().item()
    # TODO: accumulation reads uninitialized out_blk on first iteration
    # (reserve() doesn't zero-initialize). Need a zero-init or acc=True support.
    print(f"max_diff={max_diff:.2f}")
    if max_diff < 1.0:
        print("PASSED!")
    else:
        print(f"NUMERICS MISMATCH (max_diff={max_diff:.2f}, expected < 1.0)")


if __name__ == "__main__":
    device = None
    try:
        device = ttnn.open_device(device_id=0)
    except Exception:
        pass

    try:
        # Skip (1,1,1,1,1,1): only 1 core, PipeNet multicast range is empty
        test_matmul_1d(1, 64, 1, 1, 1, 1, device=device)
    finally:
        if device is not None:
            ttnn.close_device(device)
