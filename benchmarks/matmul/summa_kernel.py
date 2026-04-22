# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure SUMMA matmul kernel (K_parts = 1).

Grid layout: (N_parts, M_parts). A core at logical (col_c, row_c) owns
output block (row_c * M_BPN + i_m, col_c * N_BPN + i_n) for i_m in
[0, M_BPN) and i_n in [0, N_BPN). No K-split, no gather: the compute
loop reduces over Kb blocks and writes out_cb straight to the output
tensor.

Use ksplit_kernel.py for K_parts > 1 configurations.
"""

from __future__ import annotations

from typing import Tuple

import ttl

TILE = 32


def make_kernel(
    M: int,
    K: int,
    N: int,
    block_cfg: Tuple[int, int, int],
    part_cfg: Tuple[int, int, int],
    *,
    fp32_dest_acc_en: bool = True,
):
    """Build a pure-SUMMA matmul operation (K_parts must be 1).

    M, N must be the *padded* dimensions. K must be tile-aligned. The
    caller is responsible for padding tensors before dispatch.
    """
    bm, bn, bk = block_cfg  # block dims in tiles
    Mp, Np, Kp = part_cfg

    if Kp != 1:
        raise ValueError(
            f"summa_kernel requires K_parts == 1, got Kp={Kp}. "
            f"Use ksplit_kernel for K_parts > 1."
        )

    if M % TILE or N % TILE or K % TILE:
        raise ValueError(f"M/K/N must be tile-aligned: M={M} K={K} N={N}")

    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape in tiles: Mt={Mt} Nt={Nt} Kt={Kt} "
            f"block=(bm={bm}, bn={bn}, bk={bk})"
        )

    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Mb % Mp or Nb % Np:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} must divide "
            f"Mp={Mp} Np={Np} (caller must pad M and N)"
        )

    M_BPN = Mb // Mp
    N_BPN = Nb // Np

    COL = Np
    ROW = Mp

    @ttl.operation(grid=(COL, ROW), fp32_dest_acc_en=fp32_dest_acc_en)
    def summa_matmul(a, w, out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p)) for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)

        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp))) for n_p in range(Np)]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(M_BPN):
                for _ in range(N_BPN):
                    p = out_cb.reserve()
                    for _ in range(Kb):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        p += a_blk @ b_blk

        @ttl.datamovement()
        def dm_read():
            _, row_c = ttl.node(dims=2)
            for local_mb in range(M_BPN):
                mb = row_c * M_BPN + local_mb
                mr = mb * bm
                for _ in range(N_BPN):
                    for kb in range(Kb):
                        kc = kb * bk
                        a_blk = a_cb.reserve()

                        def read_a(pipe):
                            ttl.copy(a[mr : mr + bm, kc : kc + bk], a_blk).wait()
                            ttl.copy(a_blk, pipe).wait()

                        mcast_a_net.if_src(read_a)
                        mcast_a_net.if_dst(lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            for local_mb in range(M_BPN):
                mb = row_c * M_BPN + local_mb
                mr = mb * bm
                for local_nb in range(N_BPN):
                    nb = col_c * N_BPN + local_nb
                    nc = nb * bn
                    for kb in range(Kb):
                        kc = kb * bk
                        b_blk = b_cb.reserve()

                        def read_b(pipe):
                            ttl.copy(w[kc : kc + bk, nc : nc + bn], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(lambda pipe: (ttl.copy(pipe, b_blk).wait(),))
                    o = out_cb.wait()
                    ttl.copy(o, out[mr : mr + bm, nc : nc + bn]).wait()

    return summa_matmul
