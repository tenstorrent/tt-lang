# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""K-split matmul kernel (K_parts >= 2).

Grid layout: (N_parts * K_parts, M_parts). A core at logical
(col_c, row_c) owns the output block (row_c * M_BPN + i_m, n_p * N_BPN + i_n)
for i_m in [0, M_BPN) and i_n in [0, N_BPN), where n_p = col_c % N_parts
and k_p = col_c // N_parts. Root cores have k_p == 0 (col_c < N_parts);
non-root cores gather their partials to the root at (n_p, row_c).

Use summa_kernel.py for K_parts == 1 configurations.
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
    """Build a K-split matmul operation (K_parts >= 2).

    M, N must be the *padded* dimensions (i.e. M = M_parts * M_BPN * bm * TILE
    and similarly for N). K must equal K_parts * K_BPN * bk * TILE. The caller
    is responsible for padding tensors before dispatch.
    """
    bm, bn, bk = block_cfg  # block dims in tiles
    Mp, Np, Kp = part_cfg

    if Kp < 2:
        raise ValueError(
            f"ksplit_kernel requires K_parts >= 2, got Kp={Kp}. "
            f"Use summa_kernel for K_parts == 1."
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
    if Mb % Mp or Nb % Np or Kb % Kp:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} Kb={Kb} must divide "
            f"Mp={Mp} Np={Np} Kp={Kp} (caller must pad M and N)"
        )

    M_BPN = Mb // Mp
    N_BPN = Nb // Np
    K_BPN = Kb // Kp

    COL = Np * Kp
    ROW = Mp

    @ttl.operation(grid=(COL, ROW), fp32_dest_acc_en=fp32_dest_acc_en)
    def ksplit_matmul(a, w, out):
        a_pipes = [
            ttl.Pipe(src=(k_p * Np, m_p), dst=(slice(k_p * Np, (k_p + 1) * Np), m_p))
            for k_p in range(Kp)
            for m_p in range(Mp)
        ]
        mcast_a_net = ttl.PipeNet(a_pipes)

        b_pipes = [
            ttl.Pipe(src=(col, 0), dst=(col, slice(0, Mp))) for col in range(COL)
        ]
        mcast_b_net = ttl.PipeNet(b_pipes)

        # Non-root k-ranks gather their partials to the root (k_p = 0) at the
        # same (n_p, m_p).
        reduce_pipes = [
            ttl.Pipe(src=(k_p * Np + n_p, m_p), dst=(n_p, m_p))
            for m_p in range(Mp)
            for n_p in range(Np)
            for k_p in range(1, Kp)
        ]
        reduce_net = ttl.PipeNet(reduce_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        partial_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        # recv_cb holds one slot per concurrent gather sender (Kp - 1), floored
        # at 2 because block_count must be >= 2.
        recv_cb = ttl.make_dataflow_buffer_like(
            out, shape=(bm, bn), block_count=max(2, Kp - 1)
        )
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=1)

        @ttl.compute()
        def compute():
            col_c, _ = ttl.node(dims=2)
            for _ in range(M_BPN):
                for _ in range(N_BPN):
                    p = partial_cb.reserve()
                    for _ in range(K_BPN):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        p += a_blk @ b_blk

                    if col_c < Np:
                        # Ideal form (would eliminate partial_cb ping-pong):
                        #   for _ in range(Kp - 1):
                        #       r = recv_cb.wait()
                        #       p += r
                        #   o = out_cb.reserve()
                        #   o.store(p)
                        # Blocked by loop-reassignment dropping the add; see
                        # https://github.com/tenstorrent/tt-lang/issues/527.
                        for _ in range(Kp - 1):
                            prev = partial_cb.wait()
                            r = recv_cb.wait()
                            new = partial_cb.reserve()
                            new.store(prev + r)
                        final = partial_cb.wait()
                        o = out_cb.reserve()
                        o.store(final)

        @ttl.datamovement()
        def dm_read():
            col_c, row_c = ttl.node(dims=2)
            k_p = col_c // Np
            for local_mb in range(M_BPN):
                mb = row_c * M_BPN + local_mb
                mr = mb * bm
                for _ in range(N_BPN):
                    for kb_local in range(K_BPN):
                        kc = (k_p * K_BPN + kb_local) * bk
                        a_blk = a_cb.reserve()

                        def read_a(pipe):
                            ttl.copy(a[mr : mr + bm, kc : kc + bk], a_blk).wait()
                            ttl.copy(a_blk, pipe).wait()

                        mcast_a_net.if_src(read_a)
                        mcast_a_net.if_dst(lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

                    if reduce_net.is_dst():

                        def recv(pipe):
                            r = recv_cb.reserve()
                            ttl.copy(pipe, r).wait()

                        reduce_net.if_dst(recv)
                    elif reduce_net.is_src():
                        p = partial_cb.wait()

                        def send(pipe):
                            ttl.copy(p, pipe).wait()

                        reduce_net.if_src(send)

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            k_p = col_c // Np
            n_p = col_c - k_p * Np
            for local_mb in range(M_BPN):
                mb = row_c * M_BPN + local_mb
                mr = mb * bm
                for local_nb in range(N_BPN):
                    nb = n_p * N_BPN + local_nb
                    nc = nb * bn
                    for kb_local in range(K_BPN):
                        kc = (k_p * K_BPN + kb_local) * bk
                        b_blk = b_cb.reserve()

                        def read_b(pipe):
                            ttl.copy(w[kc : kc + bk, nc : nc + bn], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(lambda pipe: (ttl.copy(pipe, b_blk).wait(),))
                    if col_c < Np:
                        o = out_cb.wait()
                        ttl.copy(o, out[mr : mr + bm, nc : nc + bn]).wait()

    return ksplit_matmul
