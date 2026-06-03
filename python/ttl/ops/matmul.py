# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""K-split SUMMA matmul as a single @ttl.atom.

The output is tiled into ``(bm, bn)`` blocks. The grid is ``(Np*Kp, Mp)``:
the x axis folds ``Np`` N-partitions and ``Kp`` K-partitions, the y axis is
``Mp`` M-partitions. A core at ``(col_c, row_c)`` has K-rank ``k_p =
col_c // Np`` and N-rank ``n_p = col_c % Np``; it owns the ``M_BPN x N_BPN``
output blocks of its (row_c, n_p) tile, looping over them when there are more
blocks than cores.

  - A is multicast across the ``Np`` cores of each K-band, per M row.
  - B is multicast down the ``Mp`` rows of each column.
  - Each core accumulates its ``K_BPN``-block K-slice; non-root K-bands ship
    the partial to the ``k_p == 0`` root, which sums and writes out.

Both broadcasts reuse the ``ttl.ops.mcast`` op inline. KP=2 (a single reduce
step) is supported today; KP>2 (a multi-step / tree reduce) is a follow-on.
"""

from typing import Tuple

import ttl
from ttl.ops.mcast import mcast, mcast_cols

TILE = 32


def make_ksplit(
    M: int,
    K: int,
    N: int,
    block_cfg: Tuple[int, int, int],
    part_cfg: Tuple[int, int, int],
    *,
    fp32_dest_acc_en: bool = True,
):
    """Build a K-split matmul atom. ``M``/``N`` are the padded dims; the
    caller pads tensors so block and partition counts divide evenly."""
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg

    if Kp != 2:
        raise ValueError(f"make_ksplit currently supports Kp == 2, got Kp={Kp}")
    if M % TILE or N % TILE or K % TILE:
        raise ValueError(f"M/K/N must be tile-aligned: M={M} K={K} N={N}")

    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape in tiles: Mt={Mt} Nt={Nt} Kt={Kt} "
            f"block=(bm={bm}, bn={bn}, bk={bk})")

    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Mb % Mp or Nb % Np or Kb % Kp:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} Kb={Kb} must divide "
            f"Mp={Mp} Np={Np} Kp={Kp}")

    M_BPN = Mb // Mp
    N_BPN = Nb // Np
    K_BPN = Kb // Kp
    COL = Np * Kp

    @ttl.atom(grid=(COL, Mp), fp32_dest_acc_en=fp32_dest_acc_en)
    def ksplit(a, w, out):
        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        # mcast staging buffers so the loopback src core does not double-reserve
        # a_cb/b_cb across the send (BRISC) and receive (NCRISC) sides.
        tmp_a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        tmp_b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        # Split the partial per consumer: compute folds the received partial in
        # via partial_for_sum_cb; the DM thread ships the partial off via
        # partial_for_send_cb. One producer cannot feed both compute and DM.
        partial_for_sum_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        partial_for_send_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        recv_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        # A: multicast across the Np cores of each K-band, per M row.
        a_pipes = [
            ttl.Pipe(src=(k_p * Np, m), dst=(slice(k_p * Np, (k_p + 1) * Np), m))
            for k_p in range(Kp)
            for m in range(Mp)
        ]
        mcast_a_net = ttl.PipeNet(a_pipes)
        # B: multicast down the Mp rows of each column.
        mcast_b_net = ttl.PipeNet(mcast_cols(Mp, COL))
        # Reduce: non-root K-bands ship their partial to the k_p == 0 root.
        reduce_pipes = [
            ttl.Pipe(src=(k_p * Np + n_p, m), dst=(n_p, m))
            for n_p in range(Np)
            for m in range(Mp)
            for k_p in range(1, Kp)
        ]
        reduce_net = ttl.PipeNet(reduce_pipes)

        col_c, row_c = ttl.node(dims=2)
        k_p = col_c // Np
        n_p = col_c % Np

        # One pass per output block. Per block the source order is matmul ->
        # mcast -> reduce: the mcast (DM producer of a_cb/b_cb) must precede the
        # reduce on the DM thread, else the reduce send blocks on a partial that
        # depends on a matmul that depends on the not-yet-run mcast.
        for lmb in range(M_BPN):
            mr = (row_c * M_BPN + lmb) * bm
            for lnb in range(N_BPN):
                nc = (n_p * N_BPN + lnb) * bn

                p = partial_for_sum_cb.reserve()
                for _ in range(K_BPN):
                    a_blk = a_cb.wait()
                    b_blk = b_cb.wait()
                    p += a_blk @ b_blk
                pf = partial_for_sum_cb.wait()

                for kbl in range(K_BPN):
                    kc = (k_p * K_BPN + kbl) * bk
                    mcast(mcast_a_net, a[mr : mr + bm, kc : kc + bk], tmp_a_cb, a_cb)
                    mcast(mcast_b_net, w[kc : kc + bk, nc : nc + bn], tmp_b_cb, b_cb)

                if col_c < Np:
                    def recv_partial(pipe):
                        r_dst = recv_cb.reserve()
                        ttl.copy(pipe, r_dst)
                    reduce_net.if_dst(recv_partial)

                    r = recv_cb.wait()
                    o = out_cb.reserve()
                    o.store(pf + r)
                    out_blk = out_cb.wait()
                    ttl.copy(out_blk, out[mr : mr + bm, nc : nc + bn])
                else:
                    ps = partial_for_send_cb.reserve()
                    ps.store(pf)

                    def send_partial(pipe):
                        p_to_send = partial_for_send_cb.wait()
                        ttl.copy(p_to_send, pipe)
                    reduce_net.if_src(send_partial)

    return ksplit
