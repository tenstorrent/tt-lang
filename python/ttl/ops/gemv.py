# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Weight-streamed GEMV / skinny matmul as a single @ttl.atom.

Decode projections are ``[m, K] @ [K, N]`` with m tiny (one row tile), so the
weight stream from DRAM dominates and the activation is reused by every core.
The grid is ``(Np, Kp)``: each column owns an N-band, each row a K-band.
``x`` is multicast once per K-band row; ``W`` streams per-core from DRAM into
a double-buffered DFB while compute accumulates a ``[1, bn]`` partial.
Non-root rows ship partials to row 0 over unicast pipes (``reduce_send`` /
``reduce_recv`` from ``ttl.ops.matmul``); row 0 drains to DRAM.

Distinct from ``ttl.ops.matmul``: that op multicasts both operands; here W is
read exactly once per core (DRAM-BW bound) and x is the only multicast. Each
streamed W block spans the core's whole K-band so the partial accumulates in
DST without block subviews.
"""

from typing import Tuple

import ttl
from ttl.ops.matmul import reduce_recv, reduce_send
from ttl.ops.mcast import mcast

TILE = 32


def make_gemv(
    M: int,
    K: int,
    N: int,
    grid_cfg: Tuple[int, int],
    bn: int,
    *,
    fp32_dest_acc_en: bool = True,
):
    """Build a weight-streamed GEMV atom for ``x[M,K] @ w[K,N] -> out[M,N]``.

    ``M`` must be one row tile. ``grid_cfg`` is ``(Np, Kp)``: N-bands x
    K-bands. ``bn`` is the streamed W block width in tiles; each block is
    ``(Kt/Kp, bn)``.
    """
    Np, Kp = grid_cfg

    if M != TILE:
        raise ValueError(f"GEMV M must be one row tile ({TILE}), got {M}")
    if Kp != 2:
        raise ValueError(f"make_gemv currently supports Kp == 2, got Kp={Kp}")
    if K % TILE or N % TILE:
        raise ValueError(f"K/N must be tile-aligned: K={K} N={N}")

    Kt, Nt = K // TILE, N // TILE
    if Kt % Kp or Nt % (Np * bn):
        raise ValueError(
            f"bands must divide tiles: Kt={Kt} Nt={Nt} Kp={Kp} Np={Np} bn={bn}")

    K_BAND = Kt // Kp
    N_BAND = Nt // Np
    NB = N_BAND // bn

    @ttl.atom(grid=(Np, Kp), fp32_dest_acc_en=fp32_dest_acc_en)
    def gemv(x, w, out):
        x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_BAND), block_count=1)
        x_stage = ttl.make_dataflow_buffer_like(x, shape=(1, K_BAND), block_count=1)
        w_cb = ttl.make_dataflow_buffer_like(w, shape=(K_BAND, bn), block_count=2)
        partial_for_sum_cb = ttl.make_dataflow_buffer_like(out, shape=(1, bn), block_count=2)
        partial_for_send_cb = ttl.make_dataflow_buffer_like(out, shape=(1, bn), block_count=2)
        recv_cb = ttl.make_dataflow_buffer_like(out, shape=(1, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, bn), block_count=2)

        # x: column 0 of each K-band row reads its K-slice once, fans out.
        x_net = ttl.PipeNet([
            ttl.Pipe(src=(0, kp), dst=(slice(0, Np), kp)) for kp in range(Kp)
        ])
        # Partials: row 1 ships its [1, bn] to row 0 in the same column.
        reduce_net = ttl.PipeNet([
            ttl.Pipe(src=(np_, kp), dst=(np_, 0))
            for np_ in range(Np)
            for kp in range(1, Kp)
        ])

        col_c, row_c = ttl.node(dims=2)
        kr = row_c * K_BAND

        mcast(x_net, x[0:1, kr:kr + K_BAND], x_stage, x_cb)
        x_blk = x_cb.wait()

        for lnb in range(NB):
            nc = col_c * N_BAND + lnb * bn

            p = partial_for_sum_cb.reserve()
            w_blk = w_cb.wait()
            p += x_blk @ w_blk

            w_dst = w_cb.reserve()
            ttl.copy(w[kr:kr + K_BAND, nc:nc + bn], w_dst)

            if row_c == 0:
                reduce_recv(reduce_net, partial_for_sum_cb, recv_cb, out_cb)
                ttl.copy(out_cb.wait(), out[0:1, nc:nc + bn])
            else:
                reduce_send(reduce_net, partial_for_sum_cb, partial_for_send_cb)

    return gemv


def make_gemv_band_core(K_CH, n_ch, bn):
    """Inlinable GEMV band: accumulate ``x_in`` chunks against streamed
    ``w_in`` blocks into one ``[1, bn]`` partial in ``p_out``. The caller
    wires x chunks (pipes or DRAM) and streams W."""

    @ttl.atom()
    def gemv_band_core(x_in: ttl.DFB, w_in: ttl.DFB, p_out: ttl.DFB):
        p = p_out.reserve()
        for ch in range(n_ch):
            xb = x_in.wait()
            wb = w_in.wait()
            p += xb @ wb

    return gemv_band_core
