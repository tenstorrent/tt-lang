# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Indexed GEMV: stream weight slices selected by a runtime index tensor.

The MoE expert primitive: ``W`` stacks ``E`` weight matrices ``[Kt, Nt]``
row-major; ``idx`` holds ``topk`` expert ids as floats. Per active expert
the slice ``W[e*Kt:(e+1)*Kt, :]`` streams to the per-core ``(K_BAND, bn)``
DFB exactly like ttl.ops.gemv. The same id tile is replicated to every core
by the host, so each DM thread reads it with ttl.read_index without a
broadcast.

Outputs ``topk`` stacked rows: out[t] = x @ W[idx[t]]; the caller folds in
gate weights and the local-expert mask.
"""

from typing import Tuple

import ttl
from ttl.ops.matmul import reduce_recv, reduce_send
from ttl.ops.mcast import mcast

TILE = 32


def make_indexed_gemv(
    E: int,
    K: int,
    N: int,
    topk: int,
    grid_cfg: Tuple[int, int],
    bn: int,
    *,
    fp32_dest_acc_en: bool = True,
):
    """x[1t,K] @ W[idx[t]] for each of ``topk`` indices -> out[t, N]."""
    Np, Kp = grid_cfg

    if Kp != 2:
        raise ValueError(f"make_indexed_gemv currently supports Kp == 2, got Kp={Kp}")
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
    def indexed_gemv(x, idx, w, out):
        x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, K_BAND), block_count=1)
        x_stage = ttl.make_dataflow_buffer_like(x, shape=(1, K_BAND), block_count=1)
        idx_cb = ttl.make_dataflow_buffer_like(idx, shape=(1, 1), block_count=1)
        w_cb = ttl.make_dataflow_buffer_like(w, shape=(K_BAND, bn), block_count=2)
        partial_for_sum_cb = ttl.make_dataflow_buffer_like(out, shape=(1, bn), block_count=2)
        partial_for_send_cb = ttl.make_dataflow_buffer_like(out, shape=(1, bn), block_count=2)
        recv_cb = ttl.make_dataflow_buffer_like(out, shape=(1, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, bn), block_count=2)

        x_net = ttl.PipeNet([
            ttl.Pipe(src=(0, kp), dst=(slice(0, Np), kp)) for kp in range(Kp)
        ])
        reduce_net = ttl.PipeNet([
            ttl.Pipe(src=(np_, kp), dst=(np_, 0))
            for np_ in range(Np)
            for kp in range(1, Kp)
        ])

        col_c, row_c = ttl.node(dims=2)
        kr = row_c * K_BAND

        mcast(x_net, x[0:1, kr:kr + K_BAND], x_stage, x_cb)
        x_blk = x_cb.wait()

        idxd = idx_cb.reserve()
        ttl.copy(idx[0, 0], idxd)
        idx_blk = idx_cb.wait()

        for t in range(topk):
            e = ttl.read_index(idx_blk, 0, t)
            for lnb in range(NB):
                nc = col_c * N_BAND + lnb * bn

                p = partial_for_sum_cb.reserve()
                w_blk = w_cb.wait()
                p += x_blk @ w_blk

                w_dst = w_cb.reserve()
                ttl.copy(w[e * Kt + kr:e * Kt + kr + K_BAND, nc:nc + bn], w_dst)

                if row_c == 0:
                    reduce_recv(reduce_net, partial_for_sum_cb, recv_cb, out_cb)
                    ttl.copy(out_cb.wait(), out[t:t + 1, nc:nc + bn])
                else:
                    reduce_send(reduce_net, partial_for_sum_cb, partial_for_send_cb)

    return indexed_gemv
