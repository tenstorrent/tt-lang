# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Flash attention simple fused test.

Exercises the broadcast-after-reduce pattern with multi-store CBs (init +
loop update) that previously triggered a spurious 'ambiguous reduce trace'
error inside the pattern rewriter.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1

TILE = 32
HD_TILES = 8  # head_dim=256 / 32
N_HEADS = 8
N_KV = 2
GQA = N_HEADS // N_KV
KV_CHUNK = 1


@ttl.operation(grid=(N_HEADS, 1))
def flash_attention(
    Q_all,
    K_all,
    V_all,
    scale_tile,
    scaler,
    neg_inf_tile,
    zero_tile,
    zero_head,
    mask,
    out,
):
    kv_seq_tiles = K_all.shape[0] // N_KV // TILE
    n_chunks = kv_seq_tiles // KV_CHUNK

    q_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, HD_TILES), block_count=1)
    k_dfb = ttl.make_dataflow_buffer_like(
        K_all, shape=(KV_CHUNK, HD_TILES), block_count=2
    )
    v_dfb = ttl.make_dataflow_buffer_like(
        V_all, shape=(KV_CHUNK, HD_TILES), block_count=2
    )
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=1)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
    ninf_dfb = ttl.make_dataflow_buffer_like(neg_inf_tile, shape=(1, 1), block_count=1)
    zero_dfb = ttl.make_dataflow_buffer_like(zero_tile, shape=(1, 1), block_count=1)
    zero_head_dfb = ttl.make_dataflow_buffer_like(
        zero_head, shape=(1, HD_TILES), block_count=1
    )
    mask_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, KV_CHUNK), block_count=2)

    kt_dfb = ttl.make_dataflow_buffer_like(
        K_all, shape=(HD_TILES, KV_CHUNK), block_count=2
    )
    qk_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, KV_CHUNK), block_count=2)
    scaled_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, KV_CHUNK), block_count=2)
    chunk_max_dfb = ttl.make_dataflow_buffer_like(
        scale_tile, shape=(1, 1), block_count=2
    )
    m_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
    alpha_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
    m_new_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
    m_bcast_dfb = ttl.make_dataflow_buffer_like(
        mask, shape=(1, KV_CHUNK), block_count=2
    )
    alpha_bcast_dfb = ttl.make_dataflow_buffer_like(
        Q_all, shape=(1, HD_TILES), block_count=2
    )
    exp_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, KV_CHUNK), block_count=2)
    chunk_sum_dfb = ttl.make_dataflow_buffer_like(
        scale_tile, shape=(1, 1), block_count=2
    )
    l_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
    o_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, HD_TILES), block_count=2)
    o_corr_dfb = ttl.make_dataflow_buffer_like(
        Q_all, shape=(1, HD_TILES), block_count=2
    )
    pv_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, HD_TILES), block_count=2)
    l_bcast_dfb = ttl.make_dataflow_buffer_like(
        Q_all, shape=(1, HD_TILES), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HD_TILES), block_count=2)

    @ttl.compute()
    def compute():
        nx, ny = ttl.node(dims=2)
        h = ny * N_HEADS + nx

        with (
            q_dfb.wait() as q,
            sc_dfb.wait() as scale,
            scaler_dfb.wait() as sclr,
            ninf_dfb.wait() as ninf,
            zero_dfb.wait() as zero,
            zero_head_dfb.wait() as zh,
        ):
            with m_dfb.reserve() as m_init:
                m_init.store(ninf)
            with l_dfb.reserve() as l_init:
                l_init.store(zero)
            with o_dfb.reserve() as o_init:
                o_init.store(zh)

            for c in range(n_chunks):
                with k_dfb.wait() as kc, kt_dfb.reserve() as kt:
                    kt.store(ttl.transpose(kc))
                with kt_dfb.wait() as kt, qk_dfb.reserve() as qk:
                    qk.store(q @ kt)
                with (
                    qk_dfb.wait() as qk,
                    mask_dfb.wait() as msk,
                    scaled_dfb.reserve() as sc_out,
                ):
                    sc_out.store(scale * qk + msk)

                with scaled_dfb.wait() as sv:
                    with chunk_max_dfb.reserve() as cm:
                        cm.store(ttl.math.reduce_max(sv, sclr, dims=[1]))
                    with scaled_dfb.reserve() as sv_copy:
                        sv_copy.store(sv)

                with m_dfb.wait() as m_old:
                    with chunk_max_dfb.wait() as cm:
                        with m_new_dfb.reserve() as mn:
                            mn.store(ttl.math.max(m_old, cm))
                    with m_new_dfb.wait() as mn:
                        with alpha_dfb.reserve() as alpha:
                            alpha.store(ttl.math.exp(m_old - mn))
                        with m_bcast_dfb.reserve() as mn_bc:
                            mn_bc.store(ttl.math.broadcast(mn, mn_bc, dims=[1]))
                        with m_dfb.reserve() as m_next:
                            m_next.store(mn)

                with (
                    scaled_dfb.wait() as sv2,
                    m_bcast_dfb.wait() as mn_bc,
                    exp_dfb.reserve() as ex,
                ):
                    ex.store(ttl.math.exp(sv2 - mn_bc))

                with exp_dfb.wait() as ex:
                    with chunk_sum_dfb.reserve() as cs:
                        cs.store(ttl.math.reduce_sum(ex, sclr, dims=[1]))
                    with exp_dfb.reserve() as ex_copy:
                        ex_copy.store(ex)

                with (
                    alpha_dfb.wait() as alph,
                    l_dfb.wait() as l_old,
                    chunk_sum_dfb.wait() as cs,
                ):
                    with l_dfb.reserve() as l_new:
                        l_new.store(alph * l_old + cs)
                    with alpha_bcast_dfb.reserve() as ab:
                        ab.store(ttl.math.broadcast(alph, ab, dims=[1]))
                with (
                    alpha_bcast_dfb.wait() as ab,
                    o_dfb.wait() as o_old,
                    o_corr_dfb.reserve() as oc,
                ):
                    oc.store(ab * o_old)

                with exp_dfb.wait() as ex2, v_dfb.wait() as vc, pv_dfb.reserve() as pv:
                    pv.store(ex2 @ vc)

                with (
                    o_corr_dfb.wait() as oc,
                    pv_dfb.wait() as pv,
                    o_dfb.reserve() as o_new,
                ):
                    o_new.store(oc + pv)

            with l_dfb.wait() as l_final, l_bcast_dfb.reserve() as lb:
                lb.store(ttl.math.broadcast(l_final, lb, dims=[1]))
            with (
                o_dfb.wait() as o_final,
                l_bcast_dfb.wait() as lb,
                out_dfb.reserve() as final_out,
            ):
                final_out.store(o_final * ttl.math.recip(lb))

    @ttl.datamovement()
    def dm_read():
        nx, ny = ttl.node(dims=2)
        h = ny * N_HEADS + nx

        with q_dfb.reserve() as blk:
            tx = ttl.copy(Q_all[h, 0:HD_TILES], blk)
            tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scale_tile[0, 0], blk)
            tx.wait()
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()
        with ninf_dfb.reserve() as blk:
            tx = ttl.copy(neg_inf_tile[0, 0], blk)
            tx.wait()
        with zero_dfb.reserve() as blk:
            tx = ttl.copy(zero_tile[0, 0], blk)
            tx.wait()
        with zero_head_dfb.reserve() as blk:
            tx = ttl.copy(zero_head[0, 0:HD_TILES], blk)
            tx.wait()

        kv_base = (h // GQA) * kv_seq_tiles
        for c in range(n_chunks):
            kv_off = kv_base + c * KV_CHUNK
            with k_dfb.reserve() as blk:
                tx = ttl.copy(K_all[kv_off : kv_off + KV_CHUNK, 0:HD_TILES], blk)
                tx.wait()
            with mask_dfb.reserve() as blk:
                tx = ttl.copy(mask[0, c * KV_CHUNK : (c + 1) * KV_CHUNK], blk)
                tx.wait()
            with v_dfb.reserve() as blk:
                tx = ttl.copy(V_all[kv_off : kv_off + KV_CHUNK, 0:HD_TILES], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        nx, ny = ttl.node(dims=2)
        h = ny * N_HEADS + nx
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[h, 0:HD_TILES])
            tx.wait()


def test_flash_attention_ambiguous_store_broadcast(device):
    hd = 256
    padded_seq = 128

    zeros = lambda shape: to_dram(torch.zeros(shape, dtype=torch.bfloat16), device)
    scalar = to_l1(torch.full((TILE, TILE), 1.0, dtype=torch.bfloat16), device)

    out_torch = torch.zeros(N_HEADS * TILE, hd, dtype=torch.bfloat16)
    out_t = to_dram(out_torch, device)

    flash_attention(
        zeros((N_HEADS * TILE, hd)),  # Q
        zeros((N_KV * padded_seq, hd)),  # K
        zeros((N_KV * padded_seq, hd)),  # V
        scalar,  # scale
        scalar,  # scaler
        scalar,  # neg_inf
        scalar,  # zero
        zeros((TILE, hd)),  # zero_head
        zeros((TILE, padded_seq)),  # mask
        out_t,  # out
    )

    result = ttnn.to_torch(out_t).float()
    assert result.shape == out_torch.shape
