# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fused single-head attention kernel: score matmul + softmax + output matmul.

Replaces 5 separate kernel calls (score_matmul + 3×softmax + output_matmul)
with 1 fused kernel call. Uses DRAM scratch for softmax intermediates.

Called 14 times (once per Q head) from the Python attention loop.
With batch RoPE (2 calls) + fused RMSNorm (2 calls), total attention
becomes 14 calls instead of 70.
"""

import torch
import ttl
import ttnn

TILE = 32
CACHE_TILES = 16  # max_seq / TILE = 512/32


@ttl.kernel(grid=(1, 1))
def fused_attn_head_kernel(Q_rot, K_T, V, mask, scaler, scratch_a, scratch_b, attn_out):
    """Fused attention for 1 Q head: score + softmax + output.

    Q_rot:     [TILE, head_dim] = [32, 64] — single head Q, pre-rotated
    K_T:       [head_dim, max_seq] = [64, 512] = [2, 16] tiles
    V:         [max_seq, head_dim] = [512, 64] = [16, 2] tiles
    mask:      [TILE, max_seq] = [32, 512] = [1, 16] tiles
    scaler:    [TILE, TILE] ones
    scratch_a: [TILE, max_seq] — DRAM scratch for scores/masked/exp
    scratch_b: [TILE, max_seq] — DRAM scratch for normalized weights
    attn_out:  [TILE, head_dim] = [32, 64] — output
    """
    Kt_q = 2   # head_dim / TILE
    Nt_s = K_T.shape[1] // TILE  # 16
    Kt_v = Nt_s  # 16
    Nt_v = Kt_q  # 2

    # DFBs
    q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    # Compute-local
    acc_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    score_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    masked_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    mx_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    mx_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    exp_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    exp_local_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    sum_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    sum_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(attn_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        # Phase 1: Q + K for score matmul (K-inner per output tile)
        for nt in range(Nt_s):
            for kt in range(Kt_q):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(Q_rot[0, kt], blk)
                    tx.wait()
                with k_dfb.reserve() as blk:
                    tx = ttl.copy(K_T[kt, nt], blk)
                    tx.wait()

        # Phase 2: scaler + mask
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()
        for nt in range(Nt_s):
            with m_dfb.reserve() as blk:
                tx = ttl.copy(mask[0, nt], blk)
                tx.wait()

        # Phase 3: re-read masked from scratch_a
        for nt in range(Nt_s):
            with score_dfb.reserve() as blk:
                tx = ttl.copy(scratch_a[0, nt], blk)
                tx.wait()

        # Phase 4: re-read exp from scratch_a
        for nt in range(Nt_s):
            with score_dfb.reserve() as blk:
                tx = ttl.copy(scratch_a[0, nt], blk)
                tx.wait()

        # Phase 5: weights from scratch_b + V for output matmul
        for nt_out in range(Nt_v):
            for kt_v in range(Kt_v):
                with w_dfb.reserve() as blk:
                    tx = ttl.copy(scratch_b[0, kt_v], blk)
                    tx.wait()
                with v_dfb.reserve() as blk:
                    tx = ttl.copy(V[kt_v, nt_out], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        # Phase 1: scores = Q @ K^T
        for _ in range(Nt_s):
            with q_dfb.wait() as q0, k_dfb.wait() as k0:
                with acc_dfb.reserve() as acc:
                    acc.store(q0 @ k0)
            with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + q1 @ k1)
            with acc_dfb.wait() as score:
                with score_dfb.reserve() as s:
                    s.store(score)

        # Phase 2: softmax — mask + max
        with sc_dfb.wait() as sc_blk:
            with score_dfb.wait() as s, m_dfb.wait() as m:
                with masked_dfb.reserve() as msk:
                    msk.store(s + m)
            with masked_dfb.wait() as msk_blk:
                with tmp_dfb.reserve() as tmp:
                    tmp.store(ttl.math.reduce_max(msk_blk, sc_blk, tmp, dims=[1]))
                with tmp_dfb.wait() as rd:
                    with mx_acc_dfb.reserve() as mx:
                        mx.store(rd)

            for _ in range(Nt_s - 1):
                with score_dfb.wait() as s, m_dfb.wait() as m:
                    with masked_dfb.reserve() as msk:
                        msk.store(s + m)
                with masked_dfb.wait() as msk_blk:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_max(msk_blk, sc_blk, tmp, dims=[1]))
                    with tmp_dfb.wait() as rd, mx_acc_dfb.wait() as prev:
                        with mx_acc_dfb.reserve() as mx:
                            mx.store(ttl.math.max(prev, rd))

            with mx_acc_dfb.wait() as max_blk:
                with mx_bc_dfb.reserve() as mx_bc:
                    mx_bc.store(ttl.math.broadcast(max_blk, mx_bc, dims=[0, 1]))

            # Phase 3: exp + sum
            with mx_bc_dfb.wait() as max_bc:
                with score_dfb.wait() as masked_blk:
                    with exp_dfb.reserve() as e:
                        e.store(ttl.math.exp(masked_blk - max_bc))
                    with exp_local_dfb.reserve() as el:
                        el.store(ttl.math.exp(masked_blk - max_bc))
                with exp_local_dfb.wait() as el_blk:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_sum(el_blk, sc_blk, tmp, dims=[1]))
                    with tmp_dfb.wait() as rd:
                        with sum_acc_dfb.reserve() as sm:
                            sm.store(rd)

                for _ in range(Nt_s - 1):
                    with score_dfb.wait() as masked_blk:
                        with exp_dfb.reserve() as e:
                            e.store(ttl.math.exp(masked_blk - max_bc))
                        with exp_local_dfb.reserve() as el:
                            el.store(ttl.math.exp(masked_blk - max_bc))
                    with exp_local_dfb.wait() as el_blk:
                        with tmp_dfb.reserve() as tmp:
                            tmp.store(ttl.math.reduce_sum(el_blk, sc_blk, tmp, dims=[1]))
                        with tmp_dfb.wait() as rd, sum_acc_dfb.wait() as prev:
                            with sum_acc_dfb.reserve() as sm:
                                sm.store(prev + rd)

            with sum_acc_dfb.wait() as sum_blk:
                with sum_bc_dfb.reserve() as s_bc:
                    s_bc.store(ttl.math.broadcast(sum_blk, s_bc, dims=[0, 1]))

            # Phase 4: normalize
            with sum_bc_dfb.wait() as sum_bc:
                for _ in range(Nt_s):
                    with score_dfb.wait() as exp_blk:
                        with exp_dfb.reserve() as w:
                            w.store(exp_blk * ttl.math.recip(sum_bc))

        # Phase 5: output = weights @ V
        for _ in range(Nt_v):
            with w_dfb.wait() as w_blk, v_dfb.wait() as v_blk:
                with acc_dfb.reserve() as acc:
                    acc.store(w_blk @ v_blk)
            for _ in range(Kt_v - 1):
                with w_dfb.wait() as w_blk, v_dfb.wait() as v_blk, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + w_blk @ v_blk)
            with acc_dfb.wait() as result:
                with out_dfb.reserve() as out:
                    out.store(result)

    @ttl.datamovement()
    def write():
        # Phase 1: scores to scratch_a
        for nt in range(Nt_s):
            with score_dfb.wait() as blk:
                tx = ttl.copy(blk, scratch_a[0, nt])
                tx.wait()
        # Phase 2: masked to scratch_a (overwrites scores — safe, reader done with them)
        for nt in range(Nt_s):
            with masked_dfb.wait() as blk:
                tx = ttl.copy(blk, scratch_a[0, nt])
                tx.wait()
        # Phase 3: exp to scratch_a (overwrites masked — safe, reader done with them)
        for nt in range(Nt_s):
            with exp_dfb.wait() as blk:
                tx = ttl.copy(blk, scratch_a[0, nt])
                tx.wait()
        # Phase 4: normalized weights to scratch_b (DIFFERENT tensor — no race!)
        for nt in range(Nt_s):
            with exp_dfb.wait() as blk:
                tx = ttl.copy(blk, scratch_b[0, nt])
                tx.wait()
        # Phase 5: output
        for nt in range(Nt_v):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, attn_out[0, nt])
                tx.wait()


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_fused_attn(device):
    """Test fused attention for single head."""
    import math
    print("  fused_attn_head (1 head)...", end="", flush=True)

    Q_t = torch.randn(TILE, 64, dtype=torch.bfloat16) * 0.01
    K_T_t = torch.randn(64, 512, dtype=torch.bfloat16) * 0.01
    V_t = torch.randn(512, 64, dtype=torch.bfloat16) * 0.01
    mask_t = torch.full((TILE, 512), float("-inf"), dtype=torch.bfloat16)
    mask_t[0, :50] = 0.0
    sc_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)

    Q = _to_device(Q_t, device)
    K_T = _to_device(K_T_t, device)
    V = _to_device(V_t, device)
    mask = _to_device(mask_t, device)
    sc = _to_device(sc_t, device)
    scratch_a = _to_device(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)
    scratch_b = _to_device(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)
    out = _to_device(torch.zeros(TILE, 64, dtype=torch.bfloat16), device)

    fused_attn_head_kernel(Q, K_T, V, mask, sc, scratch_a, scratch_b, out)
    result = ttnn.to_torch(out)

    # Reference
    scores = Q_t[0:1, :64].float() @ K_T_t.float() + mask_t[0:1].float()
    weights = torch.nn.functional.softmax(scores, dim=-1)
    expected = (weights @ V_t.float()).bfloat16()

    pcc = torch.corrcoef(
        torch.stack([result[0, :64].float(), expected[0].float()])
    )[0, 1].item()
    print(f" PCC={pcc:.4f} {'PASS' if pcc > 0.98 else 'FAIL'}")
    return pcc > 0.98


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("Fused attention tests:")
        test_fused_attn(device)
    finally:
        ttnn.close_device(device)
