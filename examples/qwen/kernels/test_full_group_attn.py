"""Test full group attention: score + softmax + output matmul, 2 heads."""
import math
import torch
import ttl
import ttnn

TILE = 32
NUM_HEADS = 2  # start with 2, scale to 7


def td(t, d):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.kernel(grid=(1, 1))
def group_attn_2heads_kernel_old(Q_rot, K_T, V, mask, scaler, scratch, attn_out):
    """Full attention for 2 Q heads. Column-offset Q read, direct-write output.

    Q_rot:    [TILE, >=4*TILE] — Q combined (>=2 heads × 2 tiles)
    K_T:      [64, 512] = [2, 16] tiles
    V:        [512, 64] = [16, 2] tiles
    mask:     [TILE, 512] = [1, 16] tiles
    scaler:   [TILE, TILE] ones
    scratch:  [TILE, 512] = [1, 16] tiles — DRAM scratch
    attn_out: [TILE, >=4*TILE] — output (heads write at column offsets)
    """
    Kt_q = 2    # head_dim / TILE
    Nt_s = K_T.shape[1] // TILE  # 16 (score tiles)
    Kt_v = Nt_s  # K-dim for output matmul = cache_len / TILE = 16
    Nt_v = 2     # output tiles per head = head_dim / TILE

    q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(scratch, shape=(1, 1), buffer_factor=2)
    score_dfb = ttl.make_dataflow_buffer_like(scratch, shape=(1, 1), buffer_factor=2)
    masked_dfb = ttl.make_dataflow_buffer_like(scratch, shape=(1, 1), buffer_factor=2)
    mx_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    mx_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    exp_dfb = ttl.make_dataflow_buffer_like(scratch, shape=(1, 1), buffer_factor=2)
    exp_local_dfb = ttl.make_dataflow_buffer_like(scratch, shape=(1, 1), buffer_factor=2)
    sum_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    sum_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(scratch, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for head in range(NUM_HEADS):
            q_base = head * Kt_q

            # Phase 1: Q + K for score (K-inner per output tile)
            for nt in range(Nt_s):
                for kt in range(Kt_q):
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(Q_rot[0, q_base + kt], blk)
                        tx.wait()
                    with k_dfb.reserve() as blk:
                        tx = ttl.copy(K_T[kt, nt], blk)
                        tx.wait()

            # Phase 2: scaler + mask for softmax
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk)
                tx.wait()
            for nt in range(Nt_s):
                with m_dfb.reserve() as blk:
                    tx = ttl.copy(mask[0, nt], blk)
                    tx.wait()

            # Phase 3: re-read masked from scratch for exp+sum
            for nt in range(Nt_s):
                with score_dfb.reserve() as blk:
                    tx = ttl.copy(scratch[0, nt], blk)
                    tx.wait()

            # Phase 4: re-read exp from scratch for normalize
            for nt in range(Nt_s):
                with score_dfb.reserve() as blk:
                    tx = ttl.copy(scratch[0, nt], blk)
                    tx.wait()

            # Phase 5: weights + V for output matmul (K-inner per output tile)
            for nt_out in range(Nt_v):
                for kt_v in range(Kt_v):
                    with w_dfb.reserve() as blk:
                        tx = ttl.copy(scratch[0, kt_v], blk)  # weights from scratch
                        tx.wait()
                    with v_dfb.reserve() as blk:
                        tx = ttl.copy(V[kt_v, nt_out], blk)
                        tx.wait()

    @ttl.compute()
    def compute():
        for head in range(NUM_HEADS):
            # Phase 1: scores
            for _ in range(Nt_s):
                with q_dfb.wait() as q0, k_dfb.wait() as k0:
                    with acc_dfb.reserve() as acc:
                        acc.store(q0 @ k0)
                with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + q1 @ k1)
                with acc_dfb.wait() as s:
                    with score_dfb.reserve() as out:
                        out.store(s)

            # Phase 2: softmax (mask+max, exp+sum, normalize)
            with sc_dfb.wait() as sc_blk:
                # mask + max
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

                # exp + sum
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

                # normalize
                with sum_bc_dfb.wait() as sum_bc:
                    for _ in range(Nt_s):
                        with score_dfb.wait() as exp_blk:
                            with exp_dfb.reserve() as w:
                                w.store(exp_blk * ttl.math.recip(sum_bc))

            # Phase 5: output = weights @ V (K-inner per output tile)
            for _ in range(Nt_v):
                # K=0 init
                with w_dfb.wait() as w_blk, v_dfb.wait() as v_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(w_blk @ v_blk)
                # K=1..Kt_v-1 accumulate
                for _ in range(Kt_v - 1):
                    with w_dfb.wait() as w_blk, v_dfb.wait() as v_blk, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + w_blk @ v_blk)
                with acc_dfb.wait() as result:
                    with out_dfb.reserve() as out:
                        out.store(result)

    @ttl.datamovement()
    def write():
        for head in range(NUM_HEADS):
            out_base = head * Kt_q

            # Phase 1: scores to scratch
            for nt in range(Nt_s):
                with score_dfb.wait() as blk:
                    tx = ttl.copy(blk, scratch[0, nt])
                    tx.wait()
            # Phase 2: masked to scratch
            for nt in range(Nt_s):
                with masked_dfb.wait() as blk:
                    tx = ttl.copy(blk, scratch[0, nt])
                    tx.wait()
            # Phase 3: exp to scratch
            for nt in range(Nt_s):
                with exp_dfb.wait() as blk:
                    tx = ttl.copy(blk, scratch[0, nt])
                    tx.wait()
            # Phase 4: normalized weights to scratch (for output matmul reader)
            for nt in range(Nt_s):
                with exp_dfb.wait() as blk:
                    tx = ttl.copy(blk, scratch[0, nt])
                    tx.wait()
            # Phase 5: output to attn_out at column offset
            for nt in range(Nt_v):
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, attn_out[0, out_base + nt])
                    tx.wait()


CACHE_TILES = 16  # max_seq / TILE = 512/32


@ttl.kernel(grid=(1, 1))
def group_attn_2heads_kernel(Q_rot, K_T, V, mask, scaler, attn_out):
    """Full attention for 2 Q heads. NO DRAM scratch — uses L1 DFBs.

    Key: masked_buf and exp_buf have buffer_factor=CACHE_TILES to hold
    all intermediate tiles in L1. This eliminates DRAM scratch and
    the associated race condition.
    """
    Kt_q = 2
    Nt_s = K_T.shape[1] // TILE
    Kt_v = Nt_s
    Nt_v = 2

    q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, 1), buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)

    acc_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)

    # HIGH CAPACITY L1 buffers for softmax intermediates (16 tiles each)
    masked_buf = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=CACHE_TILES)
    exp_buf = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=CACHE_TILES)

    mx_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    mx_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    exp_local_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    sum_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    sum_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for head in range(NUM_HEADS):
            q_base = head * Kt_q

            # Phase 1: Q + K for scores
            for nt in range(Nt_s):
                for kt in range(Kt_q):
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(Q_rot[0, q_base + kt], blk)
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

            # Phase 3 & 4: softmax uses L1 buffers (no DRAM read needed)

            # Phase 5: V tiles for output matmul
            for nt_out in range(Nt_v):
                for kt_v in range(Kt_v):
                    with w_dfb.reserve() as blk:
                        tx = ttl.copy(V[kt_v, nt_out], blk)
                        tx.wait()

    @ttl.compute()
    def compute():
        for head in range(NUM_HEADS):
            # Phase 1: score matmul Q @ K^T
            for _ in range(Nt_s):
                with q_dfb.wait() as q0, k_dfb.wait() as k0:
                    with acc_dfb.reserve() as acc:
                        acc.store(q0 @ k0)
                with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + q1 @ k1)
                # Store score as masked score + mask → masked_buf (L1, capacity 16)
                with acc_dfb.wait() as score:
                    with masked_buf.reserve() as msk:
                        msk.store(score)

            # Phase 2: mask + max (read from masked_buf, compute max, put back for exp)
            with sc_dfb.wait() as sc_blk:
                # Apply mask AND find max in one pass
                # Read from masked_buf (which has raw scores), apply mask, find max
                # But masked_buf already has the scores — we need scores + mask.
                # Let me restructure: in Phase 1, store raw scores to masked_buf.
                # Then here, read them back, add mask, compute max.

                # First tile: masked_buf has raw scores, add mask, store to exp_buf, reduce
                with masked_buf.wait() as raw_score, m_dfb.wait() as m:
                    with exp_buf.reserve() as msk_store:
                        msk_store.store(raw_score + m)
                with exp_buf.wait() as msk_blk:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_max(msk_blk, sc_blk, tmp, dims=[1]))
                    with tmp_dfb.wait() as rd:
                        with mx_acc_dfb.reserve() as mx:
                            mx.store(rd)
                    # Put masked back into masked_buf for phase 3
                    with masked_buf.reserve() as store_back:
                        store_back.store(msk_blk)

                # Remaining tiles
                for _ in range(Nt_s - 1):
                    with masked_buf.wait() as raw_score, m_dfb.wait() as m:
                        with exp_buf.reserve() as msk_store:
                            msk_store.store(raw_score + m)
                    with exp_buf.wait() as msk_blk:
                        with tmp_dfb.reserve() as tmp:
                            tmp.store(ttl.math.reduce_max(msk_blk, sc_blk, tmp, dims=[1]))
                        with tmp_dfb.wait() as rd, mx_acc_dfb.wait() as prev:
                            with mx_acc_dfb.reserve() as mx:
                                mx.store(ttl.math.max(prev, rd))
                        with masked_buf.reserve() as store_back:
                            store_back.store(msk_blk)

                # Broadcast max
                with mx_acc_dfb.wait() as max_blk:
                    with mx_bc_dfb.reserve() as mx_bc:
                        mx_bc.store(ttl.math.broadcast(max_blk, mx_bc, dims=[0, 1]))

                # Phase 3: exp + sum (read masked from exp_buf, write exp back to masked_buf)
                with mx_bc_dfb.wait() as max_bc:
                    # First tile
                    with exp_buf.wait() as masked_blk:
                        with masked_buf.reserve() as e_store:
                            e_store.store(ttl.math.exp(masked_blk - max_bc))
                        with exp_local_dfb.reserve() as el:
                            el.store(ttl.math.exp(masked_blk - max_bc))
                    with exp_local_dfb.wait() as el_blk:
                        with tmp_dfb.reserve() as tmp:
                            tmp.store(ttl.math.reduce_sum(el_blk, sc_blk, tmp, dims=[1]))
                        with tmp_dfb.wait() as rd:
                            with sum_acc_dfb.reserve() as sm:
                                sm.store(rd)

                    # Remaining tiles
                    for _ in range(Nt_s - 1):
                        with exp_buf.wait() as masked_blk:
                            with masked_buf.reserve() as e_store:
                                e_store.store(ttl.math.exp(masked_blk - max_bc))
                            with exp_local_dfb.reserve() as el:
                                el.store(ttl.math.exp(masked_blk - max_bc))
                        with exp_local_dfb.wait() as el_blk:
                            with tmp_dfb.reserve() as tmp:
                                tmp.store(ttl.math.reduce_sum(el_blk, sc_blk, tmp, dims=[1]))
                            with tmp_dfb.wait() as rd, sum_acc_dfb.wait() as prev:
                                with sum_acc_dfb.reserve() as sm:
                                    sm.store(prev + rd)

                # Broadcast sum
                with sum_acc_dfb.wait() as sum_blk:
                    with sum_bc_dfb.reserve() as s_bc:
                        s_bc.store(ttl.math.broadcast(sum_blk, s_bc, dims=[0, 1]))

                # Phase 4: normalize (read exp from masked_buf)
                with sum_bc_dfb.wait() as sum_bc:
                    for _ in range(Nt_s):
                        with masked_buf.wait() as exp_blk:
                            with exp_buf.reserve() as w:
                                w.store(exp_blk * ttl.math.recip(sum_bc))

            # Phase 5: output = weights @ V (read weights from exp_buf)
            for _ in range(Nt_v):
                with exp_buf.wait() as w_blk, w_dfb.wait() as v_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(w_blk @ v_blk)
                for _ in range(Kt_v - 1):
                    with exp_buf.wait() as w_blk, w_dfb.wait() as v_blk, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + w_blk @ v_blk)
                with acc_dfb.wait() as result:
                    with out_dfb.reserve() as out:
                        out.store(result)

    @ttl.datamovement()
    def write():
        for head in range(NUM_HEADS):
            out_base = head * Kt_q
            # Only write final output tiles (no DRAM scratch needed!)
            for nt in range(Nt_v):
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, attn_out[0, out_base + nt])
                    tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        Q_t = torch.randn(TILE, 128, dtype=torch.bfloat16) * 0.01  # 2 heads
        K_T_t = torch.randn(64, 512, dtype=torch.bfloat16) * 0.01
        V_t = torch.randn(512, 64, dtype=torch.bfloat16) * 0.01
        mask_t = torch.full((TILE, 512), float("-inf"), dtype=torch.bfloat16)
        mask_t[0, :50] = 0.0
        sc_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)

        Q = td(Q_t, device)
        K_T = td(K_T_t, device)
        V = td(V_t, device)
        mask = td(mask_t, device)
        sc = td(sc_t, device)
        scratch = td(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)
        attn_out = td(torch.zeros(TILE, 128, dtype=torch.bfloat16), device)

        print("Full group attention (2 heads, L1 buffers)...", end="", flush=True)
        group_attn_2heads_kernel(Q, K_T, V, mask, sc, attn_out)
        r = ttnn.to_torch(attn_out)

        # Reference for both heads
        for h in range(2):
            q_h = Q_t[0:1, h*64:(h+1)*64].float()
            scores = q_h @ K_T_t.float() + mask_t[0:1].float()
            w = torch.nn.functional.softmax(scores, dim=-1)
            exp_h = (w @ V_t.float())[0]
            got = r[0, h*64:(h+1)*64].float()
            pcc = torch.corrcoef(torch.stack([got, exp_h]))[0, 1].item()
            print(f" h{h}:PCC={pcc:.4f}", end="")

        print(" PASS" if pcc > 0.98 else " FAIL")

    finally:
        ttnn.close_device(device)
