# Test group attention kernel in pieces to find where it hangs.

import torch
import ttl
import ttnn

TILE = 32
HEADS_PER_GROUP = 7
HEAD_DIM_TILES = 2


def td(t, d):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# =========================================================================
# Piece 1: Score matmul with column-offset Q indexing (1 head)
# Q_rot[0, col_offset:col_offset+2] @ K_cache_T → scores
# =========================================================================
@ttl.kernel(grid=(1, 1))
def score_matmul_coloffset_kernel(Q_rot, K_cache_T, scores_out):
    """Score = Q_head @ K^T where Q_head is at columns 0,1 of Q_rot."""
    Kt = HEAD_DIM_TILES  # 2
    Nt = K_cache_T.shape[1] // TILE

    q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_cache_T, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(scores_out, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(scores_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        # Read Q tiles at col 0,1 (head 0), then K tiles for K-accumulation
        for kt in range(Kt):
            with q_dfb.reserve() as blk:
                tx = ttl.copy(Q_rot[0, kt], blk)  # column-offset indexing
                tx.wait()
            for nt in range(Nt):
                with k_dfb.reserve() as blk:
                    tx = ttl.copy(K_cache_T[kt, nt], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        # First K tile: Q[0] @ K[0, :] → init scores
        with q_dfb.wait() as q_blk:
            for _ in range(Nt):
                with k_dfb.wait() as k_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(q_blk @ k_blk)

        # Second K tile: accumulate Q[1] @ K[1, :]
        with q_dfb.wait() as q_blk:
            for _ in range(Nt):
                with k_dfb.wait() as k_blk, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + q_blk @ k_blk)

        # Write output
        for _ in range(Nt):
            with acc_dfb.wait() as result:
                with out_dfb.reserve() as out:
                    out.store(result)

    @ttl.datamovement()
    def write():
        for nt in range(Nt):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, scores_out[0, nt])
                tx.wait()


# =========================================================================
# Piece 2: Score matmul looped over 2 heads (column offset varies)
# =========================================================================
@ttl.kernel(grid=(1, 1))
def score_matmul_2heads_kernel(Q_rot, K_cache_T, scores_out):
    """Score matmul for 2 Q heads, writing to same scores_out (overwritten)."""
    Kt = HEAD_DIM_TILES
    Nt = K_cache_T.shape[1] // TILE

    q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_cache_T, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(scores_out, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(scores_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for head in range(2):
            col_base = head * HEAD_DIM_TILES
            for kt in range(Kt):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(Q_rot[0, col_base + kt], blk)
                    tx.wait()
                for nt in range(Nt):
                    with k_dfb.reserve() as blk:
                        tx = ttl.copy(K_cache_T[kt, nt], blk)
                        tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(2):
            with q_dfb.wait() as q_blk:
                for _ in range(Nt):
                    with k_dfb.wait() as k_blk:
                        with acc_dfb.reserve() as acc:
                            acc.store(q_blk @ k_blk)
            with q_dfb.wait() as q_blk:
                for _ in range(Nt):
                    with k_dfb.wait() as k_blk, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + q_blk @ k_blk)
            for _ in range(Nt):
                with acc_dfb.wait() as result:
                    with out_dfb.reserve() as out:
                        out.store(result)

    @ttl.datamovement()
    def write():
        for _ in range(2):
            for nt in range(Nt):
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, scores_out[0, nt])
                    tx.wait()


# =========================================================================
# Piece 3: Score + fused softmax for 1 head
# =========================================================================
@ttl.kernel(grid=(1, 1))
def score_softmax_1head_kernel(Q_rot, K_cache_T, mask, scaler, weights_out):
    """Score matmul + fused softmax for 1 head."""
    Kt = HEAD_DIM_TILES
    Nt = K_cache_T.shape[1] // TILE

    q_dfb = ttl.make_dataflow_buffer_like(Q_rot, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_cache_T, shape=(1, 1), buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
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
    out_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        # Phase 1: Q + K for score matmul
        for kt in range(Kt):
            with q_dfb.reserve() as blk:
                tx = ttl.copy(Q_rot[0, kt], blk)
                tx.wait()
            for nt in range(Nt):
                with k_dfb.reserve() as blk:
                    tx = ttl.copy(K_cache_T[kt, nt], blk)
                    tx.wait()
        # Phase 2: mask + scaler for softmax
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()
        for nt in range(Nt):
            with m_dfb.reserve() as blk:
                tx = ttl.copy(mask[0, nt], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        # Phase 1: score matmul
        with q_dfb.wait() as q_blk:
            for _ in range(Nt):
                with k_dfb.wait() as k_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(q_blk @ k_blk)
        with q_dfb.wait() as q_blk:
            for _ in range(Nt):
                with k_dfb.wait() as k_blk, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + q_blk @ k_blk)

        # Phase 2: fused softmax (mask+max, exp+sum, normalize)
        with sc_dfb.wait() as sc_blk:
            # 2a: mask + max
            with acc_dfb.wait() as s:
                with m_dfb.wait() as m:
                    with masked_dfb.reserve() as msk:
                        msk.store(s + m)
                with masked_dfb.wait() as msk_blk:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_max(msk_blk, sc_blk, tmp, dims=[1]))
                    with tmp_dfb.wait() as rd:
                        with mx_acc_dfb.reserve() as mx:
                            mx.store(rd)

            for _ in range(Nt - 1):
                with acc_dfb.wait() as s:
                    with m_dfb.wait() as m:
                        with masked_dfb.reserve() as msk:
                            msk.store(s + m)
                    with masked_dfb.wait() as msk_blk:
                        with tmp_dfb.reserve() as tmp:
                            tmp.store(ttl.math.reduce_max(msk_blk, sc_blk, tmp, dims=[1]))
                        with tmp_dfb.wait() as rd, mx_acc_dfb.wait() as prev:
                            with mx_acc_dfb.reserve() as mx:
                                mx.store(ttl.math.max(prev, rd))

            # Broadcast max
            with mx_acc_dfb.wait() as max_blk:
                with mx_bc_dfb.reserve() as mx_bc:
                    mx_bc.store(ttl.math.broadcast(max_blk, mx_bc, dims=[0, 1]))

            # 2b: exp + sum (re-read masked scores from acc_dfb... wait, they were consumed)
            # BUG: masked scores were consumed in phase 2a. Need DRAM scratch.
            # For this test, skip the full softmax — just verify score+mask+max works.
            pass

        # Output scores (pre-softmax for testing)
        # Actually let's just output the max for verification
        with mx_bc_dfb.wait() as mx:
            with out_dfb.reserve() as out:
                out.store(mx)

    @ttl.datamovement()
    def write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, weights_out[0, 0])
            tx.wait()


# =========================================================================
# Main: test pieces progressively
# =========================================================================
def main():
    device = ttnn.open_device(device_id=0)
    try:
        Q_t = torch.randn(TILE, 896, dtype=torch.bfloat16) * 0.01
        K_T_t = torch.randn(64, 512, dtype=torch.bfloat16) * 0.01
        V_t = torch.randn(512, 64, dtype=torch.bfloat16) * 0.01
        mask_t = torch.full((TILE, 512), float("-inf"), dtype=torch.bfloat16)
        mask_t[0, :50] = 0.0
        sc_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)

        Q = td(Q_t, device)
        K_T = td(K_T_t, device)

        # Piece 1: single head score matmul
        print("Piece 1: score matmul (1 head, col offset)...", end="", flush=True)
        scores = td(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)
        score_matmul_coloffset_kernel(Q, K_T, scores)
        r = ttnn.to_torch(scores)
        expected = Q_t[0:1, :64].float() @ K_T_t.float()
        pcc = torch.corrcoef(torch.stack([r[0].float(), expected[0]]))[0, 1].item()
        print(f" PCC={pcc:.4f} {'PASS' if pcc > 0.98 else 'FAIL'}")

        # Piece 2: 2 heads looped
        print("Piece 2: score matmul (2 heads looped)...", end="", flush=True)
        scores2 = td(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)
        score_matmul_2heads_kernel(Q, K_T, scores2)
        r2 = ttnn.to_torch(scores2)
        expected2 = Q_t[0:1, 64:128].float() @ K_T_t.float()  # last head written
        pcc2 = torch.corrcoef(torch.stack([r2[0].float(), expected2[0]]))[0, 1].item()
        print(f" PCC={pcc2:.4f} (head 1) {'PASS' if pcc2 > 0.98 else 'FAIL'}")

        # Piece 3: score + softmax max (1 head)
        print("Piece 3: score + softmax max (1 head)...", end="", flush=True)
        wout = td(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)
        score_softmax_1head_kernel(Q, K_T, td(mask_t, device), td(sc_t, device), wout)
        r3 = ttnn.to_torch(wout)
        # Expected: max of (Q[0,:64] @ K_T + mask)[0, :]
        ref_scores = (Q_t[0:1, :64].float() @ K_T_t.float() + mask_t[0:1].float())
        ref_max = ref_scores[0, :50].max().item()
        print(f" max={r3[0,0].item():.4f} expected~{ref_max:.4f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
