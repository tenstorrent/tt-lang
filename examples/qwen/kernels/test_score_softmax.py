"""Test score matmul + fused softmax for 1 head in single kernel."""
import math
import torch
import ttl
import ttnn

TILE = 32


def td(t, d):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.kernel(grid=(1, 1))
def score_softmax_kernel(Q, K_T, mask, scaler, scratch, weights_out):
    """Score matmul + fused softmax for 1 Q head.

    Q:           [TILE, Nt_q*TILE] — Q_combined, read cols 0:2 for head 0
    K_T:         [head_dim=64, max_seq=512] = [2, 16] tiles
    mask:        [TILE, max_seq=512] = [1, 16] tiles
    scaler:      [TILE, TILE] ones
    scratch:     [TILE, max_seq=512] = [1, 16] tiles — DRAM scratch
    weights_out: [TILE, max_seq=512] — output softmax weights
    """
    Kt = 2
    Nt = K_T.shape[1] // TILE  # 16

    # DFBs
    q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    # Score/softmax compute-local
    acc_dfb = ttl.make_dataflow_buffer_like(weights_out, shape=(1, 1), buffer_factor=2)
    score_dfb = ttl.make_dataflow_buffer_like(weights_out, shape=(1, 1), buffer_factor=2)
    masked_dfb = ttl.make_dataflow_buffer_like(weights_out, shape=(1, 1), buffer_factor=2)
    mx_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    mx_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    exp_dfb = ttl.make_dataflow_buffer_like(weights_out, shape=(1, 1), buffer_factor=2)
    exp_local_dfb = ttl.make_dataflow_buffer_like(weights_out, shape=(1, 1), buffer_factor=2)
    sum_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    sum_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(weights_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        # Phase 1: Q + K tiles for score matmul (K-inner loop)
        for nt in range(Nt):
            for kt in range(Kt):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(Q[0, kt], blk)
                    tx.wait()
                with k_dfb.reserve() as blk:
                    tx = ttl.copy(K_T[kt, nt], blk)
                    tx.wait()

        # Phase 2: scaler + mask for softmax pass 1 (mask+max)
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()
        for nt in range(Nt):
            with m_dfb.reserve() as blk:
                tx = ttl.copy(mask[0, nt], blk)
                tx.wait()

        # Phase 3: re-read masked scores from scratch for exp+sum
        for nt in range(Nt):
            with score_dfb.reserve() as blk:
                tx = ttl.copy(scratch[0, nt], blk)
                tx.wait()

        # Phase 4: re-read exp scores from scratch for normalize
        for nt in range(Nt):
            with score_dfb.reserve() as blk:
                tx = ttl.copy(scratch[0, nt], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        # Phase 1: scores = Q @ K^T with K-inner loop
        for _ in range(Nt):
            with q_dfb.wait() as q0, k_dfb.wait() as k0:
                with acc_dfb.reserve() as acc:
                    acc.store(q0 @ k0)
            with q_dfb.wait() as q1, k_dfb.wait() as k1, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + q1 @ k1)
            # Write score to output (will go to scratch via writer)
            with acc_dfb.wait() as score:
                with score_dfb.reserve() as s:
                    s.store(score)

        # Phase 2: softmax — mask+max
        with sc_dfb.wait() as sc_blk:
            # First tile
            with score_dfb.wait() as s, m_dfb.wait() as m:
                with masked_dfb.reserve() as msk:
                    msk.store(s + m)
            with masked_dfb.wait() as msk_blk:
                with tmp_dfb.reserve() as tmp:
                    tmp.store(ttl.math.reduce_max(msk_blk, sc_blk, tmp, dims=[1]))
                with tmp_dfb.wait() as rd:
                    with mx_acc_dfb.reserve() as mx:
                        mx.store(rd)

            # Remaining tiles
            for _ in range(Nt - 1):
                with score_dfb.wait() as s, m_dfb.wait() as m:
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

            # Phase 3: exp + sum (re-read masked scores from scratch)
            with mx_bc_dfb.wait() as max_bc:
                # First tile
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

                # Remaining tiles
                for _ in range(Nt - 1):
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

            # Broadcast sum
            with sum_acc_dfb.wait() as sum_blk:
                with sum_bc_dfb.reserve() as s_bc:
                    s_bc.store(ttl.math.broadcast(sum_blk, s_bc, dims=[0, 1]))

            # Phase 4: normalize (re-read exp from scratch)
            with sum_bc_dfb.wait() as sum_bc:
                for _ in range(Nt):
                    with score_dfb.wait() as exp_blk:
                        with out_dfb.reserve() as out:
                            out.store(exp_blk * ttl.math.recip(sum_bc))

    @ttl.datamovement()
    def write():
        # Phase 1: write scores to scratch (masked scores go there too via phase 2)
        for nt in range(Nt):
            with score_dfb.wait() as blk:
                tx = ttl.copy(blk, scratch[0, nt])
                tx.wait()

        # Phase 2: write masked scores back to scratch
        for nt in range(Nt):
            with masked_dfb.wait() as blk:
                tx = ttl.copy(blk, scratch[0, nt])
                tx.wait()

        # Phase 3: write exp scores to scratch
        for nt in range(Nt):
            with exp_dfb.wait() as blk:
                tx = ttl.copy(blk, scratch[0, nt])
                tx.wait()

        # Phase 4: write final weights
        for nt in range(Nt):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, weights_out[0, nt])
                tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        Q_t = torch.randn(TILE, 128, dtype=torch.bfloat16) * 0.01  # 2 heads worth
        K_T_t = torch.randn(64, 512, dtype=torch.bfloat16) * 0.01
        mask_t = torch.full((TILE, 512), float("-inf"), dtype=torch.bfloat16)
        mask_t[0, :50] = 0.0
        sc_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)

        Q = td(Q_t, device)
        K_T = td(K_T_t, device)
        mask = td(mask_t, device)
        sc = td(sc_t, device)
        scratch = td(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)
        wout = td(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)

        print("Score + fused softmax (1 head)...", end="", flush=True)
        score_softmax_kernel(Q, K_T, mask, sc, scratch, wout)
        r = ttnn.to_torch(wout)

        # Reference
        scores = Q_t[0:1, :64].float() @ K_T_t.float()
        scores = scores + mask_t[0:1].float()
        expected = torch.nn.functional.softmax(scores, dim=-1).bfloat16()

        pcc = torch.corrcoef(torch.stack([r[0].float(), expected[0].float()]))[0, 1].item()
        print(f" PCC={pcc:.4f} sum={r[0].float().sum().item():.3f} {'PASS' if pcc > 0.98 else 'FAIL'}")

    finally:
        ttnn.close_device(device)
