"""Test score matmul with correct K-inner-loop pattern."""
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
def score_1head_kernel(Q, K_T, out):
    """Q[0, 0:2] @ K_T[2, Nt] → out[0, 0:Nt].

    Inner K-loop (K=2), outer output tile loop (Nt=16).
    For each output tile: init from K=0, accumulate K=1, write.
    """
    Kt = 2   # head_dim / TILE
    Nt = K_T.shape[1] // TILE  # max_seq / TILE = 16

    q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        # For each output tile: read Q[0,0], K_T[0,nt], Q[0,1], K_T[1,nt]
        for nt in range(Nt):
            for kt in range(Kt):
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(Q[0, kt], blk)
                    tx.wait()
                with k_dfb.reserve() as blk:
                    tx = ttl.copy(K_T[kt, nt], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Nt):
            # K=0: init
            with q_dfb.wait() as q_blk, k_dfb.wait() as k_blk:
                with acc_dfb.reserve() as acc:
                    acc.store(q_blk @ k_blk)
            # K=1: accumulate
            with q_dfb.wait() as q_blk, k_dfb.wait() as k_blk, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + q_blk @ k_blk)
            # Write
            with acc_dfb.wait() as result:
                with y_dfb.reserve() as out_blk:
                    out_blk.store(result)

    @ttl.datamovement()
    def write():
        for nt in range(Nt):
            with y_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, nt])
                tx.wait()


# Two heads looped: same pattern but Q reads at different column offsets
@ttl.kernel(grid=(1, 1))
def score_2heads_kernel(Q, K_T, out):
    """2 Q heads from Q[0, head*2 : head*2+2], write to same out."""
    Kt = 2
    Nt = K_T.shape[1] // TILE

    q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(1, 1), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_T, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for head in range(2):
            q_base = head * Kt
            for nt in range(Nt):
                for kt in range(Kt):
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(Q[0, q_base + kt], blk)
                        tx.wait()
                    with k_dfb.reserve() as blk:
                        tx = ttl.copy(K_T[kt, nt], blk)
                        tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(2):  # 2 heads
            for _ in range(Nt):
                with q_dfb.wait() as q_blk, k_dfb.wait() as k_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(q_blk @ k_blk)
                with q_dfb.wait() as q_blk, k_dfb.wait() as k_blk, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + q_blk @ k_blk)
                with acc_dfb.wait() as result:
                    with y_dfb.reserve() as out_blk:
                        out_blk.store(result)

    @ttl.datamovement()
    def write():
        for _ in range(2):
            for nt in range(Nt):
                with y_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[0, nt])
                    tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        Q_t = torch.randn(TILE, 896, dtype=torch.bfloat16) * 0.01
        K_T_t = torch.randn(64, 512, dtype=torch.bfloat16) * 0.01

        Q = td(Q_t, device)
        K_T = td(K_T_t, device)

        # Test 1: single head
        print("Test 1: score matmul 1 head (K-inner)...", end="", flush=True)
        out1 = td(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)
        score_1head_kernel(Q, K_T, out1)
        r1 = ttnn.to_torch(out1)
        exp1 = Q_t[0:1, :64].float() @ K_T_t.float()
        pcc1 = torch.corrcoef(torch.stack([r1[0].float(), exp1[0]]))[0, 1].item()
        print(f" PCC={pcc1:.4f} {'PASS' if pcc1 > 0.98 else 'FAIL'}")

        # Test 2: 2 heads looped
        print("Test 2: score matmul 2 heads looped...", end="", flush=True)
        out2 = td(torch.zeros(TILE, 512, dtype=torch.bfloat16), device)
        score_2heads_kernel(Q, K_T, out2)
        r2 = ttnn.to_torch(out2)
        # Last head (head 1) wrote last, so out should match head 1
        exp2 = Q_t[0:1, 64:128].float() @ K_T_t.float()
        pcc2 = torch.corrcoef(torch.stack([r2[0].float(), exp2[0]]))[0, 1].item()
        print(f" PCC={pcc2:.4f} (head 1) {'PASS' if pcc2 > 0.98 else 'FAIL'}")

    finally:
        ttnn.close_device(device)
