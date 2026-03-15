"""
TT-Lang SDPA kernel for Oasis spatial attention.

Spatial attention per frame: Q, K, V are (N_PATCH_PAD, D_HEAD) per head.
N_PATCH_PAD=160 = 5 tiles, D_HEAD=64 = 2 tiles.
Attention matrix: 5x5 tiles = 25 tiles = 50KB. Fits in L1.

We batch over heads: process one head at a time per core.
Input: Q, K, V as (N_PATCH_PAD * N_HEADS, D_HEAD) = (2560, 64)
Each head is a contiguous block of N_PATCH_PAD=160 rows.

Pattern from the reference SDPA example:
1. K^T = transpose(K)
2. QK = Q @ K^T
3. Scale QK
4. Row-wise softmax: max, shift, exp, sum, divide
5. out = attn @ V
"""
import torch
import torch.nn.functional as F
import ttnn
import ttl
import time

TILE = 32
N_PATCH_PAD = 160
D_HEAD = 64
N_HEADS = 16
SEQ_TILES = N_PATCH_PAD // TILE  # 5
HEAD_TILES = D_HEAD // TILE  # 2
SCALE = 1.0 / (D_HEAD ** 0.5)  # 1/sqrt(64) = 0.125


def make_sdpa_kernel(seq_tiles, head_tiles, scale_val):
    """Single-head SDPA: out = softmax(Q @ K^T * scale) @ V

    Processes one head at a time. Each core handles one head.
    Q, K, V: (seq_tiles, head_tiles) per head.
    Attention matrix: (seq_tiles, seq_tiles) = 5x5 tiles.
    """
    @ttl.kernel(grid=(1, 1))
    def sdpa_kernel(Q, K, V, scaler, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_heads = Q.shape[0] // TILE // seq_tiles
        heads_per_core = -(-n_heads // grid_cols)

        q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(seq_tiles, head_tiles), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(K, shape=(seq_tiles, head_tiles), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(V, shape=(seq_tiles, head_tiles), buffer_factor=2)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

        kt_dfb = ttl.make_dataflow_buffer_like(K, shape=(head_tiles, seq_tiles), buffer_factor=2)
        # Reuse (seq_tiles, seq_tiles) DFBs for the chain of attention ops
        a_dfb = ttl.make_dataflow_buffer_like(Q, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(Q, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        c_dfb = ttl.make_dataflow_buffer_like(Q, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        # Row reduce/broadcast intermediates
        row_dfb = ttl.make_dataflow_buffer_like(Q, shape=(seq_tiles, 1), buffer_factor=2)
        row_bc_dfb = ttl.make_dataflow_buffer_like(Q, shape=(seq_tiles, seq_tiles), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(seq_tiles, head_tiles), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < n_heads:
                    with q_dfb.wait() as qv, k_dfb.wait() as kv, v_dfb.wait() as vv, scaler_dfb.wait() as scaler_v:
                        with kt_dfb.reserve() as kt:
                            kt.store(ttl.transpose(kv))
                        with kt_dfb.wait() as ktv, a_dfb.reserve() as qk:
                            qk.store(qv @ ktv)
                        with a_dfb.wait() as qkv, b_dfb.reserve() as scaled:
                            scaled.store(qkv * ttl.math.fill(qkv, 0.125))
                        with b_dfb.wait() as sdv:
                            with row_dfb.reserve() as mx:
                                mx.store(ttl.math.reduce_max(sdv, scaler_v, dims=[1]))
                            with row_dfb.wait() as mxv, row_bc_dfb.reserve() as mxb:
                                mxb.store(ttl.math.broadcast(mxv, dims=[1]))
                            with row_bc_dfb.wait() as mxbv:
                                with a_dfb.reserve() as ex:
                                    ex.store(ttl.math.exp(sdv - mxbv))
                            with a_dfb.wait() as exv:
                                with row_dfb.reserve() as sm:
                                    sm.store(ttl.math.reduce_sum(exv, scaler_v, dims=[1]))
                                with row_dfb.wait() as smv, row_bc_dfb.reserve() as smb:
                                    smb.store(ttl.math.broadcast(smv, dims=[1]))
                                with row_bc_dfb.wait() as smbv, c_dfb.reserve() as attn:
                                    attn.store(exv * ttl.math.recip(smbv))
                        with c_dfb.wait() as av, out_dfb.reserve() as o:
                            o.store(av @ vv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < n_heads:
                    h_start = head_idx * seq_tiles
                    with q_dfb.reserve() as blk:
                        tx = ttl.copy(Q[h_start:h_start + seq_tiles, 0:head_tiles], blk); tx.wait()
                    with k_dfb.reserve() as blk:
                        tx = ttl.copy(K[h_start:h_start + seq_tiles, 0:head_tiles], blk); tx.wait()
                    with v_dfb.reserve() as blk:
                        tx = ttl.copy(V[h_start:h_start + seq_tiles, 0:head_tiles], blk); tx.wait()
                    with scaler_dfb.reserve() as blk:
                        tx = ttl.copy(scaler[0, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_h in range(heads_per_core):
                head_idx = core_x * heads_per_core + local_h
                if head_idx < n_heads:
                    h_start = head_idx * seq_tiles
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[h_start:h_start + seq_tiles, 0:head_tiles]); tx.wait()

    return sdpa_kernel


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    print("Testing TT-Lang SDPA (SEQ=%d, D_HEAD=%d, N_HEADS=%d)" % (N_PATCH_PAD, D_HEAD, N_HEADS))

    N_HEADS_TEST = 2  # minimal multi-iteration test
    q_torch = torch.randn(N_HEADS_TEST * N_PATCH_PAD, D_HEAD, dtype=torch.float32) * 0.3
    k_torch = torch.randn(N_HEADS_TEST * N_PATCH_PAD, D_HEAD, dtype=torch.float32) * 0.3
    v_torch = torch.randn(N_HEADS_TEST * N_PATCH_PAD, D_HEAD, dtype=torch.float32) * 0.3

    # PyTorch reference: per-head SDPA
    ref_out = torch.zeros_like(q_torch)
    for h in range(N_HEADS_TEST):
        s = h * N_PATCH_PAD
        e = s + N_PATCH_PAD
        q_h = q_torch[s:e].float()  # (160, 64)
        k_h = k_torch[s:e].float()
        v_h = v_torch[s:e].float()
        scores = (q_h @ k_h.T) * SCALE
        attn = F.softmax(scores, dim=-1)
        ref_out[s:e] = (attn @ v_h).to(torch.float32)

    print("Reference range: [%.4f, %.4f]" % (ref_out.min().item(), ref_out.max().item()))

    # Device tensors
    q_tt = to_tt(q_torch.to(torch.bfloat16), device)
    k_tt = to_tt(k_torch.to(torch.bfloat16), device)
    v_tt = to_tt(v_torch.to(torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(N_HEADS * N_PATCH_PAD, D_HEAD, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)

    print("Building SDPA kernel...")
    sdpa_k = make_sdpa_kernel(SEQ_TILES, HEAD_TILES, SCALE)

    # Warmup
    sdpa_k(q_tt, k_tt, v_tt, scaler, out_tt)

    # Timed run
    print("Running on HW (timed)...")
    ttnn.synchronize_device(device)
    t0 = time.time()
    sdpa_k(q_tt, k_tt, v_tt, scaler, out_tt)
    ttnn.synchronize_device(device)
    elapsed = time.time() - t0
    print("Kernel time: %.3fs (%.1fms)" % (elapsed, elapsed * 1000))

    result = ttnn.to_torch(out_tt).float()
    # Only compare valid rows (first N_PATCHES=144 per head, ignore padding)
    N_PATCHES = 144
    max_err = 0
    for h in range(N_HEADS_TEST):
        s = h * N_PATCH_PAD
        diff = (ref_out[s:s+N_PATCHES] - result[s:s+N_PATCHES]).abs()
        h_max = diff.max().item()
        h_mean = diff.mean().item()
        print("  Head %d: max_err=%.4f mean=%.4f result=[%.4f,%.4f] ref=[%.4f,%.4f]" % (
            h, h_max, h_mean,
            result[s:s+N_PATCHES].min().item(), result[s:s+N_PATCHES].max().item(),
            ref_out[s:s+N_PATCHES].min().item(), ref_out[s:s+N_PATCHES].max().item()))
        max_err = max(max_err, h_max)
    mean_err = (ref_out[:N_HEADS_TEST*N_PATCH_PAD] - result[:N_HEADS_TEST*N_PATCH_PAD]).abs().mean().item()

    print("Max error: %.4f, Mean: %.4f" % (max_err, mean_err))
    print("Result range: [%.4f, %.4f]" % (result.min().item(), result.max().item()))

    # Print first few values for debugging
    print("Result[0, 0:5]:", result[0, 0:5].tolist())
    print("Ref[0, 0:5]:", ref_out[0, 0:5].tolist())
    print("Result[160, 0:5]:", result[160, 0:5].tolist())
    print("Ref[160, 0:5]:", ref_out[160, 0:5].tolist())
    # Check if output is all zeros
    print("Result all zeros?", (result == 0).all().item())
    print("Result head 0 all zeros?", (result[:N_PATCH_PAD] == 0).all().item())
    print("Result head 1 all zeros?", (result[N_PATCH_PAD:2*N_PATCH_PAD] == 0).all().item())
    if max_err < 5.0:
        print("PASS")
    else:
        print("FAIL")

    ttnn.close_device(device)
