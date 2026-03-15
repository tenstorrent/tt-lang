"""Compare ttl SDPA vs ttnn.transformer.scaled_dot_product_attention."""
import torch
import torch.nn.functional as F
import ttnn
import ttl

TILE = 32
N_PATCH_PAD = 160
D_HEAD = 64
SEQ_TILES = N_PATCH_PAD // TILE  # 5
HEAD_TILES = D_HEAD // TILE  # 2
SCALE = 0.125


@ttl.kernel(grid=(1, 1))
def sdpa_single_head(Q, K, V, scaler, out):
    q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    k_dfb = ttl.make_dataflow_buffer_like(K, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    v_dfb = ttl.make_dataflow_buffer_like(V, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)

    kt_dfb = ttl.make_dataflow_buffer_like(K, shape=(HEAD_TILES, SEQ_TILES), buffer_factor=2)
    a_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    row_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(SEQ_TILES, 1), buffer_factor=2)
    row_bc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        with q_dfb.wait() as qv, k_dfb.wait() as kv, v_dfb.wait() as vv, sc_dfb.wait() as sc:
            with kt_dfb.reserve() as kt:
                kt.store(ttl.transpose(kv))
            with kt_dfb.wait() as ktv, a_dfb.reserve() as qk:
                qk.store(qv @ ktv)
            with a_dfb.wait() as qkv, b_dfb.reserve() as scaled:
                scaled.store(qkv * ttl.math.fill(qkv, 0.125))
            with b_dfb.wait() as sdv:
                with row_dfb.reserve() as mx:
                    mx.store(ttl.math.reduce_max(sdv, sc, dims=[1]))
                with row_dfb.wait() as mxv, row_bc_dfb.reserve() as mxb:
                    mxb.store(ttl.math.broadcast(mxv, dims=[1]))
                with row_bc_dfb.wait() as mxbv:
                    with a_dfb.reserve() as ex:
                        ex.store(ttl.math.exp(sdv - mxbv))
                with a_dfb.wait() as exv:
                    with row_dfb.reserve() as sm:
                        sm.store(ttl.math.reduce_sum(exv, sc, dims=[1]))
                    with row_dfb.wait() as smv, row_bc_dfb.reserve() as smb:
                        smb.store(ttl.math.broadcast(smv, dims=[1]))
                    with row_bc_dfb.wait() as smbv, c_dfb.reserve() as attn:
                        attn.store(exv * ttl.math.recip(smbv))
            with c_dfb.wait() as av, out_dfb.reserve() as o:
                o.store(av @ vv)

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(Q[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(K[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(V[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HEAD_TILES]); tx.wait()


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    q = torch.randn(N_PATCH_PAD, D_HEAD, dtype=torch.float32) * 0.3
    k = torch.randn(N_PATCH_PAD, D_HEAD, dtype=torch.float32) * 0.3
    v = torch.randn(N_PATCH_PAD, D_HEAD, dtype=torch.float32) * 0.3

    # f32 reference
    ref_scores = (q @ k.T) * SCALE
    ref_attn = F.softmax(ref_scores, dim=-1)
    ref_out = ref_attn @ v
    print("Ref range: [%.4f, %.4f]" % (ref_out.min().item(), ref_out.max().item()))

    # bf16 torch reference (best possible bf16 accuracy)
    qb = q.to(torch.bfloat16).float()
    kb = k.to(torch.bfloat16).float()
    vb = v.to(torch.bfloat16).float()
    bf16_out = F.softmax((qb @ kb.T) * SCALE, dim=-1) @ vb
    bf16_err = (ref_out - bf16_out).abs()
    print("=== torch bf16 SDPA (best possible) ===")
    print("  Max err: %.6f, Mean: %.6f" % (bf16_err.max().item(), bf16_err.mean().item()))

    # === ttl SDPA ===
    q_tt = to_tt(q.to(torch.bfloat16), device)
    k_tt = to_tt(k.to(torch.bfloat16), device)
    v_tt = to_tt(v.to(torch.bfloat16), device)
    out_tt = to_tt(torch.zeros(N_PATCH_PAD, D_HEAD, dtype=torch.bfloat16), device)
    scaler = to_tt_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)

    sdpa_single_head(q_tt, k_tt, v_tt, scaler, out_tt)
    ttl_result = ttnn.to_torch(out_tt).float()
    ttl_err = (ref_out - ttl_result).abs()
    print("=== ttl SDPA ===")
    print("  Max err: %.6f, Mean: %.6f" % (ttl_err.max().item(), ttl_err.mean().item()))
    print("  Result range: [%.4f, %.4f]" % (ttl_result.min().item(), ttl_result.max().item()))

    # === ttnn transformer SDPA ===
    try:
        q_4d = q.to(torch.bfloat16).unsqueeze(0).unsqueeze(0)  # (1,1,160,64)
        k_4d = k.to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        v_4d = v.to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        q_4d_tt = to_tt(q_4d, device)
        k_4d_tt = to_tt(k_4d, device)
        v_4d_tt = to_tt(v_4d, device)
        sdpa_out = ttnn.transformer.scaled_dot_product_attention(
            q_4d_tt, k_4d_tt, v_4d_tt, is_causal=False)
        ttnn_result = ttnn.to_torch(sdpa_out).float().squeeze(0).squeeze(0)
        ttnn_err = (ref_out - ttnn_result).abs()
        print("=== ttnn SDPA ===")
        print("  Max err: %.6f, Mean: %.6f" % (ttnn_err.max().item(), ttnn_err.mean().item()))
        print("  Result range: [%.4f, %.4f]" % (ttnn_result.min().item(), ttnn_result.max().item()))
    except Exception as e:
        print("=== ttnn SDPA failed: %s ===" % str(e)[:300])

    # === Summary ===
    print("\n=== Summary ===")
    print("  torch bf16:  max=%.6f mean=%.6f" % (bf16_err.max().item(), bf16_err.mean().item()))
    print("  ttl SDPA:    max=%.6f mean=%.6f" % (ttl_err.max().item(), ttl_err.mean().item()))

    ttnn.close_device(device)
