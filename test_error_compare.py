"""Compare ttl SDPA error against ttnn manual SDPA and ttnn transformer SDPA."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
N_PATCH_PAD = 160
D_HEAD = 64
SCALE = 0.125


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


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

    # bf16 torch reference
    qb = q.to(torch.bfloat16).float()
    kb = k.to(torch.bfloat16).float()
    vb = v.to(torch.bfloat16).float()
    bf16_scores = (qb @ kb.T) * SCALE
    bf16_attn = F.softmax(bf16_scores, dim=-1)
    bf16_out = bf16_attn @ vb
    bf16_err = (ref_out - bf16_out).abs()
    print("=== torch bf16 SDPA ===")
    print("  Max err: %.6f, Mean: %.6f" % (bf16_err.max().item(), bf16_err.mean().item()))

    # === ttnn matmul: Q @ K^T (pre-transpose on host) ===
    kt_torch = k.to(torch.bfloat16).T.contiguous()
    q_tt = to_tt(q.to(torch.bfloat16), device)
    kt_tt = to_tt(kt_torch, device)
    qk_tt = ttnn.matmul(q_tt, kt_tt)
    qk_result = ttnn.to_torch(qk_tt).float()
    ref_qk = q @ k.T
    qk_err = (ref_qk - qk_result).abs()
    print("=== ttnn matmul Q@K^T ===")
    print("  Max err: %.6f, Mean: %.6f" % (qk_err.max().item(), qk_err.mean().item()))

    # === ttnn manual SDPA chain ===
    try:
        scaled_tt = ttnn.multiply(qk_tt, SCALE)
        softmax_tt = ttnn.softmax(scaled_tt, dim=-1)
        v_tt = to_tt(v.to(torch.bfloat16), device)
        out_tt = ttnn.matmul(softmax_tt, v_tt)
        manual_result = ttnn.to_torch(out_tt).float()
        manual_err = (ref_out - manual_result).abs()
        print("=== ttnn manual SDPA (matmul+scale+softmax+matmul) ===")
        print("  Max err: %.6f, Mean: %.6f" % (manual_err.max().item(), manual_err.mean().item()))
        print("  Result range: [%.4f, %.4f]" % (manual_result.min().item(), manual_result.max().item()))
    except Exception as e:
        print("=== ttnn manual SDPA failed: %s ===" % str(e)[:200])

    # === ttnn transformer SDPA ===
    try:
        q_4d = q.to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        k_4d = k.to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        v_4d = v.to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        q_4d_tt = to_tt(q_4d, device)
        k_4d_tt = to_tt(k_4d, device)
        v_4d_tt = to_tt(v_4d, device)
        sdpa_out = ttnn.transformer.scaled_dot_product_attention(
            q_4d_tt, k_4d_tt, v_4d_tt, is_causal=False)
        sdpa_result = ttnn.to_torch(sdpa_out).float().squeeze(0).squeeze(0)
        sdpa_err = (ref_out - sdpa_result).abs()
        print("=== ttnn transformer SDPA ===")
        print("  Max err: %.6f, Mean: %.6f" % (sdpa_err.max().item(), sdpa_err.mean().item()))
        print("  Result range: [%.4f, %.4f]" % (sdpa_result.min().item(), sdpa_result.max().item()))
    except Exception as e:
        print("=== ttnn transformer SDPA failed: %s ===" % str(e)[:200])

    print("\n=== For reference ===")
    print("  ttl SDPA Head 0 (from prior run): max_err~0.08, mean~0.02")
    print("  ttl SDPA Head 1 (from prior run): max_err~0.30, mean~0.12")

    ttnn.close_device(device)
