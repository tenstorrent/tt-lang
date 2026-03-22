"""Debug: compare fused_attn_head_kernel vs host softmax on real model data."""
import math, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))

import ttnn
from model import QwenModel
from kernels.linear import linear_kernel
from kernels.fused_attn import fused_attn_head_kernel

TILE = 32

device = ttnn.open_device(device_id=0)
model = QwenModel(device)
model.quiet = True

# Prefill to populate KV cache
input_ids = [785, 6722, 315, 9625, 374]  # "The capital of France is"
with model._suppress_output():
    logits = model.prefill(input_ids)
print(f"Prefill done. First token: {logits[-1].argmax().item()}")

# Prepare for first decode step
pos = model.cache_pos
token_id = logits[-1].argmax().item()
print(f"Decode pos={pos}, token={token_id}")

# Prepare cos/sin for decode
cos_pos = torch.ones(TILE, model.head_dim, dtype=torch.bfloat16)
sin_pos = torch.zeros(TILE, model.head_dim, dtype=torch.bfloat16)
cos_pos[0] = model.rope_cos[pos].bfloat16()
sin_pos[0] = model.rope_sin[pos].bfloat16()

with model._suppress_output():
    # Run one decode layer to get Q, K^T, V
    from kernels.rope import batch_rope_kernel
    from kernels.rmsnorm import fused_device_rmsnorm

    x = model.embed_weight[token_id:token_id+1]
    x_padded = torch.zeros(TILE, model.hidden_size, dtype=torch.bfloat16)
    x_padded[0] = x[0].bfloat16()
    x_dev = model._to_device(x_padded)
    cos_dev = model._to_device(cos_pos)
    sin_dev = model._to_device(sin_pos)

    w = model.layer_weights[0]
    normed = fused_device_rmsnorm(x_dev, w["input_layernorm_weight"],
                                   model.mean_scaler_device, model.device)

    # Combined Q projection
    q_out = model._alloc_zeros((TILE, model.hidden_size))
    from kernels.linear import linear_bias_kernel
    linear_bias_kernel(normed, w["q_proj_weight"], w["q_proj_bias"], q_out)

    # Batch RoPE
    q_rot = model._alloc_zeros((TILE, model.hidden_size))
    batch_rope_kernel(q_out, cos_dev, sin_dev, q_rot)

    # Get Q for head 0
    q_rot_host = ttnn.to_torch(q_rot)
    scale_val = 1.0 / math.sqrt(model.head_dim)
    q_head = q_rot_host[:, :model.head_dim].clone()
    q_head[0] = q_head[0] * scale_val
    q_head_dev = model._to_device(q_head)

    # Get KV cache for head 0
    k_t_dev = model.kv_cache_dev[0][0]["k_t"]
    v_dev = model.kv_cache_dev[0][0]["v"]

    # Mask
    attend_len = pos + 1
    mask_t = torch.full((TILE, 512), float("-inf"), dtype=torch.bfloat16)
    mask_t[0, :attend_len] = 0.0
    mask_dev = model._to_device(mask_t)

    # === Host softmax path ===
    scores_dev = model._alloc_zeros((TILE, 512))
    linear_kernel(q_head_dev, k_t_dev, scores_dev)
    scores_host = ttnn.to_torch(scores_dev).float()
    weights_host = torch.nn.functional.softmax(
        scores_host + mask_t.float(), dim=-1
    ).bfloat16()
    weights_dev = model._to_device(weights_host)
    host_out_dev = model._alloc_zeros((TILE, model.head_dim))
    linear_kernel(weights_dev, v_dev, host_out_dev)
    host_out = ttnn.to_torch(host_out_dev)

    # === Fused kernel path ===
    scratch = model._alloc_zeros((TILE, 512))
    fused_out_dev = model._alloc_zeros((TILE, model.head_dim))
    fused_attn_head_kernel(
        q_head_dev, k_t_dev, v_dev,
        mask_dev, model.ones_scaler_device,
        scratch, fused_out_dev,
    )
    fused_out = ttnn.to_torch(fused_out_dev)

print(f"\nHost  out[0,:8]: {host_out[0,:8].tolist()}")
print(f"Fused out[0,:8]: {fused_out[0,:8].tolist()}")
pcc = torch.corrcoef(torch.stack([host_out[0].float(), fused_out[0].float()]))[0, 1].item()
print(f"PCC: {pcc:.6f}")

# Check intermediate: what do scores look like?
print(f"\nScores[0,:8]: {scores_host[0,:8].tolist()}")
print(f"Scores range: [{scores_host[0,:attend_len].min():.4f}, {scores_host[0,:attend_len].max():.4f}]")
print(f"Host weights sum: {weights_host[0].float().sum():.4f}")

# Check fused kernel's scratch (should have softmax weights after kernel)
scratch_out = ttnn.to_torch(scratch)
print(f"Scratch[0,:8]: {scratch_out[0,:8].tolist()}")
print(f"Scratch sum: {scratch_out[0].float().sum():.4f}")

ttnn.close_device(device)
