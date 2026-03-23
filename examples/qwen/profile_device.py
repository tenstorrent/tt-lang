#!/usr/bin/env python3
"""Profile individual kernel device times using PERF_DUMP."""
import os, sys, math, time
import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from kernels.linear import linear_kernel, linear_bias_kernel
from kernels.elementwise import add_kernel, silu_mul_kernel
from kernels.rope import batch_rope_kernel
from kernels.rmsnorm import fused_rmsnorm_kernel
from kernels.group_attn import head_attn_kernels
from kernels.kv_cache_update import get_kv_cache_update_kernel

TILE = 32

def td(t, d):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)

device = ttnn.open_device(device_id=0)

H, I, KV, HD = 896, 4864, 128, 64
MAX_SEQ = 512

# Allocate representative tensors
x = td(torch.randn(TILE, H, dtype=torch.bfloat16) * 0.1, device)
w_q = td(torch.randn(H, H, dtype=torch.bfloat16) * 0.01, device)
b_q = td(torch.randn(TILE, H, dtype=torch.bfloat16) * 0.01, device)
w_gate = td(torch.randn(H, I, dtype=torch.bfloat16) * 0.01, device)
w_down = td(torch.randn(I, H, dtype=torch.bfloat16) * 0.01, device)
w_lm = td(torch.randn(H, 151936, dtype=torch.bfloat16) * 0.01, device)
gamma = td(torch.ones(TILE, H, dtype=torch.bfloat16), device)
mean_sc = td(torch.full((TILE, TILE), 1.0/H, dtype=torch.bfloat16), device)
ones_sc = td(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
attn_sc = td(torch.full((TILE, TILE), 1.0/math.sqrt(HD), dtype=torch.bfloat16), device)
cos_d = td(torch.ones(TILE, HD, dtype=torch.bfloat16), device)
sin_d = td(torch.zeros(TILE, HD, dtype=torch.bfloat16), device)
kt = td(torch.randn(HD, MAX_SEQ, dtype=torch.bfloat16) * 0.01, device)
v = td(torch.randn(MAX_SEQ, HD, dtype=torch.bfloat16) * 0.01, device)
mask = td(torch.full((TILE, MAX_SEQ), float("-inf"), dtype=torch.bfloat16), device)

# Warmup all kernels
y = td(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
linear_bias_kernel(x, w_q, b_q, y)
y2 = td(torch.zeros(TILE, I, dtype=torch.bfloat16), device)
linear_kernel(x, w_gate, y2)
y3 = td(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
fused_rmsnorm_kernel(x, mean_sc, gamma, y3)
q_rot = td(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
batch_rope_kernel(x, cos_d, sin_d, q_rot)
attn_out = td(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
head_attn_kernels[0](q_rot, kt, v, mask, ones_sc, attn_sc, attn_out)
y_add = td(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
add_kernel(x, y, y_add)
lm_out = td(torch.zeros(TILE, 151936, dtype=torch.bfloat16), device)
linear_kernel(x, w_lm, lm_out)
ttnn.synchronize_device(device)

# Time each kernel type (warmup + N iterations, with sync)
def time_kernel(name, fn, n=10):
    # Warmup (ensure compiled)
    fn()
    ttnn.synchronize_device(device)
    # Measure
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ttnn.synchronize_device(device)
    elapsed = (time.perf_counter() - t0) / n
    return elapsed

print(f"{'Kernel':<35} {'Time':>8} {'×/layer':>7} {'×24 layers':>10}")
print("=" * 65)

t = time_kernel("fused_rmsnorm", lambda: fused_rmsnorm_kernel(x, mean_sc, gamma, y3))
print(f"{'fused_rmsnorm':<35} {t*1000:>7.2f}ms {'×2':>7} {t*2*24*1000:>9.1f}ms")

t = time_kernel("linear_bias (Q proj 896→896)", lambda: linear_bias_kernel(x, w_q, b_q, y))
print(f"{'linear_bias (Q proj 896→896)':<35} {t*1000:>7.2f}ms {'×3':>7} {t*3*24*1000:>9.1f}ms")

t = time_kernel("batch_rope (Q 896)", lambda: batch_rope_kernel(x, cos_d, sin_d, q_rot))
print(f"{'batch_rope (Q 896)':<35} {t*1000:>7.2f}ms {'×2':>7} {t*2*24*1000:>9.1f}ms")

t = time_kernel("head_attn (1 head)", lambda: head_attn_kernels[0](q_rot, kt, v, mask, ones_sc, attn_sc, attn_out))
print(f"{'head_attn (1 head)':<35} {t*1000:>7.2f}ms {'×14':>7} {t*14*24*1000:>9.1f}ms")

t = time_kernel("linear (O proj 896→896)", lambda: linear_kernel(attn_out, w_q, y))
print(f"{'linear (O proj 896→896)':<35} {t*1000:>7.2f}ms {'×1':>7} {t*1*24*1000:>9.1f}ms")

t = time_kernel("add_kernel", lambda: add_kernel(x, y, y_add))
print(f"{'add_kernel':<35} {t*1000:>7.2f}ms {'×2':>7} {t*2*24*1000:>9.1f}ms")

t = time_kernel("linear (gate 896→4864)", lambda: linear_kernel(x, w_gate, y2))
print(f"{'linear (gate/up 896→4864)':<35} {t*1000:>7.2f}ms {'×2':>7} {t*2*24*1000:>9.1f}ms")

y_silu = td(torch.zeros(TILE, I, dtype=torch.bfloat16), device)
t = time_kernel("silu_mul", lambda: silu_mul_kernel(y2, y2, y_silu))
print(f"{'silu_mul':<35} {t*1000:>7.2f}ms {'×1':>7} {t*1*24*1000:>9.1f}ms")

y_down = td(torch.zeros(TILE, H, dtype=torch.bfloat16), device)
t = time_kernel("linear (down 4864→896)", lambda: linear_kernel(y_silu, w_down, y_down))
print(f"{'linear (down 4864→896)':<35} {t*1000:>7.2f}ms {'×1':>7} {t*1*24*1000:>9.1f}ms")

print("=" * 65)
t = time_kernel("linear (lm_head 896→151936)", lambda: linear_kernel(x, w_lm, lm_out), n=3)
print(f"{'linear (lm_head 896→151936)':<35} {t*1000:>7.2f}ms {'×1':>7} {t*1000:>9.1f}ms")

# Sum up
print("\n--- Estimated device time per token ---")
attn_t = time_kernel("head_attn", lambda: head_attn_kernels[0](q_rot, kt, v, mask, ones_sc, attn_sc, attn_out))
rmsnorm_t = time_kernel("rmsnorm", lambda: fused_rmsnorm_kernel(x, mean_sc, gamma, y3))
qkv_t = time_kernel("qkv", lambda: linear_bias_kernel(x, w_q, b_q, y))
rope_t = time_kernel("rope", lambda: batch_rope_kernel(x, cos_d, sin_d, q_rot))
oproj_t = time_kernel("oproj", lambda: linear_kernel(attn_out, w_q, y))
add_t = time_kernel("add", lambda: add_kernel(x, y, y_add))
gate_t = time_kernel("gate", lambda: linear_kernel(x, w_gate, y2))
silu_t = time_kernel("silu", lambda: silu_mul_kernel(y2, y2, y_silu))
down_t = time_kernel("down", lambda: linear_kernel(y_silu, w_down, y_down))
lm_t = time_kernel("lm", lambda: linear_kernel(x, w_lm, lm_out), n=3)

per_layer = (rmsnorm_t*2 + qkv_t*3 + rope_t*2 + attn_t*14 + oproj_t + add_t*2 + gate_t*2 + silu_t + down_t)
total = per_layer * 24 + rmsnorm_t + lm_t  # 24 layers + final norm + lm_head
print(f"Per layer: {per_layer*1000:.1f}ms × 24 = {per_layer*24*1000:.1f}ms")
print(f"Final norm + lm_head: {(rmsnorm_t + lm_t)*1000:.1f}ms")
print(f"Total device: {total*1000:.1f}ms → {1/total:.1f} tok/s (device-only theoretical)")

ttnn.close_device(device)
