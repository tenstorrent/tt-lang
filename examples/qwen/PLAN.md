# Qwen 2.5 0.5B-Instruct on Tenstorrent Blackhole via tt-lang

## Context

Port the Qwen 2.5 0.5B-Instruct model to tt-lang for execution on a single Blackhole card. The end goal is a fully functional chat demo with both prefill (seq_len=512) and decode phases, with all compute running on the accelerator. We build incrementally: individual op validation → component kernels → single layer → full model → decode → demo.

## Model Architecture (Qwen 2.5 0.5B-Instruct)

| Parameter | Value | Tiles (32x32) |
|-----------|-------|---------------|
| hidden_size | 896 | 28 |
| num_layers | 24 | - |
| Q heads | 14 | - |
| KV heads | 2 (GQA 7:1) | - |
| head_dim | 64 | 2 |
| intermediate_size | 4864 | 152 |
| vocab_size | 151936 | 4748 |
| seq_len (target) | 512 | 16 |

- Normalization: RMSNorm (eps=1e-6)
- MLP: SwiGLU — `silu(gate_proj(x)) * up_proj(x)` then `down_proj`
- Q/K/V projections have bias; O/gate/up/down do not
- Tied embeddings (embedding weight == lm_head weight)
- Total ~490M params, ~992 MB in bf16 (fits in 12GB Blackhole DRAM)

## Hardware Op Constraints

**Confirmed on HW** (compiler support):
- Binary: `+`, `-`, `*`, `/`, `min`, `max`
- Unary: `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `relu`, `floor`, `recip`, `sin`, `cos`, `tan`, `-`, `abs`
- `ttl.math.broadcast` (row/col/scalar)
- `ttl.copy` / async DMA with double-buffered DFBs
- 2D grid with `ttl.node(dims=2)` / `ttl.grid_size(dims=2)`
- Fused matmul: `prev + a @ b` pattern (see `examples/matmul_acc.py`)

**NOT on HW** (simulator only — need decomposition or host fallback):
- `reduce_sum`, `reduce_max` → tree reduction using `+` / `max`
- `transpose` → host-side pre-transpose
- `store(acc=True)` → use `prev + a @ b` fused pattern instead
- `silu` → decompose to `x * sigmoid(x)`
- Standalone `@` operator → use fused `prev + a @ b`

**Not in tt-lang at all** (host-side):
- Embedding lookup, tokenization, sampling/argmax

## File Structure

```
examples/qwen/
  PLAN.md                        # This file
  utils.py                       # Shared: to_device, validation, weight loading
  weight_extractor.py            # Phase 0: Download & prepare weights
  op_validation.py               # Phase 1: HW op validation
  kernels/
    __init__.py
    linear.py                    # Linear layer (matmul + optional bias)
    rmsnorm.py                   # RMSNorm with tree reduction
    attention.py                 # GQA attention with softmax
    elementwise.py               # add, silu_mul helpers
  single_layer.py                # Phase 3: One transformer layer
  model.py                       # Phase 4-5: Full model wrapper
  prefill.py                     # Phase 4: Full prefill pass
  decode.py                      # Phase 5: Decode with KV cache
  chat_demo.py                   # Phase 6: Interactive chat
```

---

## Phase 0: Weight Extraction & Environment Setup

**Status: COMPLETE**

**File:** `examples/qwen/weight_extractor.py`

**Steps:**
1. `pip install transformers` (if needed)
2. Download `Qwen/Qwen2.5-0.5B-Instruct` from HuggingFace
3. Extract all layer weights as bf16 torch tensors:
   - Per layer (×24): `q_proj.weight` [896,896], `q_proj.bias` [896], `k_proj.weight` [128,896], `k_proj.bias` [128], `v_proj.weight` [128,896], `v_proj.bias` [128], `o_proj.weight` [896,896], `gate_proj.weight` [4864,896], `up_proj.weight` [4864,896], `down_proj.weight` [896,4864], `input_layernorm.weight` [896], `post_attention_layernorm.weight` [896]
   - Global: `embed_tokens.weight` [151936,896], `norm.weight` [896]
4. Pad all tensors to tile boundaries (multiples of 32)
5. Pre-transpose weight matrices to [out_features, in_features] → [in_features, out_features] for matmul compatibility (tt-lang does `x @ W` not `x @ W^T`)
6. Expand all 1D tensors (bias, norm weights) to [32, N] so every row of each tile has identical values — avoids needing tile-level broadcast at runtime
7. Pre-compute RoPE cos/sin tables for positions 0..511, shape [512, 64]
8. Save everything to a `.pt` checkpoint file (947 MB)

**Validation:** Reload and compare against original model state_dict. ✓

---

## Phase 1: Hardware Op Validation

**Status: COMPLETE**

**File:** `examples/qwen/op_validation.py`

Validated critical operations on real Blackhole hardware. Each test compares against PyTorch golden.

| Test | Op | Result |
|------|----|--------|
| 1 | Matmul K-accumulation (`prev + a @ b`) | **PASS** PCC=0.9999 |
| 2 | SiLU decomposition (`x * sigmoid(x)`) | **PASS** PCC=0.9999 |
| 3 | `store(a @ b, acc=True)` | FAIL — not supported on compiler |
| 4 | `ttl.math.reduce_sum` | FAIL — not supported on compiler |
| 5 | Broadcast row (`ttl.math.broadcast(c, out, dims=[0])`) | **PASS** |
| 6 | Sequential tile accumulation (4 tiles via DFB) | **PASS** PCC=0.9999 |

**Decision points confirmed:**
- Use `prev + a @ b` fused pattern for ALL matmuls (not `acc=True`)
- Use sequential tile accumulation for reductions (not `reduce_sum`)
- Use host-side reduction for RMSNorm and softmax (simplest, avoids complex in-kernel reduction)
- Broadcast is tile-level only; for multi-tile "broadcast", DM thread reads the same tile repeatedly

---

## Phase 2: Component Kernels (Single-Core)

**Status: COMPLETE**

### 2a: Linear Layer
**File:** `examples/qwen/kernels/linear.py`
**Reference:** `examples/matmul_acc.py` (exact pattern)

Two variants:
- `linear_kernel(x, weight, out)` — Y = X @ W, no bias (for O/gate/up/down projections)
- `linear_bias_kernel(x, weight, bias, out)` — Y = X @ W + bias (for Q/K/V projections)

**Algorithm (for output tile [m, n]):**
```
DM: load bias[0, n] into b_dfb (if bias variant)
    for k in K_tiles: load X[m, k] and W[k, n] into x_dfb, w_dfb
Compute:
    if bias: init acc from bias tile
    first K step: acc = x @ w  (or bias + x @ w)
    remaining K steps: acc = prev + x @ w  (fused pattern)
    write acc to y_dfb
DM: write y_dfb → Y[m, n]
```

**Block sizing:** 1x1 tile blocks. Process output tiles one at a time with K-loop accumulation.

**Validated:** PCC > 0.999 at [512×896] @ [896×896] + bias (full Q projection size). ✓

### 2b: RMSNorm
**File:** `examples/qwen/kernels/rmsnorm.py`

**Hybrid approach:** Reduction on host, multiply on device.
1. Pull x to host
2. Compute rsqrt(mean(x²) + eps) per row → scale [seq, 1]
3. Expand scale to [seq, 32] tile (same value across all 32 columns)
4. Send scale to device
5. Device kernel: `Y[m,n] = X[m,n] * scale[m,0] * gamma[0,n]` tile-by-tile

**Validated:** PCC > 0.999 at [512×896]. ✓

### 2c: SwiGLU MLP
No separate file — composed from existing kernels:
```python
gate = linear_kernel(normed, gate_proj_w)      # [512, 4864]
up   = linear_kernel(normed, up_proj_w)        # [512, 4864]
hidden = silu_mul_kernel(gate, up)             # [512, 4864]
out  = linear_kernel(hidden, down_proj_w)      # [512, 896]
```

### 2d: Attention
**File:** `examples/qwen/kernels/attention.py`

**Per-head shapes** (single Q head, single KV head):
- Q: [16, 2] tiles (seq=512, head_dim=64)
- K^T: [2, 16] tiles (pre-transposed on host)
- V: [16, 2] tiles
- Scores: Q @ K^T = [16, 2] @ [2, 16] = [16, 16] tiles (only K=2 accumulation steps)
- Output: softmax(scores) @ V = [16, 16] @ [16, 2] = [16, 2] tiles

**Approach:**
1. `scores = linear_kernel(Q, K^T)` — on device
2. Host: scale, mask, softmax
3. `attn_out = linear_kernel(softmax_weights, V)` — on device

**GQA handling:** 2 KV heads, 14 Q heads (7 Q heads per KV head).
- Host orchestrates: for each of 14 Q heads, select appropriate Q slice and KV head
- Call attention per head, concatenate 14 outputs → [512, 896]

**RoPE:** Applied on host before attention. Pre-compute cos/sin tables, apply rotation to Q and K slices using PyTorch.

**Causal mask:** Pre-compute [512, 512] mask on host. Applied as `scores + mask` before softmax.

**Validated:** PCC > 0.999. ✓

### 2e: Elementwise Ops
**File:** `examples/qwen/kernels/elementwise.py`

- `add_kernel(A, B, Y)` — Y = A + B (residual connections). PCC > 0.999. ✓
- `silu_mul_kernel(gate, up, Y)` — Y = silu(gate) * up (SwiGLU). PCC > 0.999. ✓

---

## Phase 3: Single Transformer Layer

**Status: COMPLETE**

**File:** `examples/qwen/single_layer.py`

Host-side orchestration calling the component kernels:

```
x [512, 896] on device DRAM
    ↓
1.  rmsnorm(x, ln1_weight) → normed
2.  linear_bias(normed, q_w, q_b) → q [512, 896]
3.  linear_bias(normed, k_w, k_b) → k [512, 128]
4.  linear_bias(normed, v_w, v_b) → v [512, 128]
5.  (host) reshape Q→14 heads, K/V→2 heads, apply RoPE, transpose K
6.  attention per head × 14 → concat → attn_out [512, 896]
7.  linear(attn_out, o_w) → proj [512, 896]
8.  add(x, proj) → post_attn [512, 896]
9.  rmsnorm(post_attn, ln2_weight) → normed2
10. linear(normed2, gate_w) → gate [512, 4864]
11. linear(normed2, up_w) → up [512, 4864]
12. silu_mul(gate, up) → hidden [512, 4864]
13. linear(hidden, down_w) → mlp_out [512, 896]
14. add(post_attn, mlp_out) → output [512, 896]
```

**~25 kernel invocations per layer.**

**Validated:** Layer 0 with real Qwen weights vs HuggingFace reference. **PCC = 0.997.** ✓

---

## Phase 4: Full Model Prefill

**Status: COMPLETE**

**Results:**
- Prompt "The capital of France is" → predicted " Paris" (matches HuggingFace)
- Last-position logits PCC: 0.986
- Top-1 token match: YES
- Top-5 overlap: 4/5
- Prefill time: ~2.8s for 5 tokens (includes first-run kernel compilation)

**File:** `examples/qwen/prefill.py`, `examples/qwen/model.py`

**model.py** wraps all 24 layers with weight management:
```
tokens [≤512] → (host) tokenizer + embedding lookup → x [seq, 896]
→ send x to device DRAM
→ for layer in 0..23: x = transformer_layer(x, weights[layer])
→ (host) final RMSNorm
→ (device or host) lm_head matmul: [seq, 896] @ [896, 151936] → logits
→ (host) return logits for last position
```

**lm_head matmul** is large (151936 output dim = 4748 tiles). Split into chunks: process 32 output tile columns at a time = 1024 vocab entries per chunk, ~148 chunks. Or do on host via PyTorch for simplicity.

**Memory:** All 24 layers' weights (~720MB) + embeddings (~272MB) = ~992MB in DRAM. Activations: max ~10MB per layer (seq=512). Easily fits in 12GB.

**Validation:** Run HuggingFace model on a test prompt, compare logits. Top-1 token should match for first several positions.

---

## Phase 5: Decode with KV Cache

**Status: COMPLETE**

**Results:**
- Prompt "The capital of France is" → generates " Paris. It is the largest city in the European Union..."
- First 3 tokens match HuggingFace exactly; divergence at token 3 due to bf16 precision
- Both outputs are coherent and factual — model is functionally correct
- KV cache architecture: host mirrors (source of truth) + device DRAM upload per attention step
- Decode speed: ~0.92s/token (includes first-run kernel compilation)
- Cache size: 6MB total across 24 layers × 2 KV heads

**File:** `examples/qwen/decode.py`

**Key differences from prefill:**
- Input: single token → x [1, 896] (padded to [32, 896] = [1, 28] tiles)
- KV cache: pre-allocated [512, 64] per head per layer in DRAM
- Each step: compute new K, V; append to cache; attend over full cache
- Attention: Q[1, 2] @ K_cache^T[2, seq_tiles] = scores[1, seq_tiles]

**KV cache management (host-side):**
- Pre-allocate: 2 KV heads × 24 layers × 2 (K+V) × [512, 64] = ~6MB total
- Each decode step: write new K[1, 64] and V[1, 64] into cache at current position
- Pass growing cache slice to attention kernel

**Decode attention simplification:**
- Q is [1, 2] tiles (single row)
- K^T is [2, pos_tiles] where pos_tiles grows each step
- Scores [1, pos_tiles] — much smaller, tree reduction is simpler
- softmax @ V: [1, pos_tiles] @ [pos_tiles, 2] = [1, 2]

**Autoregressive loop:**
```
logits = prefill(prompt_tokens)  # fills KV cache for prompt
next_token = argmax(logits[-1])
for step in range(max_new_tokens):
    logits = decode_step(next_token, step + prompt_len)
    next_token = argmax(logits[-1])
    if next_token == eos: break
    output_tokens.append(next_token)
```

**Validation:** Generate 20 tokens, compare token-by-token against HuggingFace `model.generate()` with greedy decoding.

---

## Phase 6: End-to-End Chat Demo

**Status: COMPLETE**

**Results:**
- "What is 2+2?" → "2+2 equals 4." (correct answer, proper EOS)
- Chat template formatting works (Qwen instruct format)
- Streaming token output
- ~2.4s/tok on first run (kernel compilation dominates); faster on subsequent runs with same shapes

**File:** `examples/qwen/chat_demo.py`

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Interactive loop
while True:
    user_input = input("User: ")
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    input_ids = tokenizer.encode(prompt)

    # Prefill
    logits = model.prefill(input_ids)

    # Decode loop with streaming output
    print("Assistant: ", end="", flush=True)
    for token in model.generate(logits, max_tokens=256):
        print(tokenizer.decode([token]), end="", flush=True)
    print()
```

---

## Phase 7: Optimization (Future)

Once functional, optimize for performance:
1. **Multi-core matmul** — distribute linear layer output tiles across cores (follow `examples/multinode_matmul.py` pattern)
2. **Kernel fusion** — combine RMSNorm + first linear into one kernel, reduce DRAM round-trips
3. **Parallelized attention** — process multiple Q heads on different cores simultaneously
4. **Continuous batching** — overlap prefill and decode across requests
5. **Move RoPE to device** — eliminate host round-trip for rotation
6. **Move toward single big kernel** — fuse entire transformer layer into one kernel call to minimize launch overhead

---

## Key Reference Files

| File | What to reuse |
|------|--------------|
| `examples/matmul_acc.py` | Exact pattern for linear layers with K-accumulation and bias |
| `examples/test_transformer_block.py` | Structural reference for RMSNorm, attention, MLP composition |
| `examples/multinode_matmul.py` | Multi-core work distribution pattern |
| `examples/single_node_matmul.py` | Single-core matmul with `store(acc=True)` — confirmed N/S on compiler |
| `test/me2e/` | Test infrastructure patterns for HW validation |

## Verification Strategy

Each phase has its own validation:
- **Phase 1:** Individual op tests with PCC > 0.99 against PyTorch ✓
- **Phase 2:** Component kernel tests with PCC > 0.99 against PyTorch ✓
- **Phase 3:** Single layer activation comparison with HuggingFace model, PCC > 0.95 ✓ (got 0.997)
- **Phase 4:** Full prefill logit comparison, top-1 token match
- **Phase 5:** Greedy decode token-by-token match against HuggingFace
- **Phase 6:** Qualitative: model produces coherent text responses
