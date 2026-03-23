# Device-Side Argmax for tt-lang

## Motivation

The logits readback (`ttnn.to_torch` of [32, 151936] = 9.7MB) takes **19ms per token** — 39% of decode time at 20 tok/s. We only need the argmax index (1 integer), not all 151,936 values. A device-side argmax would eliminate this readback entirely.

## Current Bottleneck

```
Per-token breakdown at 20 tok/s (49ms/token):
  trace_replay    27.0ms  55.1%  ← device compute (irreducible)
  d2h_logits      18.9ms  38.6%  ← pulling 9.7MB just for argmax
  h2d_copy         2.6ms   5.3%
  other            0.5ms   1.0%
```

Eliminating the 19ms readback would yield ~30ms/token → **~33 tok/s**.

## Design

### Two-Phase Argmax

The logits tensor is [TILE, vocab_size] = [32, 151936] = [1, 4748] tiles. Only row 0 matters (decode batch=1). We need: `argmax(logits[0, :])`.

**Phase 1: Per-tile local argmax** — multi-core, 110 cores
- Each core handles ~43 tiles
- For each tile: find max value and its column index within the tile
- Write (max_val, local_idx) to a partials tensor

**Phase 2: Global reduce** — single core
- Read all ~4748 (max_val, local_idx) pairs
- Find global max and its index
- Write result to a small [1, 1] output tensor (just the index)

### Why Not a Single Kernel?

4748 tiles is too many for a single-core reduce. Multi-core phase 1 parallelizes the scan, then phase 2 does a small reduce.

### Alternative: Approximate Top-K

Since we only need the argmax of row 0, and each [32, 32] tile has the value at position [0, j] for j=0..31:
- reduce_max(tile, dims=[1]) gives max of row 0 across 32 columns → 1 value
- But we also need the INDEX, not just the value
- tt-lang doesn't have argmax/argmin as a tile op

### Proposed Implementation

#### Option A: Compiler-level `ttl.math.argmax`

Add a new tile op that returns both the max value and its index. This requires:
1. TTL dialect op definition
2. Lowering to TTKernel
3. Hardware support: the SFPU can compute argmax with `topk` LLK

**Hardware support exists**: tt-metal has `ttnn/cpp/ttnn/operations/reduction/topk/` which implements top-K on device. The LLK `topk_local.cpp` and `topk_final.cpp` show the compute pattern.

#### Option B: Pure tt-lang kernel (no compiler changes)

Implement argmax using existing ops:
1. For each tile in row: `reduce_max(tile, dims=[1])` → max of row 0's 32 values
2. Compare max against a running global max
3. Track which tile had the global max
4. Within the winning tile, find which column by comparing each value

This is complex in tile ops but avoids compiler changes.

#### Option C: Use ttnn.argmax directly (bypass tt-lang)

ttnn has `ttnn.argmax` as a built-in op. Call it directly from Python on the device tensor, bypassing tt-lang kernels entirely. This is the **simplest approach**.

### Recommended: Option C (ttnn.argmax)

```python
# Current (19ms):
logits_host = ttnn.to_torch(self._tb["logits"]).float()  # 9.7MB D2H
token = logits_host[0].argmax().item()

# Proposed (~0.1ms):
argmax_result = ttnn.argmax(self._tb["logits"], dim=-1)  # device-side
token = ttnn.to_torch(argmax_result)[0, 0].item()        # 4 bytes D2H
```

If `ttnn.argmax` works on our tensor shape, this is a one-line fix.

### Fallback: Option B (tt-lang kernel)

If ttnn.argmax doesn't work (shape restrictions, dtype issues), write a custom kernel:

```python
@ttl.kernel(grid=(11, 10))  # 110 cores
def argmax_kernel(logits, result):
    """Find argmax of row 0 across all columns.

    logits: [TILE, vocab_size] — only row 0 matters
    result: [TILE, TILE] — result[0, 0] = argmax index

    Phase 1 (per core): scan ~43 tiles, find local max + index
    Phase 2 (core 0): reduce across all cores' results
    """
```

This is more work but gives full control.

## Integration

### In model.py `decode_step_traced`:

```python
# Replace:
logits_host = ttnn.to_torch(self._tb["logits"]).float()
token = logits_host[0].argmax().item()

# With:
argmax_dev = ttnn.argmax(self._tb["logits"], dim=-1)
token = ttnn.to_torch(argmax_dev)[0, 0].item()
```

### Trace compatibility

`ttnn.argmax` is a ttnn op, not a tt-lang kernel. It should work inside or outside the trace. If inside the trace, the result tensor is pre-allocated and the argmax is pipelined with the rest of the compute.

If outside the trace (simpler): the argmax runs after trace replay, operating on the pre-allocated logits tensor. The D2H is only 4 bytes (one int32).

## Testing

1. Verify `ttnn.argmax` works on [32, 151936] tensor:
```python
t = ttnn.from_torch(torch.randn(32, 151936, dtype=torch.bfloat16), ...)
result = ttnn.argmax(t, dim=-1)
print(ttnn.to_torch(result))  # should be [32, 1] with indices
```

2. Compare against torch argmax:
```python
ref = torch_tensor[0].argmax().item()
dev = ttnn.to_torch(ttnn.argmax(dev_tensor, dim=-1))[0, 0].item()
assert ref == dev
```

3. Measure timing:
```python
t0 = time.perf_counter()
ttnn.argmax(dev_tensor, dim=-1)
ttnn.synchronize_device(device)
print(f"argmax: {(time.perf_counter()-t0)*1e3:.2f}ms")
```

## Expected Impact

- Eliminates 19ms/token logits readback (39% of token time)
- Adds ~0.1-0.5ms for device argmax + 4-byte readback
- Net saving: ~18.5ms/token
- New throughput: 49ms - 18.5ms = ~30.5ms → **~33 tok/s**

## Files to Modify

- `examples/qwen/model.py` — `decode_step_traced()` logits handling
- Potentially: `examples/qwen/kernels/argmax.py` if custom kernel needed

## Step-by-step for Implementation Session

1. Test if `ttnn.argmax` works on [32, 151936] bf16 tensor
2. If yes: one-line change in model.py, measure speedup
3. If no: write custom argmax kernel using reduce_max + comparison
4. Integrate into traced decode path
5. Verify correctness (compare token sequences traced vs non-traced)
6. Measure final tok/s
