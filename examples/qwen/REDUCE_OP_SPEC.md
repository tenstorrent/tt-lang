# Spec: Add reduce_max and reduce_sum to tt-lang Compiler

## Problem

`ttl.math.reduce_max` and `ttl.math.reduce_sum` are listed as simulator-only (N/S on compiler). They are needed for device-side softmax and RMSNorm in LLM inference. The hardware, TTKernel MLIR dialect, and EmitC backend all support reduce operations — the gap is only in the TTL dialect and its lowering.

## Background: What Already Works

The full stack below TTL is ready:

```
Hardware:     reduce_init / reduce_tile / reduce_uninit  (PoolType::MAX, PoolType::SUM)
                ↑
TTKernel:     ttkernel.reduce_init / ttkernel.reduce_tile / ttkernel.reduce_uninit
              (ReduceType::Max, ReduceType::Sum × ReduceDim::Row, Col, Scalar)
                ↑
EmitC:        TTKernelToEmitC.cpp already lowers these to C++ template calls
                ↑
TTL:          ❌ NO reduce ops defined — this is the gap
                ↑
Python DSL:   ttl.math.reduce_max / reduce_sum exist in simulator, rejected by compiler
```

### Hardware API (target C++ output)

From `third-party/tt-metal/tt_metal/hw/inc/api/compute/reduce.h`:

```cpp
// Initialize reduce state. Called once before processing tiles.
template <PoolType reduce_type, ReduceDim reduce_dim>
void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb);

// Reduce one tile. Accumulates into DST[idst]. Call once per input tile.
template <PoolType reduce_type, ReduceDim reduce_dim>
void reduce_tile(uint32_t icb, uint32_t icb_scaler,
                 uint32_t itile, uint32_t itile_scaler, uint32_t idst);

// Cleanup. Called once after all tiles processed.
void reduce_uninit(uint32_t icb);
```

- `PoolType::MAX` or `PoolType::SUM`
- `ReduceDim::REDUCE_ROW` (reduce across rows → output has 1 row), `REDUCE_COL` (reduce across columns → output has 1 column), `REDUCE_SCALAR` (reduce to single value)
- The scaler CB holds a tile of scaling factors (typically 1.0 for max, 1/N for mean)
- Multiple `reduce_tile` calls accumulate: for sum, DST += reduced_tile; for max, DST = max(DST, reduced_tile)

### TTKernel MLIR ops (already defined)

From `third-party/tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td`:

```
ttkernel.reduce_init %in_cb, %scaler_cb, %out_cb {reduce_type = #ttkernel<reduce_type reduce_max>, reduce_dim = #ttkernel<reduce_dim reduce_dim_col>}
ttkernel.reduce_tile %in_cb, %scaler_cb, %in_tile_idx, %scaler_tile_idx, %dst_idx {reduce_type = ..., reduce_dim = ...}
ttkernel.reduce_uninit
```

### Existing analogous op: broadcast

`TTL_TileBcastOp` is a non-elementwise tile op with extra parameters (dims). It has custom lowering to TTKernel. Reduce should follow the same pattern.

Defined in `include/ttlang/Dialect/TTL/IR/TTLOps.td` (~line 583):
```tablegen
def TTL_TileBcastOp : TTL_TileOp<"tile_bcast"> {
  let arguments = (ins TTL_Tile:$input, TTL_Tile:$output,
                       I64ArrayAttr:$dims);
  let results = (outs TTL_Tile:$result);
}
```

## What to Implement

### 1. TTL Dialect: Define reduce tile ops

**File:** `include/ttlang/Dialect/TTL/IR/TTLOps.td`

Add two new tile ops near `TTL_TileBcastOp` (~line 590):

```tablegen
def TTL_TileReduceSumOp : TTL_TileOp<"tile_reduce_sum"> {
  let summary = "Reduce-sum a tile along specified dimension";
  let description = [{
    Reduces a tile by summing along the specified dimension.
    Requires a scaler tile (typically 1/N for mean, or 1.0 for raw sum).
    dims=[0] reduces rows (output: 1 row, N cols).
    dims=[1] reduces columns (output: N rows, 1 col).
  }];
  let arguments = (ins TTL_Tile:$input, TTL_Tile:$scaler, I64ArrayAttr:$dims);
  let results = (outs TTL_Tile:$result);
}

def TTL_TileReduceMaxOp : TTL_TileOp<"tile_reduce_max"> {
  let summary = "Reduce-max a tile along specified dimension";
  let description = [{
    Reduces a tile by taking the max along the specified dimension.
    Requires a scaler tile (typically 1.0).
    dims=[0] reduces rows (output: 1 row, N cols).
    dims=[1] reduces columns (output: N rows, 1 col).
  }];
  let arguments = (ins TTL_Tile:$input, TTL_Tile:$scaler, I64ArrayAttr:$dims);
  let results = (outs TTL_Tile:$result);
}
```

### 2. TTL to TTKernel lowering

**File:** `lib/Dialect/TTL/Transforms/ConvertTTLTileOpsToTTKernel.cpp`

Add lowering patterns for the new ops. Follow the pattern used by `TTL_TileBcastOp` lowering. The reduce lowering needs to emit:

```
ttkernel.reduce_init %in_cb, %scaler_cb, %out_cb {reduce_type, reduce_dim}
ttkernel.reduce_tile %in_cb, %scaler_cb, %in_idx, %scaler_idx, %dst_idx {reduce_type, reduce_dim}
ttkernel.reduce_uninit
```

**Dimension mapping:**
- `dims=[0]` → `ReduceDim::REDUCE_ROW` (sum/max across rows, each column gets one value)
- `dims=[1]` → `ReduceDim::REDUCE_COL` (sum/max across columns, each row gets one value)

Note: The `reduce_tile` API processes one tile at a time and accumulates into DST. For a single tile, call `reduce_tile` once. For multi-tile reductions (the cross-tile part), the tt-lang kernel's Python loop handles calling the operation multiple times — each call accumulates into the same DST slot.

**Key detail:** The reduce hardware op works within a single 32×32 tile. It collapses rows or columns within that tile. The cross-tile accumulation (reducing across multiple tiles) is handled by calling `reduce_tile` multiple times with the same DST index — the hardware accumulates automatically.

### 3. Python AST compiler: Register the new functions

**File:** `python/ttl/_src/ttl_ast.py`

The function `_resolve_ttl_function` dispatches via `self._fn_map`. Need to add entries for `reduce_sum` and `reduce_max` that:
1. Accept arguments: `(input_block, scaler_block, output_block, dims=[...])`
   - Note: the current simulator API is `ttl.math.reduce_sum(block, scaler, output, dims=[0])`
2. Emit the corresponding `TTL_TileReduceSumOp` / `TTL_TileReduceMaxOp` MLIR ops
3. Handle the `dims` keyword argument

Look at how `ttl.math.broadcast` is handled — it also has a `dims` argument and an output block parameter. The reduce functions should follow the same pattern.

### 4. Python code generation (if needed)

**File:** `python/gen_elementwise.py`

If the auto-generation from `TTLElementwiseOps.def` can't handle the extra arguments (scaler, dims), the reduce functions may need to be manually registered instead of auto-generated. Check how `broadcast` is registered — it's NOT in the `.def` file but is manually handled in `ttl_ast.py`.

## Semantics

### reduce_sum

```python
# In a @ttl.compute() function:
with input_dfb.wait() as inp, scaler_dfb.wait() as sc, output_dfb.reserve() as out:
    out.store(ttl.math.reduce_sum(inp, sc, out, dims=[0]))
```

- `inp`: input tile block
- `sc`: scaler tile (multiply each element by scaler during reduction)
- `out`: output tile block (required by compiler, same as broadcast pattern)
- `dims=[0]`: reduce rows → output tile has values only in row 0
- `dims=[1]`: reduce columns → output tile has values only in column 0

For `dims=[0]` (reduce rows): `out[0, j] = sum_i(inp[i, j] * sc[i, j])`
For `dims=[1]` (reduce cols): `out[i, 0] = sum_j(inp[i, j] * sc[i, j])`

### reduce_max

Same interface as reduce_sum but takes element-wise max instead of sum.

For `dims=[0]` (reduce rows): `out[0, j] = max_i(inp[i, j])`
For `dims=[1]` (reduce cols): `out[i, 0] = max_j(inp[i, j])`

Note: For reduce_max, the scaler is typically all 1.0s. The hardware still requires the scaler CB.

### Multi-tile reduction pattern

The hardware `reduce_tile` accumulates into DST. For reducing across multiple tiles (e.g., summing 28 tiles for RMSNorm), the tt-lang kernel calls `reduce_sum` in a loop:

```python
# Sum 28 tiles into one, reducing columns within each tile
with scaler_dfb.wait() as sc:
    # First tile: init
    with x_dfb.wait() as t0, acc_dfb.reserve() as acc:
        acc.store(ttl.math.reduce_sum(t0, sc, acc, dims=[0]))
    # Remaining tiles: accumulate (reduce + add to previous)
    for _ in range(27):
        with x_dfb.wait() as t, acc_dfb.wait() as prev:
            with acc_dfb.reserve() as acc:
                # reduce this tile, then add to previous
                # This might need a fused reduce+accumulate pattern,
                # OR two steps: reduce then add
                ...
```

**Important:** The exact multi-tile accumulation pattern depends on whether `reduce_tile` auto-accumulates into DST across calls, or whether each call is independent. If it auto-accumulates (like matmul does), the pattern is simpler. The implementer should check the hardware behavior.

## Test Plan

### Test 1: reduce_sum single tile, dims=[0] (reduce rows)

```python
# test/python/test_reduce_sum_row.py
@ttl.kernel(grid=(1, 1))
def reduce_sum_row_kernel(X, scaler, Y):
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=1)
    s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(X[0, 0], blk)
            tx.wait()
        with s_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.compute()
    def compute():
        with x_dfb.wait() as x_blk, s_dfb.wait() as s_blk, y_dfb.reserve() as y_blk:
            y_blk.store(ttl.math.reduce_sum(x_blk, s_blk, y_blk, dims=[0]))

    @ttl.datamovement()
    def write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, Y[0, 0])
            tx.wait()

# Test:
# X = random [32, 32] bf16
# scaler = ones [32, 32] bf16
# Expected: Y[0, j] = sum_i(X[i, j]) for j in 0..31, Y[i>0, :] = 0
# Validate: PCC(Y[0,:], X.sum(dim=0)) > 0.99
```

### Test 2: reduce_sum single tile, dims=[1] (reduce columns)

```python
# Same structure but dims=[1]
# Expected: Y[i, 0] = sum_j(X[i, j]) for i in 0..31, Y[:, j>0] = 0
# Validate: PCC(Y[:,0], X.sum(dim=1)) > 0.99
```

### Test 3: reduce_max single tile, dims=[0]

```python
# Same structure but ttl.math.reduce_max
# Expected: Y[0, j] = max_i(X[i, j])
# Validate: PCC(Y[0,:], X.max(dim=0).values) > 0.99
```

### Test 4: reduce_max single tile, dims=[1]

```python
# Expected: Y[i, 0] = max_j(X[i, j])
# Validate: PCC(Y[:,0], X.max(dim=1).values) > 0.99
```

### Test 5: reduce_sum with scaler (for RMSNorm mean)

```python
# scaler = full(1/32) [32, 32] — divides by column count to get mean
# Expected: Y[0, j] = mean_i(X[i, j])
# This tests that the scaler is correctly applied during reduction
```

### Test 6: reduce_sum multi-tile (if accumulation works)

```python
# Input: [1, 4] tiles = [32, 128]
# Reduce each tile with dims=[0], then sum the 4 reduced tiles
# Expected: Y[0, j] = sum over all 128 rows in the original tensor at column j
```

### Test 7: Integration — RMSNorm fully on device

```python
# Full RMSNorm using reduce_sum:
# 1. Square: sq = x * x
# 2. Reduce columns per tile: ttl.math.reduce_sum(sq, scaler_1_over_hidden, dims=[1])
# 3. Accumulate across tiles (sequential add)
# 4. Rsqrt
# 5. Broadcast + multiply
# Compare against torch RMSNorm, PCC > 0.98
```

### Test 8: Integration — Softmax fully on device

```python
# Full softmax using reduce_max + reduce_sum:
# 1. Find row-max: ttl.math.reduce_max(scores, ones, dims=[1]) per tile, then max across tiles
# 2. Subtract max, exp
# 3. Find row-sum: ttl.math.reduce_sum(exp_scores, ones, dims=[1]) per tile, then sum across tiles
# 4. Divide by sum
# Compare against torch softmax, PCC > 0.98
```

## Files to Modify

| File | Change |
|------|--------|
| `include/ttlang/Dialect/TTL/IR/TTLOps.td` | Add `TTL_TileReduceSumOp`, `TTL_TileReduceMaxOp` |
| `lib/Dialect/TTL/Transforms/ConvertTTLTileOpsToTTKernel.cpp` | Add lowering to `ttkernel.reduce_init` + `reduce_tile` + `reduce_uninit` |
| `python/ttl/_src/ttl_ast.py` | Register `reduce_sum`, `reduce_max` in `_fn_map`, handle dims/scaler args |
| `test/python/test_reduce_sum_row.py` | New test file (Tests 1-2) |
| `test/python/test_reduce_max.py` | New test file (Tests 3-4) |
| `test/python/test_reduce_scaler.py` | New test file (Test 5) |

## Reference Files

| File | Why |
|------|-----|
| `include/ttlang/Dialect/TTL/IR/TTLOps.td` lines 583-600 | `TTL_TileBcastOp` — same pattern (tile op with dims + extra args) |
| `lib/Dialect/TTL/Transforms/ConvertTTLTileOpsToTTKernel.cpp` | Existing broadcast lowering to follow |
| `python/ttl/_src/ttl_ast.py` line 344 | Where `_fn_map` lookup happens |
| `third-party/tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td` lines 473-507 | TTKernel reduce ops (target of lowering) |
| `third-party/tt-mlir/lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp` | EmitC reduce handling (already done) |
| `third-party/tt-metal/tt_metal/hw/inc/api/compute/reduce.h` | Hardware API (final target) |
