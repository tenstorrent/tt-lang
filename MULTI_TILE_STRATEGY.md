# Multi-Tile Flash Attention - Implementation Strategy

## The Performance Gap

**Our current demo**: Processes 1 tile (32x32 = 1024 elements)
**Production FA**: Processes hundreds of tiles per core on 80 cores

### Hand-Written FA Architecture (from tech report)

```
8x10 grid = 80 Tensix cores
Each core processes:
  - Q chunks: Multiple 32x32 tiles
  - K/V chunks: Iterate over all K/V blocks
  - Output: Accumulated attention results

Performance: 20x speedup from parallelization + tiling
```

## Problem: Our 2x2 Tiles Failed

**When we tried:**
```python
@pykernel_gen(block_factors=[(2,2), (2,2), (2,2)], grid=(1,1))
```

**Error:**
```
Assertion `maxDstSliceIdx < numDstSlices` failed
Insufficient DST capacity for all operands
```

### Root Cause Analysis

**DST capacity**: 16 tiles total (hardware limit)

**Our 6-op kernel with 2x2 tiles would need:**
```
Transpose intermediate:  2x2 = 4 tiles
Matmul accumulator:      2x2 = 4 tiles
Binary op operand 1:     2x2 = 4 tiles
Binary op operand 2:     2x2 = 4 tiles
Unary chain output:      2x2 = 4 tiles
                         ============
                Total:   20 tiles needed > 16 available ✗
```

The compiler tries to load ALL tiles simultaneously → exceeds capacity.

## Solutions

### Solution 1: Tile-Level Subblocking (Compiler Enhancement)

**Idea**: Process one tile at a time from the 2x2 block

**Current generated code (BROKEN):**
```mlir
// Tries to load all 4 tiles at once:
scf.for %i = 0 to 2 {
  scf.for %j = 0 to 2 {
    %all_tiles = load %cb[%i, %j]  // Parallel iteration
    // All 4 tiles in DST simultaneously ✗
  }
}
```

**Fixed with subblocking:**
```mlir
// Process one tile at a time:
scf.for %tile_i = 0 to 2 {
  scf.for %tile_j = 0 to 2 {
    // Load single tile
    %K_tile = load %K_cb[%tile_i, %tile_j]
    store %K_tile, %dst[0]

    // Transpose single tile
    transpose_tile(%dst[0])

    // Matmul for this tile pair
    %Q_tile = load %Q_cb[%tile_i, %tile_j]
    %S_tile = matmul(%Q_tile, %dst[0])

    // Continue 6-op chain on THIS tile
    %result_tile = compute_chain(%S_tile)

    // Store result for this tile
    store %result_tile, %out_cb[%tile_i, %tile_j]
  }
}
// Max DST usage: ~6 tiles (same as 1x1 case) ✓
```

**Where to implement:**
```cpp
// In lib/Dialect/D2M/Transforms/GenericTileComputeLoops.cpp:

// Current logic (around line 138):
if (outputVolume > dstCapacity) {
  // Try to tile the linalg op
  // Currently only handles specific patterns
}

// Enhanced logic needed:
if (outputVolume > dstCapacity) {
  // Compute tile-level subblock size
  int64_t tilesPerSubblock = dstCapacity / numOperations;

  // Tile the linalg.generic at TILE granularity
  // Create nested loops: outer=tile_blocks, inner=within_tile
  auto tilingOptions = linalg::LinalgTilingOptions()
      .setTileSizes({1, 1});  // One tile at a time

  auto tiledOp = linalg::tileLinalgOp(rewriter, linalgOp, tilingOptions);

  // Now each iteration processes 1 tile → fits in DST ✓
}
```

**Pros:**
- ✅ Solves DST capacity issue
- ✅ Enables arbitrary tile counts
- ✅ Maintains fusion benefits

**Cons:**
- ❌ Serializes tile processing (performance hit)
- ❌ Compiler complexity (tiling within tiling)
- ❌ Need careful loop structure

**Estimated effort**: 1-2 weeks (compiler pass modification + testing)

### Solution 2: Reduce Intermediate DST Usage (Optimization)

**Idea**: Reuse DST slots more aggressively

**Current allocation strategy:**
```cpp
// Allocates fresh DST slice for each operand load
for (operandIdx : getOperandsLoadFromDstRegister()) {
  dstSlice = allocate();  // New slice every time
  load_to_dst[dstSlice];
}
```

**Optimized strategy:**
```cpp
// Reuse slots as soon as value is consumed
for (operandIdx : getOperandsLoadFromDstRegister()) {
  if (value_last_use) {
    dstSlice = free_list.allocate();  // Reuse freed slots
  } else {
    dstSlice = allocate_new();
  }
}
```

**With liveness analysis**, could get:
```
2x2 tiles, 6 ops:
- Transpose: 4 tiles (reused after matmul consumes)
- Matmul: 4 tiles for accumulator
- Binary: 4+4 tiles (one freed immediately)
- Unary: 4 tiles (in-place, reuse binary output)

Max live: ~12 tiles (might fit in 16!)
```

**Pros:**
- ✅ Fits more tiles in DST
- ✅ Better utilization
- ✅ No serialization

**Cons:**
- ❌ Complex liveness tracking
- ❌ Still has limits (can't do 4x4 tiles)

**Estimated effort**: 1 week (liveness analysis + allocation strategy)

### Solution 3: Streaming/Pipelining (Advanced)

**Idea**: Process tiles in pipeline stages

**Pattern:**
```mlir
// Double-buffered processing:
Stage 1: Load tiles 0,1 → compute on 0 while loading 1
Stage 2: Load tiles 2,3 → compute on 1 while loading 2
Stage 3: Load tiles 4,5 → compute on 2 while loading 3
...
```

**Requires:**
- Explicit pipeline scheduling
- Multiple DST buffer sets
- Careful synchronization

**Pros:**
- ✅ Maximum throughput
- ✅ Hides latency
- ✅ Scales to arbitrary tile counts

**Cons:**
- ❌ Very complex
- ❌ Needs scheduling pass
- ❌ Hard to debug

**Estimated effort**: Months (research project)

## Immediate Workaround for Multi-Tile

**Use grid distribution instead of tiles/core:**

Instead of:
```python
# BROKEN: 2x2 tiles on 1 core
grid=(1, 1), block_factors=[(2,2), ...]
```

Do:
```python
# WORKS: 1 tile per core, 4 cores
grid=(2, 2), block_factors=[(1,1), ...]
```

**Effect:**
- Distributes 64x64 problem across 4 cores
- Each core processes 32x32 (fits in DST)
- Same total work, different parallelization

**Tradeoff:**
- ✅ Works with current compiler
- ✅ Good parallelism
- ❌ More communication overhead (each core independent)
- ❌ Doesn't match production "tiles per core" model

## Comparison to Hand-Written FA

| Aspect | Hand-Written FA | Our DSL FA | Gap |
|--------|----------------|------------|-----|
| **Operations** | All (matmul, exp, reduce, etc) | 6-op subset | Reductions |
| **Tiles/core** | Many (tiled loop) | 1 (DST limit) | **Subblocking** |
| **Grid** | 8x10 (80 cores) | 2x2 (4 cores tested) | Scaling (easy) |
| **KV loop** | Yes (outer loop) | Not yet | **Algorithm** |
| **Multiple matmuls** | Yes (Q@K^T, P@V) | Blocked | **Architecture** |
| **Reductions** | rowmax, rowsum | Not exposed | **API** |
| **Double buffer** | Yes | No | Optimization |
| **Performance** | 20x baseline | Unknown | **Validation** |

### Critical Gaps (Must-Fix)

1. **Reductions**: rowmax/rowsum implementation (3-4 hours)
2. **Multiple matmuls**: Type converter or MemRefProvenance (2-8 hours)
3. **Multi-tile**: Subblocking pass (1-2 weeks)

### Performance Gaps (Optimization)

4. Grid scaling: Test 8x10 (1 hour)
5. Double buffering: Multi-CB implementation (4-8 hours)
6. Streaming: Pipeline scheduling (research project)

## Recommended Immediate Action

**For next session (3-4 hours):**

1. **Implement rowmax as simple reduction**
   ```python
   # Start with basic version:
   def rowmax(self):
       # Use existing tile_reduce_max
       # Hardcode reduce_dim = columns
       # Test with 1x1 tile first
   ```

2. **Build simple looped FA**
   ```python
   for kv_idx in range(2):  # Just 2 iterations to start
       K = K_cb.pop()
       S = Q @ K.transpose()
       m = S.rowmax()  # New reduction!
       P = (S - m).exp()
       # (Skip P @ V for now)
   ```

3. **Test and iterate**
   - Verify loop structure compiles
   - Check generated MLIR
   - Validate data flow

**Deliverable**: Looped FA demonstrator (even without P @ V)

This proves the ALGORITHM structure works, then we tackle multiple matmuls as separate bug.

---

**Bottom line**: Multi-tile needs compiler subblocking (weeks of work). But we can demonstrate the ALGORITHM with single tiles + loops + reductions in a few hours!
