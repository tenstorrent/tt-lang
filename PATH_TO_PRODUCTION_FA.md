# Path from Current FA Demonstrator to Production Performance

## Current State vs. Production FlashAttention

### What We Have Now

**Single-Tile FA Demonstrator:**
```python
# Processes ONE tile at a time
K_T = K_block.transpose()      # 1 tile (32x32)
S = Q_block @ K_T               # 1 Q tile @ 1 K tile
P = S.exp()                     # 1 tile
result = P_norm.recip()         # 1 tile
```

**Characteristics:**
- ✅ 6 operations fused
- ✅ Works on 2x2 grid (4 cores)
- ❌ No loops over KV chunks
- ❌ No running statistics
- ❌ Processes only 32x32 elements
- ❌ No double buffering

### What Production FA Does

**Multi-Block FA (from tech report):**
```cpp
// Outer loop: iterate over KV chunks
for (kv_chunk = 0; kv_chunk < num_kv_blocks; kv_chunk++) {
    // Load K, V chunks
    load_kv_chunk(kv_chunk);

    // Compute attention scores: Q @ K^T
    S = matmul(Q_chunk, K_chunk_T);

    // Update running statistics
    m_new = rowmax(S, m_old);           // Running max
    l_new = rowsum(exp(S - m_new));     // Running sum

    // Compute normalized attention
    P = exp(S - m_new) / l_new;

    // Apply correction factor
    correction = exp(m_old - m_new);
    O_acc = correction * O_acc + P @ V_chunk;

    // Update state
    m_old = m_new;
    l_old = l_new;
}
```

**Characteristics:**
- ✅ Loops over multiple KV chunks (handles long sequences)
- ✅ Running max/sum updated incrementally
- ✅ Single pass (online algorithm)
- ✅ 8x10 grid (80 cores, massive parallelism)
- ✅ Double buffering (concurrent DMA + compute)
- ✅ Causal masking (efficiency)

## Gap Analysis

### Gap 1: Looping Over KV Chunks

**What's needed:**
```python
@compute()
async def flash_attention_loop(Q_cb, K_cb, V_cb, out_cb):
    NUM_KV_BLOCKS = 4  # Example

    # Initialize running statistics
    m_old = fill(-inf, shape=(32, 1))  # NOT IMPLEMENTED
    l_old = fill(0, shape=(32, 1))      # NOT IMPLEMENTED
    O_acc = fill(0, shape=(32, 32))     # NOT IMPLEMENTED

    for kv_idx in range(NUM_KV_BLOCKS):  # Python for loop
        K_block = K_cb.pop()             # Load next K chunk
        V_block = V_cb.pop()             # Load next V chunk

        # Compute scores
        S = Q_block @ K_block.transpose()

        # Update statistics
        m_new = rowmax(S, m_old)         # NOT IMPLEMENTED
        P = (S - m_new).exp()
        l_new = rowsum(P, l_old)         # NOT IMPLEMENTED

        # Apply attention
        O_acc = O_acc + P @ V_block      # Second matmul!

        m_old = m_new
        l_old = l_new
```

**Missing pieces:**
1. ❌ **Reduction ops**: `rowmax()`, `rowsum()` not exposed in Python
2. ❌ **Fill ops**: Can't initialize accumulators
3. ❌ **State management**: Running variables (m_old, l_old, O_acc) across iterations
4. ❌ **Second matmul**: P @ V blocked by temp alloc issue
5. ❌ **Correction factor**: `exp(m_old - m_new) * O_acc` needs element-wise ops on accumulated state

**Status:** Python `for` loops work, but operations inside aren't all implemented.

### Gap 2: Reduction Operations

**D2M has reduction ops:**
```tablegen
// In D2MGenericRegionOps.td:
def D2M_TileReduceSumOp  // Exists!
def D2M_TileReduceMaxOp  // Exists!
```

**Need to expose in Python:**
```python
# In operators.py:
def rowmax(ast_self: "TensorBlock", axis: int = -1) -> "TensorBlock":
    """Row-wise maximum reduction."""
    inp = ast_self
    ctx = inp.type.context

    # Create linalg.generic with reduction iterator
    # Input: [M, N], Output: [M, 1]
    reduction_map = affine_map<(d0, d1) -> (d0, d1)>  # input
    output_map = affine_map<(d0, d1) -> (d0)>          # output (reduced)

    return _create_linalg_generic_reduction(
        inp,
        output_shape=[inp.type.shape[0], 1],
        affine_maps=[reduction_map, output_map],
        iterator_types=["parallel", "reduction"],
        tile_op_builder=lambda result_type, inp_arg, acc_arg:
            d2m.tile_reduce_max(result_type, inp_arg, acc_arg)
    )
```

**Estimated effort**: 2-3 hours to implement and test

### Gap 3: Fill / Initialization Operations

**Needed:**
```python
m_old = fill(-inf, shape=(32, 1))
l_old = fill(0, shape=(32, 1))
O_acc = fill(0, shape=(32, 32))
```

**Two approaches:**

**A) Use d2m.constant:**
```python
# Check if d2m has constant op
constant = d2m.constant(value=-float('inf'), shape=(32, 1))
```

**B) Initialize from host:**
```python
# Pass pre-filled tensors as additional inputs
init_max = torch.full((32, 1), -float('inf'))
init_sum = torch.zeros((32, 1))

# In kernel:
m_old = init_max_cb.pop()  # Use as initialization
```

**Estimated effort**: 1-2 hours

### Gap 4: State Management Across Loop Iterations

**Challenge:**
```python
for kv_idx in range(NUM_KV_BLOCKS):
    m_new = rowmax(S, m_old)  # Reads m_old from previous iteration
    m_old = m_new              # Updates for next iteration
```

**How it works in Python:**
- Variables (m_old, l_old, O_acc) are SSA values in MLIR
- Each loop iteration creates new SSA values
- Loop-carried dependencies handled by compiler

**What we need:**
- ✅ Python for loops already lower to scf.for or affine.for
- ✅ SSA phi nodes handle loop-carried state automatically
- ⚠️ Need to verify tensors can be loop-carried (not just scalars)

**Likely works as-is**, but needs testing.

**Estimated effort**: 1 hour testing

### Gap 5: Multiple Tiles Per Core

**Current**: Single 32x32 tile
**Production**: Each core processes MxN tiles

**Example with 2x2 tiles:**
```python
@pykernel_gen(
    grid=(2, 2),
    block_factors=[(2, 2), (2, 2), (2, 2)]  # 2x2 tiles per core
)
def fa_multi_tile(Q, K, V, out):
    @compute()
    async def comp(Q_cb, K_cb, V_cb, out_cb):
        Q_block = Q_cb.pop()  # Now contains 2x2 tiles!

        # Operations automatically iterate over tiles
        K_T = K_block.transpose()  # Processes all 4 tiles
        S = Q_block @ K_T           # 2x2 @ 2x2 matmul
```

**Status:**
- ⚠️ Hit "Insufficient DST capacity" with 2x2 tiles
- **Cause**: 4 tiles * multiple ops exceeds 16 DST slots
- **Fix needed**: Subblocking or tiling within kernel

**The compiler tries to load ALL 4 tiles into DST simultaneously:**
```
2x2 tiles * 4 DST slots/tile = 8 slots just for operands
+ intermediate results = easily >16 total
```

**Solution - Tile-level subblocking:**
```python
# Compiler should generate (but doesn't yet):
for tile_i in range(2):
    for tile_j in range(2):
        # Load one tile at a time
        K_tile = load_single_tile(K_block, tile_i, tile_j)
        # Compute on single tile
        result_tile = compute(K_tile)
        # Store result
        store_single_tile(result, tile_i, tile_j)
```

**Estimated effort**: Requires compiler support for tile-level iteration (significant - weeks)

### Gap 6: Grid Scaling (4 cores → 80 cores)

**Current**: Works on 2x2 = 4 cores
**Production**: 8x10 = 80 cores

**Testing larger grids:**
```python
@pykernel_gen(grid=(8, 10), block_factors=[(1,1), (1,1), (1,1)])
def fa_large_grid(Q, K, V, out):
    # Should work! Just more cores processing shards in parallel
```

**Expected to work because:**
- ✅ Grid distribution is handled by MetalLayoutAttr
- ✅ Each core runs same kernel (SPMD)
- ✅ DMA indexing uses `core_index(0/1)` already

**Potential issues:**
- ⚠️ May need multicast DMA for efficiency (exists, but needs testing)
- ⚠️ Synchronization between cores if needed
- ⚠️ Memory capacity on each core

**Estimated effort**: 1-2 hours testing different grid sizes

## Roadmap to Production-Level FA

### Phase 1: Complete Algorithm (1-2 days)

**Step 1**: Expose reduction ops in Python (2-3 hours)
```python
def rowmax(self, axis=-1):  # Implement
def rowsum(self, axis=-1):  # Implement
```

**Step 2**: Implement fill/constant (1-2 hours)
```python
def fill(value, shape):  # Implement or use host tensors
```

**Step 3**: Test loop-carried state (1 hour)
```python
# Verify this pattern works:
acc = initial
for i in range(N):
    acc = acc + compute(i)
```

**Step 4**: Add correction factor ops (1 hour)
```python
# Element-wise multiply with broadcast:
correction = exp(m_old - m_new)  # Element-wise
O_acc = correction * O_acc        # Element-wise with accumulator
```

**Deliverable**: Single-tile FA with correct algorithm

### Phase 2: Fix Multiple Matmuls (2-4 days)

**Option A: MemRefProvenanceAnalysis** (Recommended - 1-2 days)
- Implement 150-line utility
- Refactor getCB, getInOrOutCB to use it
- Test with P @ V pattern

**Option B: Type Converter Enhancement** (Quick hack - 2 hours)
- Add `!memspace` → `CBType` conversion rule
- Test and iterate

**Deliverable**: Full FA with both matmuls in one kernel

### Phase 3: Multi-Tile Support (1-2 weeks)

**Step 1**: Understand DST capacity limits
- Profile current DST usage
- Identify allocation strategy

**Step 2**: Implement tile-level tiling
- Add pass to tile linalg.generic over tile dimensions
- Generate nested loops: outer=tiles, inner=within-tile

**Step 3**: Test with 2x2, 4x4 tile blocks

**Deliverable**: FA handling multiple tiles per core

### Phase 4: Performance Optimization (Ongoing)

**Step 1**: Double buffering (uses existing multi-CB infrastructure)
**Step 2**: Multicast DMA for weight sharing
**Step 3**: Causal masking (conditional execution)
**Step 4**: Grid scaling tests (8x10 cores)

**Deliverable**: Performance-competitive FA

## Immediate Next Steps (Your Next Session)

### Must-Have (2-3 hours)

1. **Implement rowmax/rowsum** in Python operators
   - Expose existing D2M reduction ops
   - Create linalg.generic with reduction iterator
   - Test standalone

2. **Add fill operation**
   - Either d2m.constant or host-side initialization
   - Test with -inf and 0 values

3. **Build looped FA**
   - Use Python for loop over KV chunks
   - Initialize running max/sum
   - Update statistics each iteration
   - Test with 2-4 KV blocks

### Should-Have (4-6 hours)

4. **Fix multiple matmuls**
   - Try type converter hack first (quick)
   - If that fails, start MemRefProvenanceAnalysis

5. **Test grid scaling**
   - Try 4x4, 8x8 grids
   - Measure compilation time
   - Check generated code quality

### Nice-to-Have (8+ hours)

6. **Multi-tile support**
   - Investigate subblocking pass
   - Implement tile-level iteration
   - Test with 2x2 tiles

## Specific Code Additions Needed

### 1. Reduction Operators

```python
# In operators.py:
def _create_linalg_generic_reduction(
    inp,
    output_shape: List[int],
    affine_maps: List[AffineMap],
    iterator_types: List[str],  # e.g., ["parallel", "reduction"]
    tile_op_builder: Callable,
) -> "TensorBlock":
    """Create linalg.generic for reduction operations."""
    ctx = inp.type.context

    # Create accumulator tensor
    out_type = RankedTensorType.get(
        output_shape, inp.type.element_type, inp.type.encoding
    )
    # Initialize accumulator
    # For max: -inf, for sum: 0
    empty = d2m.empty(out_type)

    # Rest similar to _create_linalg_generic_unary
    # but with reduction iterator type
    ...

class TensorBlock:
    def rowmax(ast_self, dim: int = -1) -> "TensorBlock":
        """Row-wise maximum reduction along dimension."""
        inp = ast_self
        # ... create reduction linalg.generic
        return _create_linalg_generic_reduction(
            inp,
            output_shape=[shape[0], 1],  # Reduce last dim
            affine_maps=[
                affine_map<(d0, d1) -> (d0, d1)>,  # Input
                affine_map<(d0, d1) -> (d0)>        # Output (reduced)
            ],
            iterator_types=["parallel", "reduction"],
            tile_op_builder=lambda result_type, inp_arg, acc_arg:
                d2m.tile_reduce_max(result_type, inp_arg, acc_arg)
        )

    def rowsum(ast_self, dim: int = -1) -> "TensorBlock":
        """Row-wise sum reduction."""
        # Similar to rowmax but with tile_reduce_sum
        ...
```

### 2. Fill Operation

**Option A - Use host tensors:**
```python
# Pass initialization tensors as CB inputs
@pykernel_gen(
    block_factors=[
        (1,1),  # Q
        (1,1),  # K
        (1,1),  # V
        (1,1),  # m_init (-inf)
        (1,1),  # l_init (0)
        (1,1),  # O_init (0)
        (1,1),  # out
    ],
    grid=(1,1)
)
def fa_with_init(Q, K, V, m_init, l_init, O_init, out):
    m_init_stream = Stream(m_init)
    # ...

    @compute()
    async def comp(Q_cb, K_cb, V_cb, m_cb, l_cb, O_cb, out_cb):
        m_old = m_cb.pop()  # Load -inf initialization
        l_old = l_cb.pop()  # Load 0 initialization
        O_acc = O_cb.pop()  # Load 0 initialization
        # ... rest of algorithm
```

**Option B - d2m.constant (if it exists):**
```python
m_old = constant(-float('inf'), shape=(32, 1))
```

### 3. Loop-Based FA Kernel

```python
@pykernel_gen(
    block_factors=[
        (1, 1),  # Q: single chunk, fixed
        (1, 1),  # K: one chunk at a time
        (1, 1),  # V: one chunk at a time
        (1, 1),  # out
    ],
    grid=(1, 1)
)
def flash_attention_looped(Q, K_all, V_all, out):
    """
    Q: [seq_len_q, d_head] - single Q chunk
    K_all: [num_kv_blocks, seq_len_kv, d_head] - all K chunks
    V_all: [num_kv_blocks, seq_len_kv, d_head] - all V chunks
    """
    Q_stream = Stream(Q)
    K_stream = Stream(K_all)
    V_stream = Stream(V_all)

    NUM_KV_BLOCKS = 4  # Hardcoded for now

    @compute()
    async def attention_compute(Q_cb, K_cb, V_cb, out_cb):
        # Load Q once (stays in CB)
        Q_block = Q_cb.pop()

        # Initialize state (using host tensors passed as extra CBs)
        # OR allocate in L1 and initialize with first values
        m_old = Q_block.rowmax()  # Use Q as mock init
        l_old = Q_block.rowmax()  # Placeholder
        O_acc = Q_block @ Q_block.transpose()  # Mock init

        # Loop over KV chunks
        for kv_idx in range(NUM_KV_BLOCKS):
            K_block = K_cb.pop()  # Pop next K chunk
            V_block = V_cb.pop()  # Pop next V chunk

            # Compute attention scores
            K_T = K_block.transpose()
            S = Q_block @ K_T

            # Update running max (if rowmax works)
            m_new = S.rowmax(dim=-1)  # Shape: [seq_q, 1]

            # Compute attention weights
            S_stable = S - m_new  # Broadcast subtract
            P = S_stable.exp()

            # Update running sum (if rowsum works)
            l_new = P.rowsum(dim=-1)  # Shape: [seq_q, 1]

            # Correction factor
            correction = (m_old - m_new).exp()

            # Update output accumulator
            # PROBLEM: Need P @ V (second matmul!)
            # Workaround: Do P @ V in separate kernel
            O_delta = P @ V_block  # BLOCKED

            # This update would work if we had O_delta:
            O_acc = correction * O_acc + O_delta

            # Update state
            m_old = m_new
            l_old = l_new

        # Write final output
        out_block = out_cb.reserve()
        out_block.store(O_acc)
        out_cb.pop()

    @datamovement()
    async def dm_reader(Q_cb, K_cb, V_cb, out_cb):
        idx = core_index(0) + core_index(1)

        # Load Q once
        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()

        # Load each KV chunk in loop
        for kv_idx in range(NUM_KV_BLOCKS):
            dma(K_stream[kv_idx, idx, 0], K_cb.reserve()).wait()
            dma(V_stream[kv_idx, idx, 0], V_cb.reserve()).wait()

    return Program(attention_compute, dm_reader)(Q, K_all, V_all, out)
```

**Blocks on**: Second matmul (P @ V)

## Practical Implementation Strategy

### Week 1: Core Algorithm

**Day 1-2**: Add reductions + fill
- Expose rowmax, rowsum in Python
- Implement fill via host tensors
- Test reduction ops standalone

**Day 3**: Implement loop structure
- Build looped FA WITHOUT second matmul
- Test loop-carried state
- Verify statistics update correctly

**Day 4-5**: Fix multiple matmuls
- Try type converter hack
- If that fails, implement MemRefProvenanceAnalysis
- Test complete algorithm

**Deliverable**: Functionally correct single-tile looped FA

### Week 2: Scale to Multiple Tiles

**Day 1-2**: Understand DST allocation
- Profile DST usage for 2x2 tiles
- Find why it exceeds capacity
- Research subblocking approach

**Day 3-4**: Implement tiling solution
- Either: Better DST allocation
- Or: Tile-level subblocking pass
- Or: Reduce intermediate DST usage

**Day 5**: Test scaling
- 2x2 tiles/core
- 4x4 tiles/core
- Large grids (8x8, 8x10)

**Deliverable**: FA handling production-scale tile counts

### Week 3: Performance Optimization

**Day 1**: Double buffering
- Use multiple CB slots
- Overlap DMA + compute

**Day 2**: Multicast DMA
- Share K/V across cores in same row/column
- Test communication patterns

**Day 3**: Causal masking
- Conditional execution in loops
- Skip unnecessary KV chunks

**Day 4-5**: Profiling and tuning
- Measure generated code quality
- Compare with hand-written

**Deliverable**: Performance-competitive FA

## Quick Win for Next Session (2 hours)

**Build "Looped FA v1" without second matmul:**

1. Implement rowmax, rowsum (30m)
2. Add fill via host tensors (30m)
3. Build loop with running stats (30m)
4. Test with 2-4 KV chunks (30m)

**Result**: Demonstrates MOST of FA algorithm, only missing final P @ V

```python
# This should compile:
for kv_idx in range(4):
    K = K_cb.pop()
    S = Q @ K.transpose()
    m_new = S.rowmax()
    P = (S - m_new).exp()
    l_new = P.rowsum()
    # Update stats...
    # (Skip P @ V for now)
```

Then tackle multiple matmuls as separate effort!

## Summary

**To reach production FA performance, we need:**

| Component | Status | Effort | Priority |
|-----------|--------|--------|----------|
| Reduction ops (rowmax/rowsum) | Missing | 2-3h | **HIGH** |
| Fill/init operations | Missing | 1-2h | **HIGH** |
| Loop over KV chunks | Works | Test 1h | **HIGH** |
| Multiple matmuls | Blocked | 2-8h | **HIGH** |
| Multi-tile per core | DST limit | Weeks | MEDIUM |
| Large grids (8x10) | Untested | 1-2h | MEDIUM |
| Double buffering | Not impl | 4-8h | LOW |
| Causal masking | Not impl | 2-4h | LOW |

**Critical path**: Reductions + Multiple Matmuls + Looping = Complete algorithm
**Performance path**: Multi-tile + Grid scaling + Optimizations = Production perf

The compiler infrastructure is mostly there - we just need to expose the right ops and fix the temp alloc memspace issue!
