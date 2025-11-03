# Multiple Matmul Fusion Issue - Root Cause & Fixes

## The Problem

**We CAN fuse**: 6 operations including 1 matmul (transpose + matmul + 4 other ops) ✅
**We CANNOT fuse**: 2 matmuls in the same kernel ❌

### What Happens

```python
# This fails:
S = Q @ K_T  # First matmul
P = S.exp()
O = P @ V    # Second matmul - CRASH!
```

**Error:**
```
failed to legalize unresolved materialization from
  ('memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>') to
  ('!ttkernel.cb<1, !ttcore.tile<32x32, f32>>')
```

## Root Cause Analysis

### Step 1: Python → Initial MLIR

```mlir
// First matmul
%S_tensor = linalg.generic {
  %result = d2m.tile_matmul(Q, K_T, ...)
  yield %result
} → tensor<1x1x!ttcore.tile<>>

// Exp
%P_tensor = linalg.generic {
  %result = d2m.tile_exp(%S_tensor)
  yield %result
} → tensor<1x1x!ttcore.tile<>>

// Second matmul
%O_tensor = linalg.generic {
  %result = d2m.tile_matmul(%P_tensor, V, ...)
  yield %result
} → tensor<1x1x!ttcore.tile<>>
```

**Key**: Intermediate results (%S_tensor, %P_tensor) are tensors between operations.

### Step 2: After Bufferization

```mlir
// Function-level d2m.empty() - gets memspace from encoding:
%storage = d2m.empty() : tensor<..., #ttcore.metal_layout<..., l1>>
           ↓ bufferize
%alloc = memref.alloc() : memref<..., #ttcore.memory_space<l1>>  ✓ Has #l1!

// Inside linalg.generic body d2m.empty() - NO encoding:
%temp = d2m.empty() : tensor<1x1x!ttcore.tile<>>  // No layout!
        ↓ bufferize
%alloc = memref.alloc() : memref<1x1x!ttcore.tile<>>  ✗ NO MEMSPACE!
```

**THE PROBLEM**: Temps inside linalg.generic don't have encodings, so bufferized memrefs lack memspace attributes.

After bufferization, we get:
```mlir
// First matmul output
%temp1 = memref.alloc() : memref<1x1x!ttcore.tile<>>  // NO #l1 !!
store %matmul1_result, %temp1

// Exp
%temp2 = memref.alloc() : memref<1x1x!ttcore.tile<>>  // NO #l1 !!
%val = load %temp1  // Load from memref WITHOUT memspace
%exp_result = d2m.tile_exp(%val)
store %exp_result, %temp2

// Second matmul
%val2 = load %temp2  // Load from memref WITHOUT memspace
%O = d2m.tile_matmul(%val2, %V, ...)  // Matmul on temp alloc value!
```

### Step 3: D2M → TTKernel Conversion

**Matmul converter expects CBs:**

```cpp
// In D2MFPUOpsRewriter<TileMatmulOp>::matchAndRewrite:
auto cbA = getCB(rewriter, op.getA());  // Traces to find source CB
auto cbB = getCB(rewriter, op.getB());

// Creates:
ttkernel::MatmulInitOp(cbA, cbB, outCB, ...)
ttkernel::MatmulTilesOp(cbA, cbB, ...)
```

**getCB tracing logic:**
```cpp
Value getCB(rewriter, value) {
  if (load = value.getDefiningOp<memref::LoadOp>()) {
    memref = load.getMemref()

    // Check memory space
    if (memspace == DeviceL1) {
      return rewriter.getRemappedValue(memref)  // ✓ Works for real CBs
    }
    if (memspace == RegisterDst) {
      // Trace through dst spill/reload (Bug #5 fix)
      return getCB(..., original_source)  // ✓ Works for DST
    }
    if (NO MEMSPACE) {
      // Our attempted fix: trace through temp alloc
      return getCB(..., source_op_input)  // ⚠️ Partially works
    }
  }
}
```

**Type Converter rules:**
```cpp
MemRefType → Type conversion:
  if (memspace == DeviceL1)   → ttkernel::CBType    ✓
  if (memspace == RegisterDst) → index              ✓
  if (has StridedLayout)      → index              ✓
  if (NO MEMSPACE)            → ??? undefined!      ✗ BROKEN
```

**What fails:**
1. Second matmul loads value from `%temp2` (no memspace)
2. getCB tries to trace back, might find CB but...
3. Type converter sees memref without memspace, doesn't know what to do
4. Creates `unrealized_conversion_cast` trying to materialize the type
5. Cast fails to resolve → compilation error

**Why first matmul works:**
- Its inputs are from `d2m.wait` (CB parameters) → have #l1 memspace ✓
- Result can spill to CB or DST ✓

**Why second matmul fails:**
- Its inputs are from temp allocs (no memspace) → type converter confused ✗
- Even if getCB finds original CB, the temp alloc itself needs conversion ✗

## Workarounds

### Workaround 1: Two Separate Kernels (WORKS NOW)

**Split FA into two @pykernel_gen functions:**

```python
# Kernel 1: Attention scores + softmax
@pykernel_gen(...)
def attention_scores(Q, K, scores_out):
    @compute()
    async def comp(Q_cb, K_cb, scores_cb):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        scores_block = scores_cb.reserve()

        K_T = K_block.transpose()
        S = Q_block @ K_T
        S_stable = S - Q_block  # mock max
        P = S_stable.exp()
        P_norm = P * P  # mock normalization

        scores_block.store(P_norm)
        scores_cb.pop()

# Kernel 2: Apply attention to V
@pykernel_gen(...)
def apply_attention(scores, V, out):
    @compute()
    async def comp(scores_cb, V_cb, out_cb):
        scores_block = scores_cb.pop()
        V_block = V_cb.pop()
        out_block = out_cb.reserve()

        O = scores_block @ V_block  # Second matmul

        out_block.store(O)
        out_cb.pop()

# Usage:
scores = torch.zeros(seq_len, seq_len)
attention_scores(Q, K, scores)  # Kernel 1
apply_attention(scores, V, out)  # Kernel 2
```

**Pros:**
- ✅ Works with current compiler
- ✅ Clear separation of concerns
- ✅ Can optimize each kernel independently
- ✅ Easier to debug

**Cons:**
- ❌ Two kernel launches (overhead)
- ❌ Intermediate CB traffic (scores written/read)
- ❌ Can't fuse softmax normalization with second matmul

### Workaround 2: Use Real CBs Instead of Temps (UNTESTED)

**Explicitly store intermediates to CBs:**

```python
@pykernel_gen(...)
def fa_with_intermediate_cb(Q, K, V, out):
    Q_stream = Stream(Q)
    K_stream = Stream(K)
    V_stream = Stream(V)

    # Create extra CB for intermediate storage
    @compute()
    async def comp(Q_cb, K_cb, V_cb, intermediate_cb, out_cb):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        V_block = V_cb.pop()

        # First matmul
        K_T = K_block.transpose()
        S = Q_block @ K_T
        P = S.exp()

        # EXPLICITLY store to CB (not temp alloc)
        temp_block = intermediate_cb.reserve()
        temp_block.store(P)
        intermediate_cb.pop()  // Signal ready

        # Load from CB for second matmul
        P_reloaded = intermediate_cb.pop()
        O = P_reloaded @ V_block  // Second matmul

        out_block = out_cb.reserve()
        out_block.store(O)
        out_cb.pop()
```

**Theory**: By using actual CB operations instead of relying on d2m.empty(), we ensure memspace attributes.

**Pros:**
- ✅ Single kernel (no launch overhead)
- ✅ Uses existing CB infrastructure
- ✅ Might work with current fixes

**Cons:**
- ❌ Requires extra CB allocation
- ❌ More complex dataflow
- ❌ UNTESTED - might still hit issues

## Longer-Term Fixes

### Fix 1: Default L1 Memspace for Temps (TRIED - Type Issues)

**What we attempted:**
```python
# In operators.py _create_linalg_generic:
encoding = lhs.type.encoding or create_default_l1_layout(output_shape)
out_type = RankedTensorType.get(output_shape, element_type, encoding)
empty = d2m.empty(out_type)  // Now has encoding with L1!
```

**Why it failed:**
- Type mismatches between tensors with/without encodings
- Bufferization expects consistent encoding patterns
- Input encoding has specific grid/shard info that doesn't apply to intermediate results

**Proper implementation would need:**
```python
def create_intermediate_encoding(ctx, shape):
    """Create minimal encoding with ONLY memspace, no grid info."""
    return ttcore.ir.MetalLayoutAttr.get(
        ctx,
        shape,  # logical_shape (just the tile shape, e.g., [1,1])
        int(ttcore.OOBVal.Undef),
        int(ttcore.MemorySpace.DeviceL1),  # KEY: L1 memspace
        int(ttcore.TensorMemoryLayout.None_),
        None,  # No collapsed_intervals
        None,  # No index_map
    )
```

Then after bufferization:
```mlir
%temp = memref.alloc() : memref<1x1x!ttcore.tile<>, #ttcore.memory_space<l1>>
```

And type converter would see L1 → convert to CBType ✓

**Estimated effort**: 2-4 hours of careful type system work

### Fix 2: Enhance Type Converter (Architectural)

**Add explicit handling for memrefs without memspace:**

```cpp
// In D2MToTTKernelPass.cpp type converter:
typeConverter.addConversion([&](MemRefType type) -> std::optional<Type> {
  if (!type.getMemorySpace()) {
    // Temp allocation - treat as L1 CB
    // These are compiler-managed intermediates, safe to assume L1
    return getCBType(type);  // Convert to CB type
  }

  if (memspace == DeviceL1) return CBType;
  if (memspace == RegisterDst) return index;
  // ... existing logic
});
```

**Pros:**
- ✅ Minimal change (single location)
- ✅ Fixes all temp alloc issues at once
- ✅ No Python changes needed

**Cons:**
- ❌ Assumes all temps are L1 (might not always be true)
- ❌ Could mask real errors if temp allocs shouldn't be CBs

**Estimated effort**: 1-2 hours

### Fix 3: MemRefProvenanceAnalysis (Recommended - From Markdown)

**Implement proper tracking utility:**

```cpp
// New file: lib/Dialect/D2M/Utils/MemRefProvenance.h
enum class MemRefSource {
  CircularBuffer,   // BlockArgument from d2m.generic
  TempAllocation,   // memref.alloc from d2m.empty
  DstRegister,      // d2m.acquire_dst
  StreamLayout,     // d2m.stream_layout
};

ProvenanceInfo traceMemRefProvenance(Value memref) {
  // Unwrap views: subview, collapse_shape, cast, wait, reserve
  Value root = unwrapViews(memref);

  if (auto blockArg = dyn_cast<BlockArgument>(root)) {
    return {CircularBuffer, root, root.getDefiningOp()};
  }
  if (auto alloc = root.getDefiningOp<memref::AllocOp>()) {
    // Trace what STORED to this temp
    // Find the linalg.generic that produced it
    // Get that generic's CB inputs
    return {TempAllocation, root, alloc, traceToCB(alloc)};
  }
  if (auto acquire = root.getDefiningOp<d2m::AcquireDstOp>()) {
    return {DstRegister, root, acquire};
  }
  // ...
}

// Enhanced getCB using provenance:
Value getCB(rewriter, value) {
  auto prov = traceMemRefProvenance(value);

  switch (prov.source) {
    case CircularBuffer:
      return rewriter.getRemappedValue(prov.rootValue);

    case TempAllocation:
      // Trace through allocation to find source CB
      return prov.sourceCB;  // Pre-computed during analysis

    case DstRegister:
      // Existing dst tracing logic
      return traceThroughDst(prov);
  }
}
```

**Pros:**
- ✅ Clean abstraction
- ✅ Single source of truth for memref classification
- ✅ Reusable across all passes
- ✅ Fixes getCB, getInOrOutCB, and other helpers
- ✅ Better error messages possible

**Cons:**
- ❌ Significant implementation effort (150+ lines)
- ❌ Requires refactoring multiple passes

**Estimated effort**: 4-8 hours implementation + testing

### Fix 4: Force Better Fusion (Prevents Separate Linalg Ops)

**Make all ops fusible by padding iterator types:**

```python
# In operators.py:
def ensure_compatible_iteration_space(op1, op2):
    """Pad ops to same rank so linalg fusion succeeds."""
    max_rank = max(get_iteration_rank(op1), get_iteration_rank(op2))

    # Pad both to max_rank
    # E.g., exp (2D) + matmul (3D) → both become 3D
    # exp gets unused reduction dimension
```

**Result:** Single fused linalg.generic containing both matmuls:
```mlir
linalg.generic {
  %S = tile_matmul(Q, K_T, ...)
  %P = tile_exp(%S)
  %O = tile_matmul(%P, V, ...)  // In same block!
  yield %O
}
```

No temp allocations created → no memspace issues!

**Pros:**
- ✅ Eliminates temp allocs entirely
- ✅ Better fusion = better performance
- ✅ No type converter changes needed

**Cons:**
- ❌ Creates redundant loop iterations
- ❌ Complex to implement correctly
- ❌ May not work for all op combinations

**Estimated effort**: 6-12 hours

## Recommended Path

### Immediate (Next Session)

**Option A: Two-Kernel Approach** (30 minutes)
- Split FA into `attention_scores` + `apply_attention` kernels
- Proven to work, clean semantics
- Good enough for demonstrator

**Option B: Type Converter Hack** (1-2 hours)
- Add `!memspace` → `CBType` conversion rule
- Test with multiple matmuls
- Quick validation if it solves the issue

### Short-Term (Next Sprint)

**Implement MemRefProvenanceAnalysis** (1 day)
- Clean architectural solution
- Fixes entire class of issues
- Referenced in markdown as "ESSENTIAL for multi-op kernels"
- Replaces all ad-hoc getCB/lookThroughSubView hacks

### Long-Term (Architectural)

**Improve Fusion + Default Memspace** (1 week)
- Pad iterator types for better fusion
- Default L1 memspace on all d2m.empty() (with proper type handling)
- Consider linalg → TTKernel holistic converter (Proposal #2 from markdown)

## Why This Matters

**Current state:**
- ✅ Can build softmax component: `exp(Q @ K^T - max) / sum`
- ❌ Can't apply to V: `softmax(...) @ V` in same kernel

**With fix:**
- ✅ Complete single-kernel FA
- ✅ Better performance (no intermediate CB traffic)
- ✅ Unlocks other multi-matmul patterns (MLP blocks, multi-head attention)

## Testing the Fix

**Once implemented, test with:**

```python
@pykernel_gen(...)
def complete_fa(Q, K, V, out):
    @compute()
    async def comp(Q_cb, K_cb, V_cb, out_cb):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        V_block = V_cb.pop()
        out_block = out_cb.reserve()

        # First matmul: attention scores
        K_T = K_block.transpose()
        S = Q_block @ K_T

        # Softmax (simplified)
        P = S.exp()

        # Second matmul: apply to V
        O = P @ V_block  # THIS SHOULD WORK!

        out_block.store(O)
        out_cb.pop()
```

**Expected result after fix:**
- ✓ Compiles without unrealized_conversion_cast errors
- ✓ Generates two matmul_tiles calls
- ✓ Proper CB/DST traffic for intermediate results

## Summary Table

| Approach | Effort | Impact | Risk | Recommended |
|----------|--------|--------|------|-------------|
| Two kernels | 30m | Medium | None | ✅ Yes (immediate) |
| Type converter hack | 2h | High | Medium | ⚠️ Test first |
| MemRefProvenance | 8h | Very High | Low | ✅ Yes (proper fix) |
| Force fusion | 12h | Very High | High | ❌ No (too complex) |

**Recommendation**: Do two-kernel approach NOW for working demo, then implement MemRefProvenanceAnalysis as proper architectural fix.
