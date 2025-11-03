# Flash Attention Development Session Notes

## Summary

Continued work on Flash Attention implementation from previous session. Attempted to add reduction operations (rowmax) incrementally but encountered compiler limitations.

## Current State

✅ **Working**: 6-operation fused kernel
- transpose → matmul → subtract → exp → sqrt → recip
- Compiles successfully on 1x1 and 2x2 grids
- All existing tests passing

## What Was Attempted

### Reduction Operations (rowmax/rowsum)

**Goal**: Add `rowmax()` operation for proper Flash Attention numerical stability

**Implementation**:
1. Created `_create_linalg_generic_ternary()` helper function for 3-operand operations
2. Implemented `rowmax()` method using `tile_reduce_max` D2M operation
3. Used 3-operand signature: `tile_reduce_max(a, b, c, reduce_dim=C)`

**Result**: ❌ **Compiler crash in D2MGenericTileComputeLoops pass**

**Files modified**:
- `python/ttlang/operators.py`:
  - Added `_create_linalg_generic_ternary()` (lines 137-194)
  - Added `rowmax()` method (lines 457-502)

### Technical Findings

**Issue 1**: Initial MLIR generates correctly
```mlir
%7 = "d2m.tile_reduce_max"(%in, %in_0, %in_1) <{reduce_dim = #d2m<reduce_dim C>}>
  : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>)
  -> !ttcore.tile<32x32, f32>
```

**Issue 2**: Crashes during pass pipeline
- Pass: `D2MGenericTileComputeLoops`
- Error varies: `d2m.iter_index` legalization failure or segfault
- Appears to be missing compiler support for reductions in this context

### Why Reductions Are Complex

1. **3-operand signature**: `result = max(A * B, C)` or `result = sum(A * B, C)`
   - More complex than unary/binary ops
   - Requires element-wise multiply then reduction

2. **Tile-level vs element-level**:
   - Reduction happens WITHIN each tile, not across tiles
   - linalg.generic still iterates tiles in parallel
   - This is different from typical linalg reductions

3. **Accumulator semantics**:
   - Operand C is accumulator (loaded from DST)
   - For standalone reduction need proper initialization

## Next Steps (Recommended Priority)

### Option 1: Fix Reduction Support in Compiler (2-4 hours)

**Investigate and fix**:
- Why `D2MGenericTileComputeLoops` crashes on reductions
- Check if reductions need special handling in that pass
- Look at `d2m.iter_index` legalization issue

**Files to check**:
- `lib/Dialect/D2M/Transforms/GenericTileComputeLoops.cpp`
- May need to handle reduction iterator types specially

### Option 2: Implement Reduction with Math (1 hour)

**Workaround approach**:
- Approximate rowmax with element-wise operations
- Example: use existing ops to compute max iteratively
- Won't be as efficient but may work for demonstration

### Option 3: Try Different Operations (30 min)

**Add other ops that might work**:
- `add()` - already have `__add__` but not as method
- Scalar operations if supported
- Other unary SFPU ops

### Option 4: Focus on Other FA Components (2-3 hours)

**Skip reductions for now, work on**:
- Fill/initialization operations
- Loop structure (for kv_blocks)
- Second kernel for P @ V (workaround for multiple matmuls)

## Recommendation

Given that:
1. Reductions hit compiler limitations
2. User guidance: "hack your way through, iterate slowly"
3. Previous session made good progress on infrastructure

**Suggested approach**:
1. Ask user if they want to:
   - Debug reduction compiler crash (deeper dive)
   - Skip reductions and continue with other ops
   - Try mathematical approximation for rowmax

2. Focus on what works and build up incrementally

## Files Changed This Session

### tt-lang
- `python/ttlang/operators.py`: Added ternary helper and rowmax method
- `examples/flash_attention.py`: Added note about rowmax limitation

### tt-mlir
- No changes (existing from previous session)

## Test Files Created

- `test_rowmax.py`: Standalone test for rowmax (currently fails)

## Current Branches

- tt-mlir: `zoecarver/tt-lang-flash-attention`
- tt-lang: `zoecarver/tt-lang-flash-attention`

Uncommitted changes in both repos (as instructed).
