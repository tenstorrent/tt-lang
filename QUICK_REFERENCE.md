# Quick Reference - Flash Attention Implementation

## What Works RIGHT NOW ✅

### Run the Demo
```bash
cd /Users/zcarver/Developer/tt-lang
source env/activate
export SYSTEM_DESC_PATH=/Users/zcarver/Downloads/system_desc.ttsys
python examples/flash_attention.py
```

**Output:**
```
=== Test 1: 1x1 grid, single KV block ===
✓ Single-block FA compiled!
=== Test 2: 2x2 grid, 1x1 tiles/core ===
✓ 2x2 grid compiled successfully!
```

### Available Operations (9)

**Unary**: transpose, exp, sqrt, recip, rowmax, rowsum
**Binary**: -, *, /
**Matrix**: @ (one per kernel)

### 6-Operation Kernel Example
```python
K_T = K_block.transpose()      # 1
S = Q_block @ K_T              # 2
S_stable = S - Q_block         # 3
P = S_stable.exp()             # 4
P_norm = P.sqrt()              # 5
result = P_norm.recip()        # 6
```

### Reduction Example (Standalone)
```python
# Works in separate kernel
m = S.rowmax()    # Max across columns (within tile)
l = P.rowsum()    # Sum across columns (within tile)
```

## Known Limitations ⚠️

1. **6-operation maximum** - 7+ ops crash
2. **1 matmul per kernel** - Need MemRefProvenanceAnalysis for multiple
3. **Reductions work but count toward 6-op limit**
4. **1 tile per core** - Multi-tile needs subblocking

## Quick Fixes Applied

### From Previous Session (Still Active)
- Bug #1-11: All fusion bugs fixed
- Same-input binary ops (S - S, S * S)
- Mixed 2D/3D iteration spaces

### From Today
- **Bug #12**: d2m.iter_index for reductions ✅ FIXED
- rowmax/rowsum operations ✅ WORKING
- Type converter null-safe ✅ IMPROVED
- getCB temp alloc tracing ✅ ENHANCED

## Files Changed (Uncommitted)

### tt-mlir
- `lib/Dialect/D2M/Transforms/InsertDstRegisterAccess.cpp`
- `lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp`
- `lib/Conversion/D2MToTTKernel/D2MToTTKernelPass.cpp`

### tt-lang
- `python/ttlang/operators.py`
- `examples/flash_attention.py`

## Test Status
- ✅ All 9 lit tests passing
- ✅ FA example compiles (6 ops)
- ✅ Reductions work standalone
- ❌ 7+ ops fail
- ❌ Multiple matmuls fail

## Build Commands

### Build tt-mlir (if changes made)
```bash
cd /Users/zcarver/Developer/tt-mlir
source env/activate
cmake --build build
```

### Run Tests
```bash
cd /Users/zcarver/Developer/tt-lang
source env/activate
export SYSTEM_DESC_PATH=/Users/zcarver/Downloads/system_desc.ttsys
llvm-lit -sv test/python/
```

## What to Try Next

1. **Larger grids**: Test 4x4, 8x8, 8x10
2. **Two-kernel FA**: Build complete algorithm across kernels
3. **Fill operation**: Add initialization support
4. **Loop structure**: Test KV block iteration

## Documentation Files

**Today's Work:**
- TODAYS_SESSION_SUMMARY.md
- FINAL_SESSION_SUMMARY.md
- REDUCTION_OPS_IMPLEMENTATION.md
- QUICK_REFERENCE.md (this file)

**Previous Session:**
- FA_IMPLEMENTATION_README.md
- FLASH_ATTENTION_RESULTS.md
- SESSION_SUMMARY.md
- MATMUL_FUSION_ISSUE.md
- PATH_TO_PRODUCTION_FA.md
- MULTI_TILE_STRATEGY.md

**Code Docs:**
- transpose-fusion-bugs.md (in tt-mlir-local/)
- compiler-architecture-deep-dive.md (in tt-mlir-local/)
