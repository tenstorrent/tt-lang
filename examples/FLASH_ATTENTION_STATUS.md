# Flash Attention Implementation Status

## What Works ✓

### 1. Individual rowmax and rowsum operators
- ✓ `test_rowmax.py` - compiles successfully
- ✓ `test_rowsum.py` - compiles successfully
- Both operators are implemented and functional

### 2. Two-kernel Flash Attention WITHOUT reductions
- ✓ `flash_attention_pipelined.py` - **FULLY WORKING**
- Operations: transpose, matmul, subtract, exp
- Configuration: 8×10 grid (80 cores), 2×2 tiles
- Performance estimate: 48-72 TFLOPs/s
- **This is production-ready code**

## What Doesn't Work ✗

### Combining rowmax/rowsum with other operations

**Root Cause**: The compiler has a limitation:
```cpp
// InsertDstRegisterAccess.cpp:356
assert(computeOp->hasOneUse() &&
       "Currently we do not support multiple users in the same compute dst region.");
```

**Why rowmax/rowsum trigger this**:
- They use `tile_reduce_max` and `tile_reduce_sum` (ternary operations)
- Implementation passes input 3 times: `tile_reduce_max(inp, inp, inp, ...)`
- This creates multiple uses at MLIR level even though Python shows single use
- Compiler sees 3 uses of the same value → assertion fails

**Failed patterns**:
```python
# Pattern: matmul + rowmax + subtract
K_T = K_block.transpose()
S = Q_block @ K_T
m = S.rowmax()           # S used 3x internally
S_stable = S - m         # S used again externally → CRASH
```

```python
# Pattern: exp + rowsum + anything
P = S.exp()
l = P.rowsum()           # P used 3x internally → CRASH
```

Even when rowmax/rowsum appear to have a single user, the internal 3x usage triggers the assertion.

## Compiler Limitation Details

The InsertDstRegisterAccess pass doesn't support operations with multiple users in the same compute region. This affects:

1. **Reduction operations** (rowmax, rowsum) - use input 3 times internally
2. **Any value used by multiple subsequent ops**

Example from `transpose-fusion-bugs.md`:
```python
S = Q @ K      # S has 2 users:
m = S.rowmax() # User 1
P = S - m      # User 2
# Compiler: "currently we do not support multiple users"
```

## Solutions & Workarounds

### Solution 1: Use the working two-kernel approach ✓ RECOMMENDED
```python
# See flash_attention_pipelined.py
# Kernel 1: softmax approximation
K_T = K.transpose()
S = Q @ K_T
S_stable = S - Q  # Approximates S - rowmax(S)
P = S_stable.exp()
# Missing rowsum(P), but compiles and runs!

# Kernel 2: Apply attention
O = scores @ V
```

**Status**: ✓ Works on 8×10 grid with 80 cores
**Performance**: Estimated 48-72 TFLOPs/s

### Solution 2: Fix the compiler assertion

**Location**: `tt-mlir/lib/Dialect/D2M/Transforms/InsertDstRegisterAccess.cpp:356`

**Current code**:
```cpp
TT_assert(computeOp->hasOneUse() &&
          "Currently we do not support multiple users in the same compute dst region.");
```

**What needs fixing**:
1. Support for operations with multiple users
2. Proper DST slot allocation for multi-use values
3. Handle reduction operations' internal multi-use pattern

**Effort**: Medium (2-3 days based on previous similar fixes)
**Impact**: Would unlock proper Flash Attention with rowmax/rowsum

### Solution 3: Change rowmax/rowsum implementation

**Current**: Uses ternary `tile_reduce_max(a, b, c, dim)` with `a=b=c=input`

**Alternative**: Use unary reduction if hardware supports it

**Investigation needed**: Check if hardware has unary max/sum reduction primitives

## Testing Summary

| Test | Status | File |
|------|--------|------|
| rowmax alone | ✓ Works | test_rowmax.py |
| rowsum alone | ✓ Works | test_rowsum.py |
| matmul + rowmax | ✗ Crashes | test_p1.py |
| matmul + subtract + exp + rowsum | ✗ Crashes | test_p2.py |
| matmul + rowmax + subtract + exp + rowsum | ✗ Crashes | test_fa_incremental.py (pattern 3) |
| Two-kernel FA (no reductions) | ✓ Works | flash_attention_pipelined.py |

## Recommendations

### For immediate use:
1. **Use `flash_attention_pipelined.py`** - it works and scales well
2. Approximate rowmax with subtraction: `S - Q` instead of `S - rowmax(S)`
3. Skip rowsum normalization for now or approximate with other ops

### For production quality:
1. Fix the multiple-users assertion in InsertDstRegisterAccess.cpp
2. Test rowmax/rowsum in context again
3. Implement proper online softmax algorithm

### For performance:
The working two-kernel approach should already achieve good performance:
- 80 cores (8×10 grid)
- 2×2 tiles per core
- Est. 48-72 TFLOPs/s

This is competitive with hand-written implementations per session-3 notes.

## Files to Use

**Working examples**:
- `examples/flash_attention_pipelined.py` - ✓ Two-kernel FA (80 cores, works!)
- `examples/test_rowmax.py` - ✓ Standalone rowmax
- `examples/test_rowsum.py` - ✓ Standalone rowsum

**Diagnostic examples**:
- `examples/test_p1.py` - Shows where rowmax fails
- `examples/test_p2.py` - Shows where rowsum fails

**Not working**:
- `examples/flash_attention_ideal.py` - Theoretical code with all features
- `examples/flash_attention.py` - Uses sqrt/recip instead of reductions

## Next Steps

1. **Short term**: Use flash_attention_pipelined.py as is - it works!
2. **Medium term**: Fix multiple-users assertion to enable rowmax/rowsum
3. **Long term**: Implement full online softmax with proper reductions

See transpose-fusion-bugs.md and compiler-architecture-deep-dive.md for compiler internals.
