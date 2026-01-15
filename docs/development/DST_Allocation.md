# DST Register Allocation

## Overview

Multi-tile blocks now use dynamic DST indices to prevent tile result overwrites. Previously all tiles computed into fixed DST slots (e.g., DST[2]), causing only the last tile's result to survive.

## DST for separate compute and pack_tile loops

### 1. Two-Loop Structure (`ConvertTTLComputeToSCF.cpp`)

**Before**: Single loop with compute + pack interleaved per tile.

**After**: Separate loops matching tt-metal pattern:
```
tile_regs_acquire
Loop 1: compute all tiles → `DST[inputs_footprint + tile_idx]`
tile_regs_commit / tile_regs_wait
Loop 2: pack all tiles from `DST[inputs_footprint + tile_idx]` → CB
tile_regs_release
```

### 2. Dynamic DST Index Computation (`ConvertTTLTileOpsToTTKernel.cpp`)

**Before**: `loops.back()` checked only outermost loop for `dst_footprint`.

**After**: Search all enclosing loops for `dst_footprint` attribute. Only tile loops (not outer block loops) contribute to linearization.

Formula: `DST_index = inputs_footprint + (i * cols + j)` for 2x2 block → DST[2,3,4,5]

### 3. Pack DST Index (`ConvertTTLToTTKernel.cpp`)

**Before**: `loops.back()` missed `dst_footprint` when block loops existed above tile loops.

**After**: Search all loops for `dst_footprint`, compute `inputs_footprint + cbTileIndex`.

### 4. DST Footprint Attribute (`TTLTileAndAssignDST.cpp`)

Sets `ttl.dst_footprint` on ComputeOp, propagated to outermost tile loop during lowering.

**Footprint Computation:**

```cpp
// Phase 1: Allocate DST slots only for inputs that need separate registers
// - Binary op inputs: need their own DST slots
// - Unary-only inputs: share DST with output, don't count
// - Track maxInputDstIndex across all actually allocated inputs

// inputs_footprint = maxInputDstIndex + 1 (or 0 if no inputs need slots)
std::uint32_t inputs_footprint = hasInputs ? (maxInputDstIndex + 1) : 0;
```

For a binary op like `mul(a, b)`:
- `a` → DST[0], `b` → DST[1] (both need separate slots)
- `inputs_footprint = 2`
- Outputs start at DST[2]: `result[i,j]` → `DST[2 + i*cols + j]`

For a unary op like `exp(a)`:
- `a` shares DST with output (no separate slot needed)
- `inputs_footprint = 0`
- Outputs start at DST[0]: `result[i,j]` → `DST[i*cols + j]`

Note that in the case of pack_tile ops in the same loops
as compute ops, the `dst_footprint` would be simply 
`inputs_footprint + numbuer_of_outputs`.

## Generated Code Example (2x2 block)

```cpp
tile_regs_acquire();
for (i = 0..2) {
  for (j = 0..2) {
    copy_tile(CB0, i*2+j, DST[0]);
    copy_tile(CB1, i*2+j, DST[1]);
    mul_binary_tile(DST[0], DST[1], DST[2+i*2+j]);  // Dynamic!
    copy_tile(CB2, i*2+j, DST[0]);
    add_binary_tile(DST[2+i*2+j], DST[0], DST[2+i*2+j]);
  }
}
tile_regs_commit(); 
tile_regs_wait();
for (i = 0..2) {
  for (j = 0..2) {
    pack_tile(DST[2+i*2+j], CB3, i*2+j);  // Dynamic!
  }
}
tile_regs_release();
```

## Future Work

* Pack multiple contiguous tiles in a single `pack_tile` call. Requires analysis to determine when output tiles are contiguous in DST (e.g., `DST[2,3,4,5]` for a 2x2 block with row-major layout). Currently each tile is packed individually.
