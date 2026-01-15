# DST Register Allocation

## Overview

DST (destination) registers are hardware registers used for tile computations. This pass (`TTLTileAndAssignDST.cpp`) assigns DST indices to tile operations based on whether they are unary or binary operations.

## Core Principle: Unary vs Binary Operations

**Binary operations** (e.g., `add`, `mul`, `max`) require separate DST slots for inputs and output:
- Each input gets its own DST slot
- Output gets a fresh DST slot
- `inputs_footprint = number of tile inputs`

**Unary operations** (e.g., `exp`, `abs`, `neg`) operate in-place:
- Output reuses the input's DST slot
- No additional DST allocation needed
- `inputs_footprint = 0` for standalone unary ops

## DST Allocation Algorithm

The algorithm consists of three phases.

Phase 1. Estimate the input DST footprint based on liveness analysis.
for each binary tile_compute_op:
  for each input:
    free_reg = find_first_unset()
    dstIndexForValue[input] = free_reg
    if there are any users of the input value, set the inUse bit for the register

max_unary_fanout = find the maximum number of unary ops that consume the same value
inputs_footprint = max(dstIndexForValue.values()) + 1 + (max_unary_fanout - 1)

Phase 2. Backward pass: Assign DST indices to *outputs*.

Available output dst registers start at base_out_dst_index = max(dstIndexForValue.values()) + 1.
for each output of the compute operation (out block argument), assign the next available output dst register:
  dstIndexForOutput[output] = base_out_dst_index + i
  i++
Go backwards through the operations in the compute operation:
  if (is_unary and the result is yielded to an output block argument):
    # Both the input and output use the dst register assigned to that block argument
  else:
    # Binary: allocate fresh DST for output
    if the result is consumed only by unary ops, then assign the result DST register
    to be the input/output dst register of the first user unary op.

Phase 3. Forward pass: Assign DST indices to all operands that have not yet been assigned, 
reuse based on liveness analysis, do not overwrite any already assigned dst registers.

Do the original DST assignment pass from origin/main, except don't overwrite any already assigned dst registers.


## Pipeline Integration

The DST allocation pass runs in this order:

1. `ttl-tile-and-assign-dst`: Assigns DST indices, adds `ttl.unroll_factor` attribute
2. `ttl-lower-to-loops`: Converts `ttl.compute` to `scf.for` loops
3. `ttl-unroll-compute-loops`: Unrolls loops (optional, controlled by `--enable-unroll`)
4. `ttl-insert-tile-regs-sync`: Inserts DST lifecycle ops (acquire/commit/wait/release)

## Generated Code Example (2x2 block with binary op)

```cpp
tile_regs_acquire();
for (i = 0..2) {
  for (j = 0..2) {
    copy_tile(CB0, i*2+j, DST[0]);
    copy_tile(CB1, i*2+j, DST[1]);
    mul_binary_tile(DST[0], DST[1], DST[2+i*2+j]);  // Dynamic output index
  }
}
tile_regs_commit();
tile_regs_wait();
for (i = 0..2) {
  for (j = 0..2) {
    pack_tile(DST[2+i*2+j], CB_out, i*2+j);
  }
}
tile_regs_release();
```

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
* Register spilling for complex operation chains that exceed DST capacity.
