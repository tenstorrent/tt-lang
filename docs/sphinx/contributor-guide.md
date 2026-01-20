# Contributor Guide

## Workflow
- Branch from `main` and keep changes focused.
- Reconfigure after rebases: `cmake -GNinja -B build .`.
- Build and lint before review: `cmake --build build` and `pre-commit run --all-files`.

## Validation
- Compiler tests: `cmake --build build --target check-ttlang`.
- Simulator tests: `pytest test/sim`.
- Targeted MLIR coverage: `llvm-lit test/ttlang/<path>.mlir`.

## Documentation
- Add new user-facing pages under `docs/sphinx` and link them in `index.rst`.
- Keep contributor-only instructions in this guide or `guidelines.md`.
- Build docs with `cmake --build build --target ttlang-docs`.

## Adding Elementwise Operations

To add a new elementwise operation (unary or binary), update these files:

### 1. `include/ttlang/Dialect/TTL/TTLElementwiseOps.def`

Add an entry with the TTL op name, tile op name, and TTKernel init/compute op names:

```cpp
// Binary op (3-arg form: DST[odst] = op(DST[src0], DST[src1]))
TTL_BINARY_TILE_OP(NewOp, NewOpTileOp, NewOpBinaryTilesInitOp, NewOpBinaryTilesOp)

// Unary op (in-place form: DST[dst_idx] = op(DST[dst_idx]))
TTL_UNARY_TILE_OP(NewOp, NewOpTileOp, NewOpTileInitOp, NewOpTileOp)

// Min/Max binary op (2-arg in-place form)
TTL_BINARY_TILE_OP_MINMAX(NewOp, NewOpTileOp, NewOpTilesInitOp, NewOpTilesOp)
```

This automatically generates:
- C++ lowering patterns (`ConvertTTLToCompute.cpp`, `ConvertTTLTileOpsToTTKernel.cpp`)
- Python bindings (`_generated_elementwise.py`)

### 2. `include/ttlang/Dialect/TTL/IR/TTLOps.td`

Add the TableGen op definitions using the multiclass:

```tablegen
// Binary op
defm TTL_NewOp : TTL_BinaryElementwisePair<"newop", "newop_tiles">;

// Unary op
defm TTL_NewOp : TTL_UnaryElementwisePair<"newop", "newop_tile">;
```

### 3. Verify the TTKernel ops exist in tt-mlir

The TTKernel init and compute ops must exist in `tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td`. If they don't, they need to be added to tt-mlir first.

### 4. Add tests

- Add a lit test in `test/ttlang/Dialect/TTL/Transforms/` for the lowering
- Add to `test/python/test_elementwise_ops.py` for end-to-end verification
