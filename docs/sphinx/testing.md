# Testing

## Compiler tests
- Full suite: `cmake --build build --target check-ttlang-all`.
- Single MLIR file: `llvm-lit test/ttlang/<path>.mlir`.

## Python tests
- Runtime simulation: `pytest test/sim`.
- Python API: `pytest test/python`.

## Build hygiene
- Format and lint before review: `pre-commit run --all-files`.
- Reconfigure if dependencies change: rerun `cmake -GNinja -B build .` after pulling toolchain updates.
