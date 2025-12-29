# Testing

## Compiler tests
- Full suite: `cmake --build build --target check-ttlang`.
- Single MLIR file: `llvm-lit test/ttlang/<path>.mlir`.

## Python tests
- Runtime simulation: `pytest test/sim`.
- Python API: `pytest test/python`.
- Set `SYSTEM_DESC_PATH` and `ttrt query --save-artifacts` when running runtime-backed tests.

## Build hygiene
- Format and lint before review: `pre-commit run --all-files`.
- Reconfigure if dependencies change: rerun `cmake -GNinja -B build .` after pulling toolchain updates.
