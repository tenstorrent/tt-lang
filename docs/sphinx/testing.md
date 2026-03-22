# Testing

For detailed guidance on writing new tests (lit test structure, FileCheck patterns, test output locations), see [`test/TESTING.md`](https://github.com/tenstorrent/tt-lang/blob/main/test/TESTING.md).

## Compiler tests
- Full compiler suite: `ninja -C build check-ttlang-all` (MLIR, bindings, end-to-end, pytest, Python lit).
- Quick check (MLIR + bindings only): `ninja -C build check-ttlang`.
- Single MLIR file: `llvm-lit test/ttlang/<path>.mlir`.

## Simulator tests
- `python -m pytest test/sim` — not included in `check-ttlang-all`, must be run separately.

## Python tests
- Python API: `python -m pytest test/python`.

## Build hygiene
- Format and lint before review: `pre-commit run --all-files`.
- Reconfigure if dependencies change: rerun `cmake -G Ninja -B build` after pulling toolchain updates.
