# Tools

- `cmake --build build` builds the compiler and Python bindings.
- `ninja -C build check-ttlang` runs compiler MLIR tests and Python binding tests.
- `ninja -C build check-ttlang-all` runs the full compiler test suite (MLIR, bindings, end-to-end, pytest, Python lit). Does not include simulator tests.
- `llvm-lit` executes individual MLIR tests.
- `python -m pytest test/sim` exercises the simulator flows (not included in `check-ttlang-all`).
- `pre-commit run --all-files` formats and enforces style.
- `cmake --build build --target ttlang-docs` builds the Sphinx HTML docs.
