# Tools

- `cmake --build build` builds the compiler and Python bindings.
- `cmake --build build --target check-ttlang` runs the compiler regression suite.
- `llvm-lit` executes individual MLIR tests.
- `pytest test/sim` exercises the simulator flows.
- `pre-commit run --all-files` formats and enforces style.
- `cmake --build build --target ttlang-docs` builds the Sphinx HTML docs.
