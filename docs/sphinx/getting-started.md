# Getting Started

## Prerequisites
- CMake 3.28+, Ninja, and Clang 18+ or GCC 11+.
- tt-mlir toolchain at `TTMLIR_TOOLCHAIN_DIR` (default `/opt/ttmlir-toolchain`).
- Python 3.11+ (toolchain venv).

## Configure
```bash
cmake -G Ninja -B build \
  -DTTMLIR_BUILD_DIR=/path/to/tt-mlir/build \
  -DTTLANG_ENABLE_DOCS=ON
```

## Build
```bash
source build/env/activate
cmake --build build
cmake --build build --target ttlang-docs
```

## View docs
Open `build/docs/sphinx/_build/html/index.html` or serve locally:
```bash
python -m http.server 8000 -d build/docs/sphinx/_build/html
```

## Quick checks
- Compiler tests: `cmake --build build --target check-ttlang`.
- Single MLIR test: `llvm-lit test/ttlang/Dialect/TTL/IR/ops.mlir`.
- Simulator smoke: `pytest test/sim -q`.
