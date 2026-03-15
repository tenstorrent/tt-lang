# Getting Started

## Prerequisites
- CMake 3.28+, Ninja, and Clang 17+ or GCC 11+.
- Python 3.11+.
- For faster builds: a pre-built toolchain at `TTLANG_TOOLCHAIN_DIR` (default `/opt/ttlang-toolchain`). Without one, LLVM and tt-metal build from submodules on first configure.

## Configure and build

### With pre-built toolchain (fast)
```bash
cmake -G Ninja -B build -DTTLANG_USE_TOOLCHAIN=ON
source build/env/activate
cmake --build build
```

### From submodules (no prerequisites beyond system packages)
```bash
cmake -G Ninja -B build
source build/env/activate
cmake --build build
```

## Quick checks
- All tests: `ninja -C build check-ttlang-all`
- Compiler tests: `ninja -C build check-ttlang-mlir`
- Single MLIR test: `llvm-lit test/ttlang/Dialect/TTL/IR/ops.mlir`
- Simulator smoke: `pytest test/sim -q`

## Docker testing

Run tests inside a Docker container using a local build (no rebuild required):
```bash
scripts/docker-test.sh all
scripts/docker-test.sh mlir
scripts/docker-test.sh -- pytest test/me2e/ -k some_test
```

## Documentation
```bash
cmake -G Ninja -B build -DTTLANG_ENABLE_DOCS=ON
cmake --build build --target ttlang-docs
python -m http.server 8000 -d build/docs/sphinx/_build/html
```
Open http://localhost:8000/index.html.
