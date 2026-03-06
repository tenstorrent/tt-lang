# Build Integration

tt-lang builds LLVM, tt-metal, and tt-mlir from git submodules under `third-party/`. Choose one of these setups.

## Build from submodules (default)
```bash
cmake -GNinja -B build .
source build/env/activate
cmake --build build
```
This builds LLVM/MLIR from `third-party/llvm-project` and installs to `build/llvm-install/`. tt-metal builds to `third-party/tt-metal/build/`. tt-mlir dialects compile inline.

## Pre-built MLIR installation
```bash
cmake -GNinja -B build . -DMLIR_PREFIX=/path/to/llvm-install
source build/env/activate
cmake --build build
```

## Using the ttmlir toolchain
```bash
cmake -GNinja -B build . -DTTLANG_USE_TTMLIR_TOOLCHAIN=ON
source build/env/activate
cmake --build build
```
This uses a pre-built LLVM from `$TTMLIR_TOOLCHAIN_DIR` (default: `/opt/ttmlir-toolchain`).

## Common options
- `-DCMAKE_BUILD_TYPE=Debug` for developer iteration.
- `-DTTLANG_ENABLE_BINDINGS_PYTHON=ON` to build Python bindings.
- `-DLLVM_INSTALL_DIR=/custom/path` to set the LLVM install location when building from submodule.
