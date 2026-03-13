# Build Integration

tt-lang builds LLVM, tt-metal, and tt-mlir from git submodules under `third-party/`. Choose one of these setups.

## Build from submodules (default)
```bash
cmake -GNinja -B build .
source build/env/activate
cmake --build build
```
This builds LLVM/MLIR from `third-party/llvm-project` and installs to `build/llvm-install/`. tt-metal builds to `third-party/tt-metal/build/`. tt-mlir dialects compile inline.

## Build with reusable toolchain
```bash
cmake -GNinja -B build -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain .
source build/env/activate
cmake --build build
```
Same as above, but LLVM, tt-metal, and the Python venv are installed into the given prefix so they can be reused by other builds.

## Use a pre-built toolchain
```bash
cmake -GNinja -B build -DTTLANG_USE_TOOLCHAIN=ON .
source build/env/activate
cmake --build build
```
Skips the LLVM and tt-metal builds entirely. Uses a pre-built toolchain at `$TTLANG_TOOLCHAIN_DIR` (default: `/opt/ttlang-toolchain`).

## Pre-built MLIR installation
```bash
cmake -GNinja -B build -DMLIR_PREFIX=/path/to/llvm-install .
source build/env/activate
cmake --build build
```
Point directly at an LLVM/MLIR install prefix. tt-metal still builds from submodule.

## Common options
- `-DCMAKE_BUILD_TYPE=Debug` for developer iteration.
- `-DTTLANG_FORCE_TOOLCHAIN_REBUILD=ON` to force rebuild into an existing toolchain prefix.

See [`docs/BUILD_SYSTEM.md`](../BUILD_SYSTEM.md) for full details on all CMake options.
