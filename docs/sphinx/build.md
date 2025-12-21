# Build Integration

tt-lang reuses tt-mlir for dialects and runtime support. Choose one of these setups.

## Installed tt-mlir (recommended)
```bash
cmake -GNinja -B build . -DTTMLIR_DIR=/opt/ttmlir-toolchain/lib/cmake/ttmlir
source build/env/activate
cmake --build build
```

## Using a tt-mlir build tree
```bash
cmake -GNinja -B build . -DTTMLIR_BUILD_DIR=/path/to/tt-mlir/build
source build/env/activate
cmake --build build
```

## Fetch and build tt-mlir automatically
```bash
cmake -GNinja -B build .
source build/env/activate
cmake --build build
```
This downloads the commit pinned in `third-party/tt-mlir.commit` and installs it under `build/tt-mlir-install/`.

## Common options
- `-DCMAKE_BUILD_TYPE=Debug` for developer iteration.
- `-DTTLANG_ENABLE_BINDINGS_PYTHON=ON` to build Python bindings.
- `-DTTLANG_ENABLE_RUNTIME=OFF` to skip hardware runtime when sim-only.
