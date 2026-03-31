# Build System

## Overview

tt-lang uses a CMake-based build system that compiles LLVM/MLIR, a minimal
tt-mlir subset, tt-metal, and tt-lang's own dialects and tools from pinned git
submodules. A single `cmake -G Ninja -B build && cmake --build build` invocation
produces a fully working environment.

## Prerequisites

- CMake 3.28+
- Ninja
- Clang/Clang++ 17+ (or GCC 11+)
- Python 3.11+
- Git (submodules must be initialized:
  `git submodule update --init --recursive`)

## Build Modes

### Build from submodules (default)

```bash
cmake -G Ninja -B build
source build/env/activate
cmake --build build
```

Builds LLVM/MLIR from `third-party/llvm-project` and installs to
`build/llvm-install/`. tt-metal builds to `third-party/tt-metal/build/`. tt-mlir
dialects compile inline. The result is cached — subsequent configures skip the
LLVM build if `build/llvm-install/lib/cmake/mlir/MLIRConfig.cmake` already
exists.

### Build with reusable toolchain

```bash
cmake -G Ninja -B build -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain
source build/env/activate
cmake --build build
```

Same as above, but LLVM, tt-metal, and the Python venv are installed into the
given prefix so they can be reused by other builds.

### Use a pre-built toolchain

```bash
cmake -G Ninja -B build -DTTLANG_USE_TOOLCHAIN=ON
source build/env/activate
cmake --build build
```

Skips the LLVM and tt-metal builds entirely. Uses a pre-built toolchain at
`$TTLANG_TOOLCHAIN_DIR` (default: `/opt/ttlang-toolchain`). The build sets
`Python3_EXECUTABLE` to the toolchain's venv so that MLIR Python bindings
resolve against the same interpreter they were built with.

### Pre-built MLIR installation

```bash
cmake -G Ninja -B build -DMLIR_PREFIX=/path/to/llvm-install
source build/env/activate
cmake --build build
```

Point directly at an LLVM/MLIR install prefix. tt-metal still builds from
submodule. tt-lang may not build successfully if the pre-built LLVM is a
significantly different version than what tt-mlir expects.

## Installing

Installation is used to create self-contained distribution packages (e.g.,
Docker images). It is not needed for development — just use
`source build/env/activate` after building to get a fully working environment.

```bash
cmake --install build --prefix /opt/ttlang-toolchain
```

This copies tt-lang binaries, Python packages, examples, tests, and the
environment activation script into the given prefix. When `TTLANG_TOOLCHAIN_DIR`
was set during configure, LLVM, tt-metal, and the Python venv are already there;
the install step adds only tt-lang's own artifacts.

## Building Documentation

```bash
cmake -G Ninja -B build -DTTLANG_ENABLE_DOCS=ON
cmake --build build --target ttlang-docs
python -m http.server 8000 -d build/docs/sphinx/_build/html
```

Open `http://localhost:8000` to browse the docs locally.

## Submodules

`.gitmodules` pins three submodules:

| Submodule | Purpose |
|---|---|
| `third-party/llvm-project` | LLVM/MLIR source (built at configure time) |
| `third-party/tt-mlir` | tt-mlir source (only select directories compiled) |
| `third-party/tt-metal` | Runtime (built at configure time) |

### LLVM SHA verification

When using a pre-built LLVM (via `MLIR_PREFIX` or `TTLANG_USE_TOOLCHAIN`), the
build verifies the installed LLVM was built from the expected commit. The
expected SHA is read from `third-party/tt-mlir/env/CMakeLists.txt`
(`LLVM_PROJECT_VERSION`), and the actual SHA is read from
`<prefix>/include/llvm/Support/VCSRevision.h`. On mismatch, cmake emits a
`FATAL_ERROR`. Pass `-DTTLANG_ACCEPT_LLVM_MISMATCH=ON` to proceed despite the
mismatch.

## CMake Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | Build type (Debug, Release, RelWithDebInfo) |
| `LLVM_BUILD_TYPE` | `Release` | LLVM build type (independent of project build type) |
| `TTLANG_TOOLCHAIN_DIR` | — | Toolchain prefix for LLVM, tt-metal, and venv |
| `TTLANG_USE_TOOLCHAIN` | `OFF` | Use pre-built toolchain at `TTLANG_TOOLCHAIN_DIR` |
| `MLIR_PREFIX` | — | Path to pre-built LLVM/MLIR install |
| `TTLANG_ACCEPT_LLVM_MISMATCH` | `OFF` | Allow LLVM SHA mismatch with pre-built installs |
| `TTLANG_ACCEPT_TTMETAL_MISMATCH` | `OFF` | Allow tt-metal SHA mismatch with pre-built installs |
| `TTLANG_ENABLE_PERF_TRACE` | `ON` | Enable tt-metal performance tracing support |
| `TTLANG_SIM_ONLY` | `OFF` | Set up Python environment for [simulator](simulator.md) only; skip compiler build |
| `TTLANG_ENABLE_DOCS` | `OFF` | Enable Sphinx documentation build (`ttlang-docs` target) |
| `CODE_COVERAGE` | `OFF` | Enable code coverage reporting |
| `TTLANG_FORCE_TOOLCHAIN_REBUILD` | `OFF` | Force rebuild of LLVM and tt-metal into `TTLANG_TOOLCHAIN_DIR` |

## Build Architecture

### Minimal tt-mlir subset

`cmake/modules/BuildTTMLIRMinimal.cmake` and `lib/ttmlir-minimal/` compile
tt-mlir sources directly from the submodule, producing 7 CMake targets:
`MLIRTTCoreDialect`, `MLIRTTTransforms`, `MLIRTTMetalDialect`,
`MLIRTTKernelDialect`, `MLIRTTKernelTransforms`, `TTMLIRTTKernelToEmitC`, and
`TTKernelTargetCpp`. Flatbuffers stub headers are generated in
`build/include/ttmlir/Target/Common/` to satisfy compile-time references without
requiring a flatc build.

### tt-metal runtime

`cmake/modules/BuildTTMetal.cmake` builds tt-metal at configure time via
`execute_process`. Post-build, `_ttnn.so` and `_ttnncpp.so` are copied so
`import ttnn` works after activating the environment.

### Python bindings

`python/ttmlir/` contains a nanobind extension (`_ttmlir`) with TTCore,
TTKernel, and TTMetal dialect bindings. A CAPI aggregation library
(`libTTLangPythonCAPI.so`) embeds upstream MLIR + tt-mlir + ttlang C API into a
single shared object. The Python package prefix is `ttl.`.

Three-stage site initialization registers all dialects on context creation:
1. `_mlirRegisterEverything` — upstream MLIR dialects (func, arith, scf, etc.)
2. `_site_initialize_0.py` — tt-mlir dialects (TTCore, TTKernel, TTMetal)
3. `_site_initialize_1.py` — TTL dialect

### Environment

`env/activate.in` is a configure-time template that produces
`build/env/activate`. Sourcing it activates the Python venv, sets `TT_LANG_HOME`
and `TTLANG_ENV_ACTIVATED=1`, prepends `build/bin` to `PATH`, prepends
`build/python_packages` and `python/` to `PYTHONPATH`, and sets
`LD_LIBRARY_PATH` for tt-metal libs.

## Troubleshooting

### LLVM build takes too long

The first submodule build compiles LLVM from source, which can take 30-60
minutes. Ensure ccache is installed (automatically detected), or use a pre-built
LLVM via `-DMLIR_PREFIX` or `-DTTLANG_USE_TOOLCHAIN=ON`. Subsequent configures
skip the build if `llvm-install/` already exists.

### LLVM SHA mismatch

If using a pre-built LLVM and cmake reports a SHA mismatch, the installed LLVM
was built from a different commit than what tt-mlir expects. Either rebuild LLVM
from the correct commit or pass `-DTTLANG_ACCEPT_LLVM_MISMATCH=ON` to proceed at
your own risk.

### Python import errors

Ensure the environment is activated and the build completed:
```bash
source build/env/activate
python3 -c "from ttl.dialects import ttl, ttkernel, ttcore"
```

### Missing submodules

```bash
git submodule update --init --recursive
```

For tt-metal specifically, nested submodules (tracy, tt_llk, umd) must also be
initialized. The build emits clear error messages if they are missing.
