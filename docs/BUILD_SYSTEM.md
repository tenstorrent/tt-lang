# tt-lang Build System Documentation

## Overview

tt-lang uses a CMake-based build system that compiles LLVM/MLIR, a minimal
tt-mlir subset, tt-metal, and tt-lang's own dialects and tools from pinned git
submodules. A single `cmake -B build && cmake --build build` invocation produces
a fully working environment.

The build system supports:
- Building LLVM/MLIR from submodule (default) or using a pre-built install
- TableGen and C++ code for MLIR dialects and passes
- Python bindings using nanobind
- tt-metal runtime (configure-time build)

## Prerequisites

- CMake 3.28+
- Ninja
- Clang/Clang++ (or GCC)
- Python 3.11+
- Git (submodules must be initialized)

## Quick Start

```bash
cd tt-lang

# Initialize submodules
git submodule update --init --recursive

# Configure and build
cmake -G Ninja -B build .
cmake --build build

# Activate environment
source build/env/activate
```

## Installing

After building, install tt-lang and its toolchain into a prefix directory:

```bash
cmake --install build --prefix /opt/ttlang-toolchain
```

When `TTLANG_TOOLCHAIN_DIR` is set during configure, LLVM, tt-metal, and the Python venv are already placed there. The install step adds tt-lang's own artifacts (binaries, Python packages, examples, tests, and the environment activation script).

The resulting prefix is self-contained and can be used to build Docker images or shared across machines (the venv will be recreated on first configure).

## Submodules

`.gitmodules` pins three submodules:

| Submodule | Purpose |
|---|---|
| `third-party/llvm-project` | LLVM/MLIR source (built at configure time) |
| `third-party/tt-mlir` | tt-mlir source (only select directories compiled) |
| `third-party/tt-metal` | Runtime (built at configure time) |

## LLVM/MLIR

`cmake/modules/BuildLLVM.cmake` supports two modes, controlled by whether
`MLIR_PREFIX` or `TTLANG_USE_TOOLCHAIN` is set.

### Option A: Build from submodule (default)

Builds LLVM from `third-party/llvm-project` at configure time. Creates a Python
venv in `build/venv/` with MLIR Python bindings and all tt-lang pip requirements.
The result is cached — subsequent configures skip the LLVM build if
`build/llvm-install/lib/cmake/mlir/MLIRConfig.cmake` already exists.

```bash
cmake -G Ninja -B build .
```

To install the toolchain (LLVM, tt-metal, Python venv) into a reusable prefix,
set `TTLANG_TOOLCHAIN_DIR`:

```bash
cmake -G Ninja -B build -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain .
```

This redirects the LLVM install, tt-metal build, and Python venv into
`/opt/ttlang-toolchain` so it can be reused by other builds via Option B.

### Option B: Use a pre-built toolchain or LLVM/MLIR install

Skip the LLVM and tt-metal builds by pointing at an existing toolchain:

```bash
# Use a toolchain previously built with TTLANG_TOOLCHAIN_DIR
cmake -G Ninja -B build \
  -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain \
  -DTTLANG_USE_TOOLCHAIN=ON .
```

`TTLANG_TOOLCHAIN_DIR` can also be set via the environment variable of the
same name. When not specified with `TTLANG_USE_TOOLCHAIN=ON`, it defaults to
`/opt/ttlang-toolchain`.

When `TTLANG_USE_TOOLCHAIN=ON`, the build sets `Python3_EXECUTABLE` to the
toolchain's venv (`${TTLANG_TOOLCHAIN_DIR}/venv/bin/python`) so that MLIR
Python bindings resolve against the same interpreter they were built with.

Alternatively, point directly at an LLVM/MLIR install prefix:

```bash
cmake -G Ninja -B build -DMLIR_PREFIX=/path/to/llvm-install .
```

WARNING: tt-lang may not build successfully if the pre-built LLVM is a
significantly different version than what tt-mlir and tt-lang expect.

### LLVM SHA verification

When using a pre-built LLVM (either via `MLIR_PREFIX` or
`TTLANG_USE_TOOLCHAIN`), the build verifies the installed LLVM was built
from the expected commit. The expected SHA is read from
`third-party/tt-mlir/env/CMakeLists.txt` (`LLVM_PROJECT_VERSION`), and the
actual SHA is read from `<prefix>/include/llvm/Support/VCSRevision.h`.

On mismatch, cmake emits a `FATAL_ERROR` with both SHAs. To proceed despite the
mismatch:

```bash
cmake -G Ninja -B build -DMLIR_PREFIX=/path/to/llvm -DTTLANG_ACCEPT_LLVM_MISMATCH=ON .
```

The verification function (`ttlang_verify_llvm_sha`) lives in
`cmake/modules/TTLangUtils.cmake`. Submodule builds skip verification since git
already pins the correct commit.

## Minimal tt-mlir subset

`cmake/modules/BuildTTMLIRMinimal.cmake` and `lib/ttmlir-minimal/` compile
tt-mlir sources directly from the submodule, producing 7 CMake targets:

- `MLIRTTCoreDialect`, `MLIRTTTransforms`
- `MLIRTTMetalDialect`
- `MLIRTTKernelDialect`, `MLIRTTKernelTransforms`
- `TTMLIRTTKernelToEmitC`, `TTKernelTargetCpp`

Flatbuffers stub headers are generated in `build/include/ttmlir/Target/Common/`
to satisfy compile-time references without requiring a flatc build.

## tt-metal runtime

`cmake/modules/BuildTTMetal.cmake` builds tt-metal at configure time via
`execute_process`. Post-build, `_ttnn.so` and
`_ttnncpp.so` are copied so `import ttnn` works after activating the environment.

## Python bindings

`python/ttmlir/` contains a nanobind extension (`_ttmlir`) with TTCore,
TTKernel, and TTMetal dialect bindings. A CAPI aggregation library
(`libTTLangPythonCAPI.so`) embeds upstream MLIR + tt-mlir + ttlang C API into a
single shared object. The Python package prefix is `ttl.`.

Three-stage site initialization registers all dialects on context creation:
1. `_mlirRegisterEverything` — upstream MLIR dialects (func, arith, scf, etc.)
2. `_site_initialize_0.py` — tt-mlir dialects (TTCore, TTKernel, TTMetal)
3. `_site_initialize_1.py` — TTL dialect

## Environment

`env/activate.in` is a configure-time template that produces `build/env/activate`.
Sourcing it:
- Activates the Python venv
- Sets `TT_LANG_HOME`, `TTLANG_ENV_ACTIVATED=1`
- Prepends `build/bin` to `PATH`
- Prepends `build/python_packages` and `python/` to `PYTHONPATH`
- Sets `LD_LIBRARY_PATH` for tt-metal libs

## Directory Structure

<details>
<summary>Source tree</summary>

```
tt-lang/
├── cmake/modules/
│   ├── BuildLLVM.cmake              # LLVM build (submodule or pre-built)
│   ├── BuildTTMetal.cmake           # tt-metal configure-time build
│   ├── BuildTTMLIRMinimal.cmake     # Minimal tt-mlir subset (7 targets)
│   ├── GetVersionFromGit.cmake      # Version extraction from git tags
│   ├── TTLangCompilerSetup.cmake    # Compiler flags and settings
│   └── TTLangUtils.cmake            # Utility functions (SHA verification, etc.)
├── env/
│   └── activate.in                  # Configure-time template → build/env/activate
├── include/ttlang/
│   ├── Bindings/Python/             # Python binding headers
│   ├── Dialect/
│   │   ├── TTKernel/Transforms/     # TTKernel pass declarations
│   │   ├── TTL/
│   │   │   ├── IR/                  # TTL dialect ODS-generated headers
│   │   │   └── Pipelines/           # TTL pass pipeline registration
│   │   └── Utils/                   # Shared dialect utilities
│   └── ...
├── include/ttlang-c/                # C API headers (CAPI aggregation)
├── lib/
│   ├── CAPI/                        # C API implementation (libTTLangPythonCAPI.so)
│   ├── Dialect/
│   │   ├── TTKernel/Transforms/     # TTKernel pass implementations
│   │   └── TTL/
│   │       ├── IR/                  # TTL dialect implementation
│   │       ├── Pipelines/           # TTL pass pipeline
│   │       └── Transforms/          # TTL passes (DST assignment, etc.)
│   └── ttmlir-minimal/
│       ├── CAPI/                    # Dialect registration bridge (TTCore, TTKernel, TTMetal)
│       ├── TTKernelToCppRegistration.cpp  # Pass registration for --convert-ttkernel-to-emitc
│       └── CMakeLists.txt           # Builds 7 tt-mlir targets from submodule sources
├── python/
│   ├── pykernel/                    # @pykernel_gen decorator and kernel DSL
│   ├── ttl/                         # TTL Python package (ttl_api.py, dialects, bindings)
│   │   ├── _mlir_libs/              # Shared objects (_ttl.so, _ttmlir.so, etc.)
│   │   ├── _src/                    # Internal modules
│   │   └── dialects/                # Auto-generated dialect Python bindings
│   ├── ttmlir/                      # nanobind extension for tt-mlir dialects
│   ├── ttutils/                     # Shared Python utilities
│   └── utils/                       # Build/codegen utility scripts
├── test/
│   ├── ttlang/                      # MLIR lit tests (FileCheck)
│   │   ├── Conversion/              # Conversion pass tests
│   │   ├── Dialect/                 # Dialect op/verifier tests
│   │   └── Translate/               # Translation tests
│   ├── python/                      # Python lit tests
│   │   ├── invalid/                 # Negative tests (expected errors)
│   │   └── utils/                   # Test utilities
│   ├── me2e/                        # Metal end-to-end tests (require device)
│   └── sim/                         # Simulator tests
├── third-party/
│   ├── llvm-project/                # LLVM/MLIR submodule (pinned commit)
│   ├── tt-mlir/                     # tt-mlir submodule (select dirs compiled)
│   └── tt-metal/                    # tt-metal submodule (runtime)
└── tools/
    ├── ttlang-opt/                  # MLIR opt-like driver
    └── ttlang-translate/            # MLIR translate driver
```

</details>

<details>
<summary>Build output (<code>build/</code>)</summary>

```
build/
├── bin/                             # ttlang-opt, ttlang-translate, llvm-lit, FileCheck
├── env/activate                     # Generated from env/activate.in
├── include/ttmlir/Target/Common/    # Generated flatbuffers stub headers
├── lib/                             # Built libraries
├── llvm-build/                      # LLVM build tree (submodule mode only)
├── llvm-install/                    # LLVM install prefix (submodule mode only)
├── python_packages/ttl/             # Assembled Python package
├── test/                            # Lit test output directory
├── tt-metal-build/                  # tt-metal build tree (note: tt-metal also builds in-source at third-party/tt-metal/build/)
└── venv/                            # Python venv (submodule mode only)
```

</details>

## CMake Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | Build type (Debug, Release, RelWithDebInfo) |
| `LLVM_BUILD_TYPE` | `Release` | LLVM build type (independent of project build type) |
| `TTLANG_TOOLCHAIN_DIR` | — | Toolchain prefix for LLVM, tt-metal, and venv (build or use mode) |
| `TTLANG_USE_TOOLCHAIN` | `OFF` | Use pre-built toolchain at `TTLANG_TOOLCHAIN_DIR` |
| `MLIR_PREFIX` | — | Path to pre-built LLVM/MLIR install (skips submodule build) |
| `TTLANG_ACCEPT_LLVM_MISMATCH` | `OFF` | Allow LLVM SHA mismatch with pre-built installs |
| `TTLANG_ACCEPT_TTMETAL_MISMATCH` | `OFF` | Allow tt-metal SHA mismatch with pre-built installs |
| `TTLANG_ENABLE_PERF_TRACE` | `ON` | Enable tt-metal performance tracing support |
| `TTLANG_ENABLE_DOCS` | `OFF` | Enable Sphinx documentation build (`ttlang-docs` target) |
| `CODE_COVERAGE` | `OFF` | Enable code coverage reporting |
| `TTLANG_FORCE_TOOLCHAIN_REBUILD` | `OFF` | Force rebuild of LLVM and tt-metal into `TTLANG_TOOLCHAIN_DIR` |

## Troubleshooting

### LLVM build takes too long

The first submodule build compiles LLVM from source, which can take 30-60 minutes.
To speed up:
- Ensure ccache is installed (automatically detected and used)
- Use a pre-built LLVM via `-DMLIR_PREFIX` or `-DTTLANG_USE_TOOLCHAIN=ON`
- Subsequent configures skip the build if `llvm-install/` already exists

### LLVM SHA mismatch

If using a pre-built LLVM and cmake reports a SHA mismatch, the installed LLVM
was built from a different commit than what tt-mlir expects. Options:
1. Rebuild LLVM from the correct commit
2. Pass `-DTTLANG_ACCEPT_LLVM_MISMATCH=ON` to proceed (at your own risk)

### Python import errors

Ensure the environment is activated and the build completed:
```bash
source build/env/activate
python3 -c "from ttl.dialects import ttl, ttkernel, ttcore"
```

### nanobind build failures

tt-mlir (and by extension LLVM) builds can be non-deterministic around nanobind
Python bindings. If you see nanobind-related compiler errors, retry the build
2-3 times.

### Missing submodules

```bash
git submodule update --init --recursive
```

For tt-metal specifically, nested submodules (tracy, tt_llk, umd) must also be
initialized. The build emits clear error messages if they are missing.
