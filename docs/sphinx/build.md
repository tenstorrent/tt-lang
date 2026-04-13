# Build System

## Overview

TT-Lang uses a CMake-based build system that compiles LLVM/MLIR, a minimal
tt-mlir subset, tt-metal, and TT-Lang's own dialects and tools from pinned git
submodules. A single `cmake -G Ninja -B build && cmake --build build` invocation
produces a fully working environment.

## Prerequisites

- CMake 3.28+
- Ninja
- Clang/Clang++ 17+ (or GCC 12+)
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

### Build a reusable toolchain

```bash
cmake -G Ninja -B build -DTTLANG_BUILD_TOOLCHAIN=ON -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain
source build/env/activate
cmake --build build
```

Builds LLVM/MLIR and tt-metal from submodules and installs them into the given
prefix so they can be reused by other builds. Any existing installation at the
target directory is cleaned automatically to prevent stale libraries from being
linked. If `TTLANG_TOOLCHAIN_DIR` is omitted, defaults to
`build/toolchain-install/`.

The convenience script `scripts/build-and-install.sh --toolchain-only` automates
this — it configures, builds LLVM + tt-metal, installs them into the toolchain
prefix, and cleans up. The build directory defaults to `build-toolchain/`; set
the `CMAKE_BINARY_DIR` environment variable to use a different location. The
toolchain install location defaults to `/opt/ttlang-toolchain`; set the
`TTLANG_TOOLCHAIN_DIR` environment variable to change it.

> **Note:** Setting only `-DTTLANG_TOOLCHAIN_DIR=...` (without
> `TTLANG_BUILD_TOOLCHAIN`) will reuse an existing installation if one is found
> at that directory. Use `TTLANG_BUILD_TOOLCHAIN=ON` to guarantee a fresh build.

### Install a toolchain locally

To build and install just the toolchain (LLVM + tt-metal) without building
tt-lang itself:

```bash
# Ensure you own the install prefix
sudo mkdir -p /opt/ttlang-toolchain && sudo chown $USER /opt/ttlang-toolchain

TTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain scripts/build-and-install.sh --toolchain-only
```

This runs the full configure (building LLVM and tt-metal from submodules),
installs tt-metal artifacts into the prefix, and finalizes the installation.
Set `TTLANG_TOOLCHAIN_DIR` to change the install location (default:
`/opt/ttlang-toolchain`). Once installed, use `-DTTLANG_USE_TOOLCHAIN=ON` for
fast rebuilds of tt-lang itself.

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
submodule. TT-Lang may not build successfully if the pre-built LLVM is a
significantly different version than what tt-mlir expects.

## Installing

Installation is used to create self-contained distribution packages (e.g.,
Docker images). It is not needed for development — just use
`source build/env/activate` after building to get a fully working environment.

```bash
cmake --install build --prefix /opt/ttlang-toolchain
```

This copies TT-Lang binaries, Python packages, examples, tests, and the
environment activation script into the given prefix. When `TTLANG_TOOLCHAIN_DIR`
was set during configure, LLVM, tt-metal, and the Python venv are already there;
the install step adds only TT-Lang's own artifacts.

## Building Documentation

```bash
cmake -G Ninja -B build -DTTLANG_ENABLE_DOCS=ON
cmake --build build --target ttlang-docs
python -m http.server 8000 -d build/docs/sphinx/_build/html
```

Open `http://localhost:8000` to browse the docs locally.

## Submodules

`.gitmodules` pins three submodules:

| Submodule                    | Purpose                                           |
| ---------------------------- | ------------------------------------------------- |
| `third-party/llvm-project` | LLVM/MLIR source (built at configure time)        |
| `third-party/tt-mlir`      | tt-mlir source (only select directories compiled) |
| `third-party/tt-metal`     | Runtime (built at configure time)                 |

### Switching branches

Different branches may pin different submodule commits. After switching branches,
update the submodules to match:

```bash
git checkout <branch>
git submodule update --init --force --depth 1
```

`--force` is required because CMake applies patches to the submodule working
trees at configure time. Without it, `git submodule update` refuses to overwrite
the patched files. This is safe because the patches are tracked in
`third-party/patches/` and re-applied automatically on the next configure.

For tt-metal's nested submodules (tracy, tt_llk, umd):

```bash
git -C third-party/tt-metal submodule update --init --force --depth 1
```

Do not use `--recursive` at the top level — LLVM's nested submodules are large
and not needed.

Or use the convenience script that handles both steps:

```bash
scripts/update-submodules.sh
```

After updating submodules, reconfigure and rebuild:

```bash
cmake -G Ninja -B build
cmake --build build
```

### LLVM SHA verification

When using a pre-built LLVM (via `MLIR_PREFIX` or `TTLANG_USE_TOOLCHAIN`), the
build verifies the installed LLVM was built from the expected commit. The
expected SHA is read from `third-party/tt-mlir/env/CMakeLists.txt`
(`LLVM_PROJECT_VERSION`), and the actual SHA is read from
`<prefix>/include/llvm/Support/VCSRevision.h`. On mismatch, cmake emits a
`FATAL_ERROR`. Pass `-DTTLANG_ACCEPT_LLVM_MISMATCH=ON` to proceed despite the
mismatch.

## Uplifting Submodules

Uplifting means updating submodule pins to newer commits. tt-mlir defines the
compatible versions of LLVM and tt-metal, so when updating tt-mlir, update the
other submodules to match. Note that you can specify other SHAs for LLVM
or tt-metal, but then may have to bypass SHA mismatch checks by specifying the
`TTLANG_ACCEPT_LLVM_MISMATCH` and `TTLANG_ACCEPT_TTMETAL_MISMATCH` options to cmake.

### Local uplift procedure

```bash
# Update tt-mlir to the desired commit
cd third-party/tt-mlir && git fetch && git checkout <commit> && cd ../..

# Update LLVM and tt-metal to the versions tt-mlir expects
scripts/update-submodules.sh

# Rebuild
cmake -G Ninja -B build
cmake --build build
```

Commit all submodule pointer changes together:

```bash
git add third-party/llvm-project third-party/tt-mlir third-party/tt-metal
git commit -m "Update submodules to tt-mlir <short-sha>"
```

### CI: toolchain cache and Docker images

CI uses two caching layers that must be rebuilt when submodule SHAs change:

1. **GitHub Actions toolchain cache** -- a cached LLVM + tt-metal build keyed
   by the LLVM and tt-metal submodule SHAs
   (`Linux-toolchain_llvm-<sha>_ttmetal-<sha>`). When an uplift changes either
   SHA, the cache key changes and the
   `call-build-toolchain.yml` workflow automatically builds and caches a new
   toolchain.

2. **Docker images** -- `ird` and `dist` container images tagged by the nearest
   git version tag (e.g. `v0.1.8`, see `.github/containers/get-version-tag.sh`).
   Rebuilds overwrite the same tag. A `latest` tag is also pushed alongside
   each versioned tag. After building, `call-build-docker.yml` runs the
   tutorial examples in the dist container to verify the image works.

#### Triggering a toolchain cache rebuild on PRs

By default, PR and push workflows use a pre-built Docker container and skip
building the toolchain from source. For uplift PRs where the submodule pins have
changed, pass `build_toolchain: true` to force a from-source build:

```yaml
# In on-pr.yml or on-push.yml, pass build_toolchain to call-build.yml:
build:
  uses: ./.github/workflows/call-build.yml
  secrets: inherit
  with:
    build_toolchain: true
    docker_tag: "v0.1.8"
```

When `build_toolchain` is true, the workflow:

1. Runs `call-build-toolchain.yml`, which checks for a cached toolchain
   matching the current submodule SHAs. On cache miss, it builds LLVM + tt-metal
   from source and saves the result.
2. Runs the build job on a bare `ubuntu-22.04` runner (instead of inside the
   Docker container), restoring the cached toolchain and building tt-lang
   against it.

When `build_toolchain` is false (the default), the build job runs inside the
pre-built `ird` Docker container, which already contains the toolchain.

#### Rebuilding Docker images

Docker images are rebuilt automatically by `call-build-docker.yml`, which runs
on version tags (`v*.*.*`) or manual dispatch. The workflow:

1. Generates a deterministic tag from submodule SHAs and Dockerfile content
   hashes.
2. Checks whether images with that tag already exist in the registry.
3. On cache miss, builds the toolchain (or restores from GitHub Actions cache),
   then packages `base`, `ird`, and `dist` images.

After an uplift merges, create a new version tag to trigger image rebuilds:

```bash
git tag v0.1.8
git push origin v0.1.8
```

Once the new images are published, update the `docker_tag` parameter in
`on-pr.yml` and `on-push.yml` to reference the new tag.

## CMake Options

| Option                             | Default     | Description                                                                          |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------ |
| `CMAKE_BUILD_TYPE`               | `Release` | Build type (Debug, Release, RelWithDebInfo)                                          |
| `LLVM_BUILD_TYPE`                | `Release` | LLVM build type (independent of project build type)                                  |
| `TTLANG_TOOLCHAIN_DIR`           | —          | Toolchain prefix for LLVM, tt-metal, and venv                                        |
| `TTLANG_USE_TOOLCHAIN`           | `OFF`     | Use pre-built toolchain at `TTLANG_TOOLCHAIN_DIR`                                  |
| `TTLANG_BUILD_TOOLCHAIN`         | `OFF`     | Build LLVM and tt-metal into a reusable toolchain directory (cleans stale artifacts) |
| `MLIR_PREFIX`                    | —          | Path to pre-built LLVM/MLIR install                                                  |
| `TTLANG_ACCEPT_LLVM_MISMATCH`    | `OFF`     | Allow LLVM SHA mismatch with pre-built installs                                      |
| `TTLANG_ACCEPT_TTMETAL_MISMATCH` | `OFF`     | Allow tt-metal SHA mismatch with pre-built installs                                  |
| `TTLANG_ENABLE_PERF_TRACE`       | `ON`      | Enable tt-metal performance tracing support                                          |
| `TTLANG_SIM_ONLY`                | `OFF`     | Set up Python environment for[simulator](simulator.md) only; skip compiler build        |
| `TTLANG_ENABLE_DOCS`             | `OFF`     | Enable Sphinx documentation build (`ttlang-docs` target)                           |
| `CODE_COVERAGE`                  | `OFF`     | Enable code coverage reporting                                                       |
| `TTLANG_FORCE_TOOLCHAIN_REBUILD` | `OFF`     | Force rebuild of LLVM and tt-metal into `TTLANG_TOOLCHAIN_DIR`                     |

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
