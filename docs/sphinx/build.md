# Build System

## Overview

TT-Lang uses a CMake-based build system that compiles LLVM/MLIR, a minimal
tt-mlir subset, tt-metal, and TT-Lang's own dialects and tools from git
submodules at recorded commits. A single
`cmake -G Ninja -B build && cmake --build build` invocation produces a
fully working environment.

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

`.gitmodules` declares three submodules:

| Submodule                    | Purpose                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `third-party/llvm-project` | LLVM/MLIR source (built at configure time)                                   |
| `third-party/tt-mlir`      | tt-mlir source (only select directories compiled)                            |
| `third-party/tt-metal`     | Runtime (built at configure time). Canonical version file: `third-party/tt-metal-version` |

To update any of these, see [Uplifting Submodules](#uplifting-submodules).

### Switching branches

Different branches may record different submodule commits. After switching branches,
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

Each submodule in `third-party` records its commit independently; the three
recorded commits are not derived from one another.

- The LLVM commit in `third-party/llvm-project` is typically newer than
  `LLVM_PROJECT_VERSION` in `third-party/tt-mlir/env/CMakeLists.txt`, and
  the tt-metal commit is on a release tag picked independently from the
  one tt-mlir records in `TT_METAL_VERSION`. Both mismatches are the
  expected steady state, not exceptions.
- tt-lang compiles a subset of tt-mlir and applies patches in
  `third-party/patches/` to make that subset build against the newer LLVM.
- Because the LLVM and tt-metal mismatches are expected, every uplift
  build must bypass cmake's SHA-match check. Pass
  `-DTTLANG_ACCEPT_LLVM_MISMATCH=ON` and `-DTTLANG_ACCEPT_TTMETAL_MISMATCH=ON`
  to cmake. `scripts/build-and-install.sh` accepts the equivalent
  `--accept-ttmetal-mismatch` flag.
- The tt-metal version is recorded in `third-party/tt-metal-version`, a
  one-line file holding a tt-metal release tag. See
  [Updating tt-metal](#updating-tt-metal).

### Updating tt-metal

Edit the canonical version file and run the verifier in update mode. The
verifier checks out `third-party/tt-metal` at the tag's commit; the ttnn
version that `setup.py` writes into the wheel's `install_requires` is
computed dynamically from the same file, so no rewrite is needed:

```bash
echo v0.69.0 > third-party/tt-metal-version
.github/scripts/check-tt-metal-version.sh --update
```

Background: `third-party/tt-metal-version` is the single source of truth
for the tt-metal version (one tt-metal release tag, e.g. `v0.69.0`). The
submodule SHA, the `ttnn` version that `setup.py` writes into the
wheel's `install_requires`, and the `--build-arg TT_METAL_TAG` passed to
`Dockerfile.base` are all derived from it. CI runs
`.github/scripts/check-tt-metal-version.sh` on every PR to catch drift.

### Updating LLVM

```bash
cd third-party/llvm-project && git fetch && git checkout <commit> && cd ../..
```

### Updating tt-mlir

```bash
cd third-party/tt-mlir && git fetch && git checkout <commit> && cd ../..
```

### Rebuilding and committing

A submodule uplift changes what the toolchain (LLVM, tt-metal) is built
from, so the toolchain must be rebuilt; rebuilding tt-lang alone against
the old toolchain will not work. It is recommended to install
the new toolchain to a separate
directory at least initially, so the working default toolchain at
`/opt/ttlang-toolchain` is preserved
in case the uplift fails to build. `scripts/build-and-install.sh` uses
`build-toolchain/` as its cmake build directory by default (set
`CMAKE_BINARY_DIR` to override); you could use a `build-uplift-toolchain/`
to keep the existing `build-toolchain/` artifacts untouched if desired. It
is best to remove any pre-existing uplift-related toolchain build directory
before starting the new toolchain build.

Build the toolchain (LLVM + tt-metal) into the parallel locations:

```bash
CMAKE_BINARY_DIR=build-uplift-toolchain \
TTLANG_TOOLCHAIN_DIR=$PWD/build-uplift/toolchain \
  scripts/build-and-install.sh --toolchain-only --accept-ttmetal-mismatch
```

Then build tt-lang against that toolchain and run the test suites to
validate the uplift before installing it to `/opt/ttlang-toolchain`:

```bash
TTLANG_TOOLCHAIN_DIR=$PWD/build-uplift/toolchain \
  cmake -G Ninja -B build-uplift -DTTLANG_USE_TOOLCHAIN=ON
cmake --build build-uplift

source build-uplift/env/activate
ninja -C build-uplift check-ttlang-mlir          # MLIR lit tests, no hardware
ninja -C build-uplift check-ttlang-all           # full suite (Docker for hw)
```

Test failures here mean the new submodule combination is incompatible —
fix patches under `third-party/patches/` or pick a different SHA before
installing the uplifted toolchain to `/opt/ttlang-toolchain`.

Once the uplift builds and tests cleanly, replace the system toolchain by
re-running without the overrides (so `CMAKE_BINARY_DIR=build-toolchain` and
`TTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain`), then commit the submodule
pointer changes together:

```bash
git add third-party/llvm-project third-party/tt-mlir third-party/tt-metal \
        third-party/tt-metal-version pyproject.toml
git commit -m "Uplift submodules"
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
   git version tag (see `.github/containers/get-version-tag.sh`). Rebuilds
   overwrite the same tag. A `latest` tag is also pushed alongside each
   versioned tag. After building, `call-build-docker.yml` runs the tutorial
   examples in the dist container to verify the image works.

   Tags may carry SemVer build metadata after `+` to mark uplift rebuilds of
   an existing release. Because Docker tags forbid `+`, `get-version-tag.sh`
   translates `+` to `-` when forming the image tag (e.g. git tag
   `<TAG>+<local>` produces Docker tag `<TAG>-<local>`). Use the sanitized
   form anywhere a `docker_tag` parameter is passed to a workflow.

#### Triggering a toolchain cache rebuild on PRs

By default, PR and push workflows use a pre-built Docker container and skip
building the toolchain from source. For uplift PRs where the recorded
submodule commits have changed, pass `build_toolchain: true` to force a
from-source build:

```yaml
# In on-pr.yml or on-push.yml, pass build_toolchain to call-build.yml:
build:
  uses: ./.github/workflows/call-build.yml
  secrets: inherit
  with:
    build_toolchain: true
    docker_tag: "<DOCKER_TAG>"
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

Docker images are built by `call-build-docker.yml`, which is invoked either by
manual `workflow_dispatch` or as a reusable sub-workflow of `publish-pypi.yml`
(see [Publishing to PyPI](#publishing-to-pypi) below). The workflow:

1. Generates a deterministic tag from submodule SHAs and Dockerfile content
   hashes.
2. Checks whether images with that tag already exist in the registry.
3. On cache miss, builds the toolchain (or restores from GitHub Actions cache),
   then packages `base`, `ird`, and `dist` images.

Pushing a release tag triggers `publish-pypi.yml`, which calls
`call-build-docker.yml` as its first step — so the same `git push <tag>` that
publishes a release also rebuilds the Docker images. For uplifts that rebuild
against a prior release (rather than advancing MAJOR/MINOR/PATCH), append
`+uplift` (or another `+<local>` identifier) so the tag preserves SemVer
ordering with the original release:

```bash
# Standard release bump:
git tag <TAG>
git push origin <TAG>

# Uplift of an existing release (new submodule SHAs on top of an existing tag):
git tag <TAG>+<local>
git push origin <TAG>+<local>
```

Once the new images are published, update the `docker_tag` parameter in
`on-pr.yml` and `on-push.yml` to reference the new tag. For `+`-suffixed
tags, use the Docker-sanitized form: git tag `<TAG>+<local>` -> docker_tag
`<TAG>-<local>`.

(publishing-to-pypi)=
#### Publishing to PyPI

`publish-pypi.yml` is the orchestrator that turns a release tag into a wheel
on PyPI. It triggers automatically on push of `v*.*.*` or `v*.*.*+<local>`
tags, and can also be dispatched manually for re-runs and dry-runs.

```text
   push release tag
 or workflow_dispatch
          |
          v
   +--------------+
   |  preflight   |   verify GITHUB_REF is a v* tag
   +--------------+   (skipped if dry_run=true)
          |
          +-----------------------+
          |                       |
          v                       |
   +--------------+               |
   | build-docker |   call-build-docker.yml
   +--------------+   (skipped if docker_tag input is set)
          |                       |
          +-----------------------+
          |
          v
   +--------------+
   | build-wheels |   call-build-wheels.yml
   +--------------+   (builds wheel inside ird container,
          |            uploads tt-lang-wheels artifact)
          |
          +-----------------------+
          v                       v
   +--------------+        +------------------+
   |   publish    |        | dry-run-summary  |
   +--------------+        +------------------+
   tag push or              workflow_dispatch
   dry_run=false            with dry_run=true
   (uploads to PyPI)        (lists artifacts only)
```

Job-by-job:

1. **`preflight`** — runs `require-release-tag.sh`, which fails unless
   `GITHUB_REF` looks like `refs/tags/v[0-9]...`. Skipped under
   `dry_run: true`. Exposes `tag_version` (tag with leading `v` stripped) for
   the wheel-version check.
2. **`build-docker`** — calls `call-build-docker.yml` on tag push (where no
   `docker_tag` input is supplied). Skipped on `workflow_dispatch`, which
   requires `docker_tag`. Outputs the freshly built ird tag.
3. **`build-wheels`** — calls `call-build-wheels.yml` against either the
   `docker_tag` input (manual dispatch) or the `build-docker` output (tag
   push). Builds the wheel inside the ird container and uploads it as the
   `tt-lang-wheels` artifact.
4. **`publish`** — runs on tag push or when `dry_run` is false. Downloads the
   artifact, verifies every wheel filename's version field matches
   `preflight.outputs.tag_version`, and uploads via
   `pypa/gh-action-pypi-publish` using OIDC trusted publishing
   (`environment: pypi`, `id-token: write`).
5. **`dry-run-summary`** — runs only on `workflow_dispatch` with
   `dry_run: true`. Downloads the artifact and lists what would have been
   uploaded. No `environment`, no PyPI credentials.

Common scenarios:

Common scenarios (`<TAG>` denotes a release tag, `<DOCKER_TAG>` an existing
ird image tag):

| Trigger                                                       | docker_tag input | Result                                                              |
| ------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------- |
| `git push origin <TAG>`                                       | (n/a)            | Build docker, build wheel, publish to PyPI as the tag's version     |
| Dispatch from a tag ref with `docker_tag: <DOCKER_TAG>`       | required         | Skip docker build, reuse the supplied ird image, publish to PyPI    |
| Dispatch from a non-tag ref with `dry_run: true`              | required         | Build wheel against the supplied tag, skip PyPI upload              |
| Dispatch from a non-tag ref with `dry_run: false`             | required         | Fails at `preflight` because `GITHUB_REF` is not a release tag      |

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
| `TTLANG_SIM_ONLY`                | `OFF`     | Set up Python environment for [simulator](simulator.md) only; skip compiler build       |
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
