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
git push
```

On push, `resolve-docker-tag` (see [Auto-resolved tag in PR /
push workflows](#auto-resolved-tag-in-pr--push-workflows)) sees the
uplift-relevant paths changed since the nearest version tag and emits
`vX.Y.Z-uplift-<hash>`; if the corresponding image is missing in GHCR,
the `build-docker` job builds and pushes it before any other downstream
job consumes it. Subsequent pushes of the same submodule SHA set reuse
the cached image.

### CI: toolchain cache and Docker images

CI uses two caching layers that must be rebuilt when submodule SHAs change:

1. **GitHub Actions toolchain cache** -- a cached LLVM + tt-metal build keyed
   by the LLVM and tt-metal submodule SHAs
   (`Linux-toolchain_llvm-<sha>_ttmetal-<sha>`). When an uplift changes either
   SHA, the cache key changes and the
   `call-build-toolchain.yml` workflow automatically builds and caches a new
   toolchain.

2. **Docker images** -- `ird` and `dist` container images at GHCR, tagged by
   `.github/containers/get-version-tag.sh` (see [Docker tag scheme](#docker-tag-scheme)).
   Uplift-hashed tags (`vX.Y.Z-uplift-<hash>`) include a hash of the content
   installed into the image (tt-metal submodule + version pin, LLVM submodule,
   `Dockerfile.base`, `requirements-runtime.txt`), so the same toolchain
   state always resolves to the same tag. The bare release tag (`vX.Y.Z`) is
   only pushed by `publish-pypi.yml` on a release tag push, and `:latest` is
   only updated from `ci.yml` on push to `main`, and only when `build-docker`
   actually runs there (i.e. an uplift commit whose image is not already in
   GHCR). `call-build-docker.yml` takes
   a `push` input (default `false`); it builds the image, smoke-tests it
   inside `docker run`, and pushes to GHCR only when `push: true`.
   Tutorial verification in the dist container runs separately as a
   pre-publish check; see [Publishing to PyPI](#publishing-to-pypi).

(docker-tag-scheme)=
#### Docker tag scheme

`get-version-tag.sh` returns one of two forms, derived deterministically from
the current checkout:

- **Clean release state** (`vX.Y.Z`): the files in
  `.github/scripts/uplift-paths.sh` match the nearest version tag commit.
  The script returns the tag name itself, with `+` translated to `-` because
  Docker tags allow only `[A-Za-z0-9_.-]`.
- **Uplift state** (`vX.Y.Z-uplift-<8char>`): one or more of those files
  differ from the nearest version tag. The hash is
  `git ls-tree HEAD -- <uplift-files> | sha256sum | cut -c1-8`, so two
  branches with identical submodule SHAs and Dockerfile/requirements content
  resolve to the same tag and share the rebuilt image. "Uplift" here means
  the dist/ird image content changed — tt-mlir and tt-lang itself are built
  fresh by `call-build.yml` against the pre-built LLVM inside the container,
  so they are not uplift files.

#### Auto-resolved tag in `ci.yml`

`ci.yml` (one workflow triggered by pull_request, push to main, scheduled
runs, and workflow_dispatch) starts with a `resolve-docker-tag` job that
runs `get-version-tag.sh` and then calls `.github/scripts/probe-docker-image.sh`
to query GHCR. If the image is present, `build-docker` is skipped and
downstream jobs proceed immediately. If the image is missing and the
resolved tag is the uplift form, `build-docker` runs `call-build-docker.yml`
with `push: true` and uploads the rebuilt image so downstream jobs
(`build`, `build-wheels`, `test-hardware`, `test-dist-tutorials`) can pull
it. If the image is missing and the resolved tag is the bare release form
(e.g. `vX.Y.Z`), the probe step fails the job with an error directing the
maintainer to re-publish the release via `publish-pypi.yml`; rebuilding the
release tag from a PR or main commit would push newer content under the
release tag and overwrite the released image.

`ci.yml` also has a `dryrun-docker` job that runs only on
pull_request events when the PR touches container-relevant files
(Dockerfile, `bin/`, `packaging/`, `CMakeLists.txt`, `examples/`,
`pyproject.toml`, etc.) but the uplift `build-docker` is not already
running. It calls `call-build-docker.yml` with `push: false`: the dist and
ird images are built locally on the runner and the in-container smoke
tests run, but nothing is uploaded to GHCR. This catches container-build
regressions at PR time without uploading a separate container image for
every PR. The path-change detection is in
`.github/scripts/wheel-or-container-changed.sh` (path list in
`wheel-or-container-paths.sh`).

`call-build.yml` retains its `build_toolchain` input for manual
`workflow_dispatch` runs, but the automated workflows no longer set it:
the correct toolchain is always available inside the container at the
resolved tag.

#### Hardware test timeouts

`call-test-hardware.yml` and `call-test-dist-tutorials.yml` pass
`--timeout=60 --timeout-method=signal` to every pytest invocation so a hung
test exits within ~60 seconds instead of holding the single `n150` runner
until the 90-minute job timeout. Tests that legitimately need longer should
set their own `@pytest.mark.timeout(...)` override.

#### Rebuilding Docker images

Docker images are built by `call-build-docker.yml`. The workflow takes a
`push` input (default `false`); the image is tagged with whatever
`get-version-tag.sh` returns and smoke-tested with `docker run` before any
push step. A failing smoke test aborts before any tag would be published.

Push policy across events:

| Event                                  | Pushes `vX.Y.Z`?     | Pushes `vX.Y.Z-uplift-<hash>`? | Updates `:latest`?           |
|----------------------------------------|----------------------|--------------------------------|------------------------------|
| PR (uplift)                            | refused by probe     | yes                            | no                           |
| PR (non-uplift, container content)     | no (dryrun)          | n/a                            | no                           |
| Main push (uplift)                     | refused by probe     | yes                            | yes                          |
| Main push (non-uplift)                 | n/a (image exists)   | n/a                            | no (`build-docker` skipped)  |
| Tag push (release, via publish-pypi)   | yes                  | n/a                            | no                           |
| `workflow_dispatch`                    | only if `push: true` | only if `push: true`           | only if `push: true` on main |

For a final release:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

For a dated dev release (preview of an in-flight version, typically used
after a toolchain uplift lands on `main` and before the next final tag),
follow the tt-metal convention: SemVer pre-release identifier of the form
`-dev<YYYYMMDD>`:

```bash
git tag v1.2.0-dev20260515
git push origin v1.2.0-dev20260515
```

SemVer orders `vX.Y.Z-dev<date>` strictly below `vX.Y.Z` (final), so users
who pin to `vX.Y.Z` are unaffected by dev releases. Within a single
`vX.Y.Z` line, dev tags order monotonically by date. The form is
Docker-tag-safe directly (no `+` translation needed). `-rc<N>` works the
same way (`v1.2.0-rc1` is a release candidate of `v1.2.0`).

Legacy `<TAG>+<local>` build-metadata tags are still translated to
`<TAG>-<local>` by `get-version-tag.sh` for image-tag compatibility, but
SemVer treats `+`-suffixed tags as equal in precedence to the base tag, so
they cannot be distinguished by `pip install`. Prefer `-dev<YYYYMMDD>` or
`-rc<N>` for new tags.

(publishing-to-pypi)=
#### Publishing to PyPI

`publish-pypi.yml` is the orchestrator that turns a release tag into a wheel
on PyPI. It triggers automatically on push of `v*.*.*`, `v*.*.*-rc*`,
`v*.*.*-dev*`, or `v*.*.*+*` tags, and can also be dispatched manually for
re-runs and dry-runs.

```text
   push release tag
 or workflow_dispatch
          |
          v
   +--------------+
   |  preflight   |   verify GITHUB_REF is a v* tag
   +--------------+   (skipped if dry_run=true)
          |
          v
   +--------------+
   | build-docker |   call-build-docker.yml
   +--------------+   (skipped if docker_tag input is set;
          |            smoke-tests image before push to GHCR)
          v
   +--------------+
   | build-wheels |   call-build-wheels.yml
   +--------------+   (builds + smoke-tests wheel inside ird container,
          |            uploads tt-lang-wheels artifact)
          |
          +-----------------------+
          v                       |
   +---------------------+        |
   | test-dist-tutorials |        |  (skipped under dry_run=true)
   +---------------------+        |
          |                       |
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
   push). Builds the wheel inside the ird container, runs
   `smoke-test-wheel.py` in an isolated venv (imports + `tt-lang-sim --help`
   + `tt-lang-sim-stats --help`), and runs the CMake-install regression test
   (`cmake --install` + `bin/tt-lang-sim --help` against the parallel-install
   layout). Uploads the result as the `tt-lang-wheels` artifact.
4. **`test-dist-tutorials`**: calls `call-test-dist-tutorials.yml` against
   the dist image at the resolved tag, running the tutorial suite on the
   `n150` hardware runner. Gates `publish`. Skipped under `dry_run: true`.
5. **`publish`**: runs on tag push or when `dry_run` is false **and**
   `test-dist-tutorials` succeeded. Downloads the artifact, verifies every
   wheel filename's version field matches `preflight.outputs.tag_version`,
   and uploads via `pypa/gh-action-pypi-publish` using OIDC trusted
   publishing (`environment: pypi`, `id-token: write`).
6. **`dry-run-summary`**: runs only on `workflow_dispatch` with
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
