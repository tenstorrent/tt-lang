# Build System

## Overview

TT-Lang uses a CMake-based build system that compiles LLVM/MLIR, tt-metal, and
TT-Lang's own dialects and tools from git submodules at recorded commits. A single
`cmake -G Ninja -B build && cmake --build build` invocation produces a
fully working environment.

## Prerequisites

- CMake 3.28+
- Ninja
- Clang/Clang++ 17+ (or GCC 12+)
- Python 3.10+ (Python 3.12 recommended)
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
`build/llvm-install/`. tt-metal builds to `third-party/tt-metal/build/`. TT-Lang's
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

### Install an LLVM-only toolchain with external tt-metal

When a developer already has a local tt-metal build, TT-Lang can install only
LLVM/MLIR and the toolchain Python venv. tt-metal stays external and is passed
to CMake at configure time:

```bash
TTLANG_TOOLCHAIN_DIR=/opt/ttlang-llvm-toolchain \
  scripts/build-and-install.sh \
    --llvm-toolchain-only \
    --force-rebuild \
    --external-tt-metal-dir /path/to/tt-metal \
    --external-tt-metal-build-dir /path/to/tt-metal/build \
    --python-venv /path/to/tt-metal/python_env
```

The resulting prefix contains LLVM/MLIR and the venv. It is not a complete
TT-Lang distribution and does not install tt-metal under the prefix. Use it
by combining the LLVM toolchain with the same external tt-metal selection:

```bash
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTTLANG_USE_TOOLCHAIN=ON \
  -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-llvm-toolchain \
  -DTTLANG_EXTERNAL_TT_METAL_DIR=/path/to/tt-metal \
  -DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR=/path/to/tt-metal/build \
  -DTTLANG_PYTHON_VENV=/path/to/tt-metal/python_env
```

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
significantly different version than what tt-lang expects.

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

`.gitmodules` declares two submodules:

| Submodule                  | Purpose                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| `third-party/llvm-project` | LLVM/MLIR source (built at configure time)                                                 |
| `third-party/tt-metal`     | Runtime (built at configure time). Canonical version file: `third-party/tt-metal-version`  |

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
expected SHA is the commit recorded by the `third-party/llvm-project` submodule
gitlink, and the actual SHA is read from
`<prefix>/include/llvm/Support/VCSRevision.h`. On mismatch, cmake emits a
`FATAL_ERROR`. Pass `-DTTLANG_ACCEPT_LLVM_MISMATCH=ON` to proceed despite the
mismatch. When the submodule is not populated (the usual case with a pre-built
toolchain), the check is skipped.

## Uplifting Submodules

Each submodule in `third-party` records its commit independently; the two
recorded commits are not derived from one another.

- tt-lang owns its MLIR dialects, conversion, and translation in-tree, so the
  LLVM commit in `third-party/llvm-project` and the tt-metal commit are chosen
  directly by tt-lang. tt-metal is typically on a release tag.
- Because a pre-built toolchain's LLVM may differ from the submodule pin, an
  uplift build may need to bypass cmake's LLVM SHA-match check. Pass
  `-DTTLANG_ACCEPT_LLVM_MISMATCH=ON` to cmake.
- The tt-metal and public `ttnn` provenance versions are recorded in
  `third-party/tt-metal-version`. See
  [Updating tt-metal](#updating-tt-metal).

### Updating tt-metal

Edit the canonical version file and run the verifier in update mode. The
verifier checks out `third-party/tt-metal` at `TT_METAL_TAG`; the `ttnn`
version that `setup.py` writes into the wheel's `install_requires` is read
from `TTNN_PYPI`, so no rewrite is needed:

```text
TTNN_PYPI="<ttnn-pypi-version>"
TTNN_PYPI_TT_METAL_TAG="<ttnn-pypi-tt-metal-tag>"
TT_METAL_TAG="<tt-metal-tag>"
```

```bash
.github/scripts/check-tt-metal-version.sh --update
```

`TTNN_PYPI_TT_METAL_TAG` records the tt-metal tag used to build the public
`ttnn` wheel. `TT_METAL_TAG` records the tt-metal tag used to build TT-Lang.
Public PyPI publishing requires these tags to have the same `vX.Y.Z` component;
S3-hosted bundled wheels can use a newer `TT_METAL_TAG` before a compatible
public `ttnn` wheel is available.

Also update the simulator pin. The macOS/simulator harness downloads a pinned
`tenstorrent/ttsim` release whose `libttsim.so` must be ABI-compatible with the
tt-metal being built, so set `TTSIM_VERSION` in `test/hw-sim/vm-install-sim.sh` to
a ttsim release compatible with the new tt-metal (see the
[libttsim API](https://github.com/tenstorrent/ttsim/blob/main/docs/libttsim_api.md)).

Background: `third-party/tt-metal-version` is the single source of truth for
the `ttnn` dependency version, the public `ttnn` provenance tag, and the
tt-metal tag passed to `Dockerfile.base`. CI runs
`.github/scripts/check-tt-metal-version.sh` on every PR to catch submodule
drift.

### Two-phase uplift: publishable release, then latest (S3-only)

When the newest tt-metal tag is ahead of the latest public `ttnn` wheel, a
single uplift to that tag cannot publish to public PyPI: `ttnn_pypi_aligned`
(`.github/scripts/lib/tt-metal-version-utils.sh`) requires `TT_METAL_TAG` and
`TTNN_PYPI_TT_METAL_TAG` to share a `vX.Y.Z` component, and `publish-pypi.yml`
refuses the release otherwise. To ship a public PyPI release *and* pick up the
newest tt-metal, split the work into two uplifts and land the publishable one
first:

1. **Publishable phase.** Set `TT_METAL_TAG` to the tt-metal tag whose `vX.Y.Z`
   matches `TTNN_PYPI_TT_METAL_TAG`, and keep `TTNN_PYPI` at that public `ttnn`
   version. The `-rc` suffix is ignored by the check, so `TT_METAL_TAG="v0.73.1"`
   aligns with `TTNN_PYPI_TT_METAL_TAG="v0.73.1-rc5"`. This uplift's `vX.Y.Z`
   release tag publishes to public PyPI.
2. **Latest phase.** On top of the first, bump `TT_METAL_TAG` to the latest
   tt-metal tag and leave `TTNN_PYPI`/`TTNN_PYPI_TT_METAL_TAG` unchanged. The tags
   now diverge on `vX.Y.Z`, so the wheel is S3-only until a matching public
   `ttnn` ships (S3 publishing runs from the weekly schedule or a manual
   dispatch on `main`, never a tag push).

Build, validate, and commit each phase separately (see [Rebuilding and
committing](#rebuilding-and-committing)) as its own PR. When both phases move the
same `third-party/tt-metal` gitlink and `TT_METAL_TAG` line, restack the second
PR after the first merges so it applies onto the updated `main`.

### Updating LLVM

`third-party/llvm-project` is a shallow clone, so a bare `git fetch` only
refreshes the default branch tip and may leave an arbitrary commit unreachable
("reference is not a tree"). Fetch the exact SHA:

```bash
git -C third-party/llvm-project fetch --depth 1 origin <full-sha>
git -C third-party/llvm-project checkout --detach <full-sha>
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

**Toolchain reuse is keyed on file existence, not on the submodule SHA.**
`BuildLLVM` skips the LLVM build when
`$TTLANG_TOOLCHAIN_DIR/lib/cmake/mlir/MLIRConfig.cmake` already exists (a SHA
mismatch against the submodule is only an `AUTHOR_WARNING`, never an error), and
`BuildTTMetal` skips when `$CMAKE_BINARY_DIR/tt-metal/ttnn/_ttnn.so` already
exists. Rebuilding an uplift into a populated toolchain or build directory
therefore silently keeps the *old* LLVM and tt-metal, and the uplifted SHAs are
never actually compiled or tested. When the SHAs change, remove both the uplift
toolchain build directory and the target toolchain directory first (or pass
`--force-rebuild`). If the target lives under `/opt`, recreate and chown it:

```bash
sudo rm -rf /opt/ttlang-toolchain-<version>
sudo mkdir -p /opt/ttlang-toolchain-<version>
sudo chown "$USER": /opt/ttlang-toolchain-<version>
```

Build the toolchain (LLVM + tt-metal) into the parallel locations:

```bash
CMAKE_BINARY_DIR=build-uplift-toolchain \
TTLANG_TOOLCHAIN_DIR=$PWD/build-uplift/toolchain \
  scripts/build-and-install.sh --toolchain-only
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

Uplift failures land in three distinct stages:

- **A tt-metal patch fails to apply**, aborting the toolchain configure with a
  `FATAL_ERROR`. The patches in `third-party/patches/` are context-sensitive and
  drift whenever tt-metal moves. Regenerate the patch against the new source and
  commit the updated `.patch` alongside the uplift. Pre-check before starting the
  long build with
  `git -C third-party/tt-metal apply --check third-party/patches/<name>.patch`.
- **tt-lang fails to compile against the new LLVM.** This surfaces in the
  validation build, not the toolchain build, because `--toolchain-only` never
  builds tt-lang. The cause is upstream MLIR API churn; fix the tt-lang source and
  include it in the uplift commit. `third-party/patches/` only patches tt-metal
  and cannot address this.
- **Tests fail**, meaning the submodule combination is incompatible. Pick a
  different SHA before installing the uplifted toolchain to
  `/opt/ttlang-toolchain`.

Once the uplift builds and tests cleanly, replace the system toolchain by
re-running without the overrides (so `CMAKE_BINARY_DIR=build-toolchain` and
`TTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain`), then commit the submodule
pointer changes together:

```bash
git add third-party/llvm-project third-party/tt-metal \
        third-party/tt-metal-version
git commit -m "Uplift submodules"
git push
```

Add any regenerated `third-party/patches/*.patch`, and any tt-lang source fix the
new LLVM required, to the same commit. `pyproject.toml` needs no edit: its
version, readme and dependencies are all `dynamic`, the wheel version is derived
from git tags, and `setup.py` reads the `ttnn` pin out of
`third-party/tt-metal-version` at build time.

On push, `resolve-docker-tag` (see [Auto-resolved tag in PR /
push workflows](#auto-resolved-tag-in-pr--push-workflows)) sees the
uplift-relevant paths changed since the nearest version tag and emits
`vX.Y.Z-uplift-<hash>`; if the corresponding image is missing in GHCR,
the `build-docker` job builds and pushes it before any other downstream
job consumes it. Subsequent pushes of the same submodule SHA set reuse
the cached image.

(tagging-after-an-uplift-merge)=
### Tagging after an uplift merge

Wait for the merge's CI run on `main` to finish building the toolchain before
pushing the release tag. A run reads caches only from its own ref and the
default branch, and the uplift PR's cache lives under `refs/pull/<N>/merge`, so
a tag pushed before `main` has cached the toolchain rebuilds it from source --
about an hour and a half, concurrent with the identical build on `main`.

The tag run cannot skip it: `call-build-docker.yml`'s `build-images` restores
the toolchain with `fail-on-cache-miss: true` and bakes it into the IRD image.
Dispatching `call-build-docker.yml` separately resolves the same cache scopes
and rebuilds too.

Confirm the cache exists first; the key uses seven-character submodule SHAs:

```bash
gh api "repos/tenstorrent/tt-lang/actions/caches?per_page=100" \
  --jq '.actions_caches[] | select(.key | startswith("Linux-toolchain")) | "\(.ref)  \(.key)"'
```

Look for `Linux-toolchain_llvm-<llvm7>_ttmetal-<ttmetal7>` at
`ref=refs/heads/main` matching the uplifted submodules, then push the tag.

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
- **Uplift state** (`vX.Y.Z-<8char>`): one or more of those files
  differ from the nearest version tag. The hash is
  `git ls-tree HEAD -- <uplift-files> | sha256sum | cut -c1-8`, so two
  branches with identical submodule SHAs and Dockerfile/requirements content
  resolve to the same tag and share the rebuilt image. "Uplift" here means
  the dist/ird image content changed — tt-lang itself is built
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
(`build`, `build-wheels`, `test-hardware`, `test-exabox`,
`test-dist-tutorials`) can pull it. If the image is missing and the resolved
tag is the bare release form
(e.g. `vX.Y.Z`), the probe step fails the job with an error directing the
maintainer to run `call-build-docker.yml` with `push: true` at that exact
release tag. Recovery uses that workflow rather than `publish-pypi.yml` so the
image is rebuilt from the tagged commit alone; rebuilding the release tag from
a PR or main commit would push newer content under the release tag and
overwrite the released image.

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

Manylinux wheel-builder images use the same deterministic tag format, but
their input list extends `.github/scripts/uplift-paths.sh` with the
manylinux builder Dockerfile, CMake driver, and builder driver scripts. This
keeps changes to those builder-only files from invalidating the shared
ird/dist image tag while still producing a new manylinux builder tag when the
builder assembly changes.

`call-build.yml` retains its `build_toolchain` input for manual
`workflow_dispatch` runs, but the automated workflows no longer set it:
the correct toolchain is always available inside the container at the
resolved tag.

#### Hardware test timeouts

`call-test-hardware.yml` and `call-test-exabox.yml` pass
`--timeout=60 --timeout-method=signal` to simulator tests. Device suites run
through `.github/scripts/run-hardware-pytests.sh`, which passes
`--timeout=300 --timeout-method=thread`; the thread method interrupts C-level
device deadlocks that `SIGALRM` cannot. `call-test-dist-tutorials.yml` uses the
60-second signal timeout for its distribution tests. Tests that legitimately
need longer set their own `@pytest.mark.timeout(...)` override. On multi-chip
hosts, `compile_only` and `multi_device` tests execute serially while
single-device tests execute in parallel with one worker per chip.

#### Exabox Galaxy tests

`call-build.yml` runs N150 tests through `call-test-hardware.yml` and Galaxy
tests through `call-test-exabox.yml`. The Exabox job uses
`exabox-multihost-ci-sc1`, which provisions a Galaxy allocation from the IRD
image while Actions steps execute in a CPU-side container. The checkout is
staged on `/ci`, copied to `/home/user/tt-lang` on every worker host, and reset,
build, and test commands execute through `mpirun --pernode --bind-to none` so
OpenMPI does not restrict each worker shell and its child processes to one core.

`manual-test-exabox.yml` builds and publishes the current revision's IRD image,
then dispatches the same reusable workflow with the resulting tag. This keeps
the worker toolchain and Exabox services aligned with the tested source.

Exabox workers and `/ci` storage have job lifetime. The CPU-side Actions
container restores ccache state under shared `/ci` storage, and the staging
script makes the job-scoped cache writable by controller and worker UIDs. The
reusable workflow defaults to a 90-minute timeout.

The Galaxy job runs Python lit, simulator, and device pytest execution with 32
parallel workers, matching the Galaxy chip count without exceeding the 64-thread
host allocation. Device pytests retain the shared 300-second timeout.

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
| Main push (non-uplift)                 | n/a (image exists)   | n/a                            | yes (`retag-latest`)         |
| Tag push (release, via publish-pypi)   | yes                  | n/a                            | no                           |
| `workflow_dispatch`                    | only if `push: true` | only if `push: true`           | only if `push: true` on main |

`build-docker` pushes `:latest`, but runs only when the probe reports the image
missing. For an image published before the main push -- by an uplift PR's run or
a release tag's `publish-pypi.yml` run -- the `retag-latest` job moves `:latest`
onto the resolved tag instead, copying the manifest with
`docker buildx imagetools create` rather than rebuilding.

For a final release (see [Tagging after an uplift
merge](#tagging-after-an-uplift-merge) before tagging an uplift):

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

For a dated dev release (preview of an in-flight version, typically used after a
toolchain uplift lands on `main` and before the next final tag), follow the
tt-metal convention: a hyphenated development identifier of the form
`-dev<YYYYMMDD>`:

```bash
git tag v<MAJOR.MINOR.PATCH>-dev<YYYYMMDD>
git push origin v<MAJOR.MINOR.PATCH>-dev<YYYYMMDD>
```

SemVer orders `vX.Y.Z-dev<date>` strictly below `vX.Y.Z` (final), so users
who pin to `vX.Y.Z` are unaffected by dev releases. Within a single
`vX.Y.Z` line, dev tags order monotonically by date. The form is
Docker-tag-safe directly (no `+` translation needed). `-rc<N>` works the
same way (a release candidate of `vX.Y.Z` is tagged `vX.Y.Z-rc<N>`).

Legacy `<TAG>+<local>` build-metadata tags are still translated to
`<TAG>-<local>` by `get-version-tag.sh` for image-tag compatibility, but
SemVer treats `+`-suffixed tags as equal in precedence to the base tag, so
they cannot be distinguished by `pip install`. Prefer `-dev<YYYYMMDD>` or
`-rc<N>` for new tags.

(publishing-to-pypi)=
#### Publishing to PyPI

`publish-pypi.yml` prepares release images and wheels on release-tag pushes,
but PyPI upload requires a manual dispatch from `refs/heads/main`. Manual
dispatch takes the full SHA of the release-tagged commit and an optional
manylinux wheel-builder tag. The selected commit must be an ancestor of the
dispatching main commit and must have exactly one supported public release tag
(`vX.Y.Z` or `vX.Y.Z-<prerelease>`). Local-version tags containing `+` are
rejected.

```text
   push release tag
 or workflow_dispatch
          |
          +---------------------------+
          |                           v
          |                    +--------------+
          v                    | build-docker |   call-build-docker.yml
   +--------------+            +--------------+   (release-tag pushes only;
   |  preflight   |                                publishes the dist and IRD
   +--------------+                                images under the release tag;
          |                                        independent of preflight so
          |   verify the selected source            an S3-only pin still gets
          |   and release tag (release              its image)
          |   checks skipped if dry_run=true)
          v
   +--------------------+
   | build-wheel-images |   call-build-wheel-images.yml
   +--------------------+   (skipped if docker_tag input is set;
          |                  reuses separate LLVM and tt-metal caches)
          v
   +--------------+
   | build-wheels |   call-build-manylinux-wheels.yml
   +--------------+   (builds cp310/cp312 public wheels and tt-lang-sim,
          |            uploads tt-lang-wheels artifact)
          |
          +-----------------------+
          v                       |
   +-------------+                |
   | test-wheels |                |  (also runs for dry runs)
   +-------------+                |
          | workflow_dispatch     |
          | with dry_run=false    |
          +-----------------------+
          v                       v
   +--------------+        +------------------+
   |   publish    |        | dry-run-summary  |
   +--------------+        +------------------+
   workflow_dispatch        workflow_dispatch
   from main with           with dry_run=true
   dry_run=false
   (uploads to PyPI)        (lists artifacts only)

   Release-tag pushes complete after build-docker and test-wheels.
```

Job-by-job:

1. **`preflight`** — keeps the workflow source checkout separate from the
   selected release source. For manual publishing, it verifies that
   `ttlang_sha` is an ancestor of the dispatching main commit and resolves the
   release tag. Current workflow scripts validate that tag and the selected
   source's `third-party/tt-metal-version`. Release checks are skipped under
   `dry_run: true`. Exposes `tag_version` for the wheel-version check.
2. **`build-wheel-images`** — calls `call-build-wheel-images.yml` when no
   `docker_tag` is supplied. The multi-stage `manylinux_2_34` build stores LLVM
   caches separately for Python 3.10 and 3.12 and stores tt-metal in a third
   cache. Unchanged component inputs restore from GHCR instead of recompiling.
3. **`build-docker`** — calls `call-build-docker.yml` with `push: true` on a
   release-tag push, publishing the dist and IRD images under the release tag.
   It passes no source override, so the images are built from the tagged commit
   that `github.sha` already points at. Manual PyPI dispatches skip it, because
   the release images belong to the tag rather than to the dispatch. This job
   is what `ci.yml`'s `resolve-docker-tag` probe depends on: without it, a
   release tag resolves to an IRD image that was never published.

   It declares no `needs`, so it starts immediately and never waits on
   `preflight`. That independence is required, not incidental: `preflight`
   decides whether the release is publishable to public PyPI, and an S3-only
   pin fails that check by design (see [Two-phase
   uplift](#two-phase-uplift-publishable-release-then-latest-s3-only)). A
   dependent job is skipped when its `needs` fail, so gating image publication
   on `preflight` would leave every S3-only release without the image `ci.yml`
   requires. The image tag comes from `get-version-tag.sh` at checkout rather
   than from `preflight`, and `on.push.tags` already restricts the event to
   release tags.
4. **`build-wheels`** — calls `call-build-manylinux-wheels.yml` against either
   the `docker_tag` input or the image-build output. It builds Python 3.10 and
   3.12 `tt-lang` wheels with an exact public `ttnn` dependency and builds the
   ABI-independent `tt-lang-sim` wheel. The workflow verifies wheel names,
   versions, dependency metadata, and the `manylinux_2_34` platform tag before
   uploading the `tt-lang-wheels` artifact.
5. **`test-wheels`**: installs the Python 3.12 public wheel and its PyPI `ttnn`
   dependency in an isolated environment on an `n150` runner, installs the sfpi
   release recorded by `ttnn`, then runs the smoke test and tutorials. It runs
   during dry runs and must pass before `publish`.
6. **`publish`**: runs only for a manual dispatch from `main` when `dry_run` is
   false and `test-wheels` succeeded. Downloads the artifact, verifies
   every wheel filename's version field matches `preflight.outputs.tag_version`,
   and uploads via `pypa/gh-action-pypi-publish` using OIDC trusted publishing
   (`environment: pypi`, `id-token: write`).
7. **`dry-run-summary`**: runs only on `workflow_dispatch` with
   `dry_run: true`. Downloads the artifact and lists what would have been
   uploaded. No `environment`, no PyPI credentials.

Common scenarios (`<TAG>` denotes a release tag, `<SHA>` its full commit SHA,
and `<DOCKER_TAG>` an existing manylinux wheel-builder tag):

| Trigger                                                        | docker_tag input | Result                                                              |
| -------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------- |
| `git push origin <TAG>`                                        | (n/a)            | Build and test the release images and wheels; do not upload to PyPI |
| Dispatch from `main` with `ttlang_sha: <SHA>`                   | optional         | Build the tagged commit and publish its version to PyPI             |
| Dispatch with `ttlang_sha: <SHA>` and `dry_run: true`           | optional         | Build and test the selected commit; skip PyPI upload                |
| Non-main dispatch with `ttlang_sha: <SHA>` and `dry_run: false` | optional         | Fail at `preflight`                                                 |

(publishing-to-s3-pypi)=
#### Publishing to S3 PyPI

`publish-s3-pypi.yml` publishes S3-hosted wheels to the Tenstorrent S3 PyPI
index at `https://pypi.eng.aws.tenstorrent.com/`. It runs weekly at 08:00 UTC
on Monday (00:00 PST / 01:00 PDT) and can also be dispatched manually.
Publishing is restricted to
workflow runs on `refs/heads/main` because the AWS OIDC role is limited to
main-branch refs; a manual dispatch from another ref can only perform a dry run.
The workflow uses GitHub OIDC for AWS access. Regular publishes upload wheel
objects directly under `tt-lang/` and write generated slash-key views consumed
with `pip --find-links`: development wheels use `tt-lang/<YYYY-MM>/`, and final
S3 release wheels use `tt-lang/releases/`.
The top-level `tt-lang/` listing shows only the README and subdirectories in a
browser, but it keeps hidden anchors for final-release root wheels so
`pip --find-links https://pypi.eng.aws.tenstorrent.com/tt-lang` remains
backward-compatible for `X.Y.Z` S3 releases.
Non-main dry runs must provide an existing `docker_tag`. If `docker_tag` is
empty, the workflow builds only the builder images required by the selected
variants: the IRD image for bundled wheels and the shared manylinux images for
light or PyPI-style wheels. Image publication is also restricted to
`refs/heads/main`.

The workflow prevents publishing a bundled S3 `tt-lang` wheel with the
same package name and version as the public PyPI wheel when public PyPI
publishing is already valid for that tt-metal tag. This avoids having two
indexes expose `tt-lang==X.Y.Z` artifacts with different dependency metadata.

S3 publishing uses this policy:

- Tag pushes are not S3 publishing triggers. Stable S3 publishes use
  manual dispatch from `refs/heads/main` with an explicit `version_override`.
- Manual stable-version S3 publishes that include the bundled variant are
  rejected when public PyPI publishing is aligned for the same tt-metal tag.
- The S3 resolver passes `TTLANG_ALLOW_FINAL_INTERNAL_VERSION=true` to the wheel
  builder only after this conflict check has passed, so final-version S3
  wheels cannot bypass the release guard.
- Do not mix public PyPI and S3 indexes for a `tt-lang` version whose artifacts
  have different dependency semantics. Use the S3 install command emitted by the
  workflow summary for S3 release wheels.
- Scheduled builds do not create Git tags. The workflow computes a
  PEP 440 development version of the form `<MAJOR.MINOR.PATCH>.dev<YYYYMMDD>`,
  where the base version matches the latest stable tag reachable from `HEAD`,
  and the numeric suffix is a UTC date.
- Before building, a scheduled run compares the selected source SHA with the
  marker written by the last successful scheduled publish. An equal SHA skips
  all image, wheel, publish, and per-tt-metal work. A changed SHA publishes and
  updates the marker only after the wheel objects and index are complete.
- Scheduled reruns after a source update overwrite the same date-based version
  in the S3 index. Existing local pip caches may still hold an older wheel for
  that version.

Manual stable-version publishes set `version_override` explicitly, build any
missing selected builder images when `docker_tag` is empty, verify each wheel
set, and publish the result to S3 PyPI.

The scheduled workflow defaults to `wheel_variant: bundled-and-light`. It keeps
building the complete bundled wheel from the IRD image, and also builds and
pushes the matching manylinux_2_34 wheel-builder images for Python 3.10 and
Python 3.12 light wheels. The manylinux images use separate BuildKit registry
caches for the two LLVM/Python combinations and for tt-metal, so unchanged
components are restored instead of rebuilt. The workflow verifies all wheel
versions before publishing the combined result to S3 PyPI.

For a manual bundled S3 wheel with an existing IRD image, dispatch the
workflow with:

```text
docker_tag: <existing-ird-tag>
wheel_variant: bundled
version_override: <s3-version>
```

The reusable wheel build sets `TTLANG_TTNN_DEP_MODE=bundled`,
`TTLANG_VERSION_OVERRIDE=<version_override>`, and
`TTLANG_BUNDLED_TT_METAL_DIR=/opt/ttlang-toolchain/tt-metal`. The resulting
`tt-lang` wheel includes the `ttnn` Python package, its native extensions, the
needed shared libraries, and the runtime/header payload copied from the
toolchain's tt-metal install.

For light wheels that must use a user-provided tt-metal build instead of a
bundled or public `ttnn` wheel, dispatch the workflow with:

```text
wheel_variant: light
version_override: <s3-version>
```

The workflow maps this publish selection to the manylinux_2_34 light wheel
builder. It emits `cp310` and `cp312` `tt-lang==<version_override>+light`
wheels. Those wheels omit `Requires-Dist: ttnn`; the normal PyPI build keeps
that requirement. The same build also emits
`tt-lang-light==<version_override>`, a metapackage that depends on
`tt-lang==<version_override>+light`.

The `pypi` selection uses the same manylinux_2_34 base and build process but
retains public package semantics: `tt-lang==<version_override>` has no `+light`
label, requires the exact `ttnn` version from `third-party/tt-metal-version`,
and includes `tt-lang-sim`. It does not emit the `tt-lang-light` metapackage.

To publish bundled and light wheels from the same workflow run, dispatch with:

```text
wheel_variant: bundled-and-light
version_override: <s3-version>
```

The workflow builds the bundled and light wheel sets separately, uploads
mode-specific artifacts, verifies each artifact with the expected version rules,
then publishes a single combined directory. The light build does not emit
`tt-lang-sim`; bundled/public wheel builds keep producing the simulator wheel.

To rebuild a light wheel from a specific older tt-lang commit or tag -- for
example to reissue a release line whose original wheel no longer runs -- dispatch
the workflow with a pinned ref, adding wheel patches when that ref's pinned
dependencies no longer resolve:

```text
wheel_variant: light
version_override: <s3-version>
ttlang_ref: <tt-lang SHA or tag>
apply_patches: true
```

`ttlang_ref` checks out that commit or tag for every wheel-building job instead
of the triggering commit. `apply_patches` runs the scripts in
`.github/wheel-patches/` against the checkout before building; the first patch
rewrites a stale `numpy` pin to the current line's constraint so an older ref
installs cleanly. The patch runner and the patches are taken from the workflow
commit, not the pinned ref, so a ref that predates them is still patched.

#### Building a light wheel for a specific tt-metal SHA

`ttmetal-light-on-demand.yml` builds and device-tests a light wheel against an
arbitrary tt-metal commit. Dispatch it with a `tt_metal_sha`; leave `ttlang_ref`
empty to search for the newest compatible tt-lang commit, or set it to pin the
tt-lang commit or tag to build. A pinned `ttlang_ref` requires `tt_metal_sha`,
because auto-detection reads the dispatch ref's `third-party/tt-metal-version`
rather than the pinned ref's. With `dry_run: true` the workflow builds and
validates without publishing and needs no S3 credentials, so it can run from a
feature branch; the scheduled per-tt-metal-SHA build in `publish-s3-pypi.yml`
is best-effort and does not fail the scheduled publish.

Successful per-SHA publishes place the wheel files (both tt-lang and
tt-lang-light) under `https://pypi.eng.aws.tenstorrent.com/tt-lang/ttmetal/<ttmetal7>/`
as a browsable directory (the listing is written to the slash-key so the
directory URL resolves; trailing slash required). Install from that directory
with `--find-links`. Browse all published SHAs at
`https://pypi.eng.aws.tenstorrent.com/tt-lang/ttmetal/`.

The `tt-lang-light` wheel is a pure metapackage; the supported CPython ABIs and
glibc floor are carried by the
`tt_lang-<version>+light-cp310-cp310-manylinux_2_34_x86_64.whl` and
`tt_lang-<version>+light-cp312-cp312-manylinux_2_34_x86_64.whl` files in the
same directory. The same directory contains a brief `README.html`.
The `ttl.build_info()["tt_metal"]` value in each `tt-lang` wheel must equal
`<ttmetal7>` expanded to the requested tt-metal commit.

`ttmetal-light-xla-on-demand.yml` builds standard `tt-lang-light` wheels for the
XLA flow from a specific tt-lang ref and tt-metal SHA. The selected tt-lang ref
must include the light-wheel packaging and `ttl.build_info()` provenance support,
because the workflow verifies that the built wheel records the requested
tt-metal commit. Dispatch it with:

```text
ttlang_ref: <tt-lang SHA or tag>
tt_metal_sha: <tt-metal SHA>
```

Leave `docker_tag` empty to resolve an existing `tt-lang-ird-ubuntu-24-04` image
from the pinned `ttlang_ref`. The resolver first tries that checkout's
`get-version-tag.sh` output exactly. If the exact tag is missing and the
computed tag is a bare release tag such as `v1.1.2`, it uses a single existing
`v1.1.2-*` image tag. Multiple matching image tags are ambiguous and require an
explicit `docker_tag`. Leave `version_override` empty to compute the wheel
version from the pinned `ttlang_ref`; the resolver runs that checkout's
`compute-nightly-version.py` so the build uses the selected tt-lang ref's
versioning rules. Set `apply_patches: true` when the pinned ref needs the
workflow commit's wheel patches. Set `hw_type` to select a device validation
runner type; it defaults to `n150`.

The XLA workflow uses the Ubuntu IRD image directly, builds the requested
tt-metal SHA with `.github/scripts/build-ttmetal-at-sha.sh`, and builds the
tt-lang wheel in `TTLANG_TTNN_DEP_MODE=external` mode. The wheels keep the
standard package names and versions: `tt-lang==<version>+light` and
`tt-lang-light==<version>`. The XLA distinction is the index location, not a
wheel suffix: the build places both wheels under `dist/xla/<ttmetal7>/` and
uploads that wheel set as the `tt-lang-light-xla-wheels` artifact. It does not
use the manylinux_2_34 light-wheel builder or publish to S3.

The workflow device-validates the uploaded wheels in a separate hardware job.
The build job uploads the `tt_metal_sha` install as a tar artifact so executable
bits are preserved. The device job installs the downloaded `tt-lang-light` wheel
with that external tt-metal environment, then runs `test/python/smoketest.py`
and the tutorial suite.

#### S3 PyPI maintenance

`s3-pypi-ops.yml` is a manual (`workflow_dispatch`) workflow for maintaining the
tt-lang prefixes of `tenstorrent-pypi`: `inspect`, `put-index` (refresh a
slash-key listing), `move`, `copy`, `delete`, and a read-only `readonly-cmd`.
Writes require `refs/heads/main`; `dry_run` defaults to true. Operations are
restricted to the `tt-lang/` prefix and cannot touch other teams' packages
(including the sibling `tt-lang-light/` and `tt-lang-sim/` package indexes) or
the bucket root. `delete` requires a `confirm` token equal to the prefix.

`s3-wheel-maintenance.yml` provides wheel-specific storage maintenance.
`deduplicate` removes older object versions only when the key, size, and ETag
match a newer retained version. It never deletes the logical wheel key.
`remove-dev-range` permanently removes every object version and delete marker
for top-level `tt-lang` wheels whose `.devYYYYMMDD` version date is within the
inclusive `start_date` and `end_date`; per-tt-metal wheel directories are
excluded. A live date removal regenerates affected month views and the root
index.

Both operations require `refs/heads/main` and default to `dry_run: true`. Live
deduplication requires `confirm: delete-duplicate-versions`; live date removal
requires `confirm: delete-dev-versions`.

#### Local S3 wheel testing

Use the same environment variables as the reusable workflow when validating the
wheel build locally.

Bundled wheel:

```bash
source /opt/ttlang-toolchain/venv/bin/activate
TTLANG_VERSION=<s3-version>

TTLANG_VERSION_OVERRIDE="$TTLANG_VERSION" \
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DTTLANG_USE_TOOLCHAIN=ON

TTLANG_TTNN_DEP_MODE=bundled \
TTLANG_VERSION_OVERRIDE="$TTLANG_VERSION" \
TTLANG_BUNDLED_TT_METAL_DIR=/opt/ttlang-toolchain/tt-metal \
pip wheel . --wheel-dir=/tmp/ttlang-wheels/bundled/raw --no-deps --no-build-isolation

auditwheel repair /tmp/ttlang-wheels/bundled/raw/tt_lang-*.whl \
  --wheel-dir=/tmp/ttlang-wheels/bundled/dist
```

Light wheels:

```bash
scripts/build-s3-light-wheels-local.sh \
  --version <s3-version> \
  --build-images
```

Install-test the light package from the local wheel directory. The setup command
copies tutorials into `./tutorials/` and skips sfpi installation for light
installs because the external tt-metal tree provides sfpi:

```bash
TTLANG_VERSION=<s3-version>
python3.12 -m venv /tmp/ttlang-light-test
source /tmp/ttlang-light-test/bin/activate
pip install --find-links=/tmp/ttlang-s3-light-wheels/dist \
  "tt-lang-light==$TTLANG_VERSION"
tt-lang-setup
```

Configure the external tt-metal environment and validate imports:

```bash
external_tt_metal_env="$(
  tt-lang-setup-external-tt-metal \
    --tt-metal-dir /opt/ttlang-toolchain/tt-metal \
    --check
)" && eval "$external_tt_metal_env"
python -c 'import ttl, ttnn; print(ttl.__version__, ttnn.__file__)'
```

The TT-Lang tutorial can then run from the local directory copied by
`tt-lang-setup`:

```bash
python tutorials/elementwise/step_4_multinode_grid_full.py
```

## CMake Options

| Option                             | Default     | Description                                                                          |
| ---------------------------------- | ----------- | ------------------------------------------------------------------------------------ |
| `CMAKE_BUILD_TYPE`               | `Release` | Build type (Debug, Release, RelWithDebInfo)                                          |
| `LLVM_BUILD_TYPE`                | `Release` | LLVM build type (independent of project build type)                                  |
| `TTLANG_TOOLCHAIN_DIR`           | —          | Toolchain prefix for LLVM, tt-metal, and venv                                        |
| `TTLANG_PYTHON_VENV`            | —          | Existing Python virtual environment used by configure/build                          |
| `TTLANG_USE_TOOLCHAIN`           | `OFF`     | Use pre-built toolchain at `TTLANG_TOOLCHAIN_DIR`                                  |
| `TTLANG_USE_TOOLCHAIN_TTMETAL`   | follows `TTLANG_USE_TOOLCHAIN` | Reuse tt-metal from the toolchain. Set `OFF` (e.g. via `scripts/build-and-install.sh --rebuild-ttmetal`) to keep LLVM from the toolchain but rebuild tt-metal from the submodule. |
| `TTLANG_BUILD_TOOLCHAIN`         | `OFF`     | Build LLVM and tt-metal into a reusable toolchain directory (cleans stale artifacts) |
| `TTLANG_EXTERNAL_TT_METAL_DIR`    | —          | Existing tt-metal source or install directory                                        |
| `TTLANG_EXTERNAL_TT_METAL_BUILD_DIR` | —       | Existing native tt-metal build directory                                             |
| `MLIR_PREFIX`                    | —          | Path to pre-built LLVM/MLIR install                                                  |
| `TTLANG_ACCEPT_LLVM_MISMATCH`    | `OFF`     | Allow LLVM SHA mismatch with pre-built installs                                      |
| `TTLANG_ENABLE_PERF_TRACE`       | `ON`      | Enable tt-metal performance tracing support                                          |
| `TTLANG_SIM_ONLY`                | `OFF`     | Set up Python environment for [simulator](simulator.md) only; skip compiler build       |
| `TTLANG_ENABLE_DOCS`             | `OFF`     | Enable Sphinx documentation build (`ttlang-docs` target)                           |
| `CODE_COVERAGE`                  | `OFF`     | Enable code coverage reporting                                                       |
| `TTLANG_FORCE_TOOLCHAIN_REBUILD` | `OFF`     | Force rebuild of LLVM and tt-metal into `TTLANG_TOOLCHAIN_DIR`                     |

## Build Architecture

### Dialects, conversion, and translation

tt-lang owns the `ttcore` and `ttkernel` dialects, the `TTKernelToEmitC`
conversion, and the `TTKernelToCpp` translation in its own tree
(`lib/Dialect/{TTCore,TTKernel}`, `lib/Conversion/TTKernelToEmitC`,
`lib/Target/TTKernel`), compiled by the normal `add_subdirectory(include)` /
`add_subdirectory(lib)` MLIR CMake tree wired by
`cmake/modules/BuildTTLangDialects.cmake`. The system-descriptor flatbuffer
loader is compiled out (`TTLANG_NO_FLATBUFFERS`).

### tt-metal runtime

`cmake/modules/BuildTTMetal.cmake` builds tt-metal at configure time via
`execute_process`. Post-build, `_ttnn.so` and `_ttnncpp.so` are copied so
`import ttnn` works after activating the environment.

### Python bindings

A single nanobind extension (`_ttlang`) exposes the `ttl`, `ttcore`, and
`ttkernel` dialects — submodules `ttl_ir`, `tt_ir`, `ttkernel_ir`, and `passes`.
The Python package prefix is `ttl.`.

Site initialization registers every dialect on context creation:
`_mlir_libs/_site_initialize_0.py` calls `_ttlang.register_dialects`, which
registers `ttcore`, `ttkernel`, `ttl`, and the minimal upstream MLIR dialects
the pipeline uses (in place of MLIR's RegisterEverything).

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
was built from a different commit than what tt-lang expects. Either rebuild LLVM
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
