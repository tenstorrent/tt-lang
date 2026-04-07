# tt-lang Docker Containers

This directory contains Dockerfiles for building tt-lang container images.

## Images

### `tt-lang-base-ubuntu-22-04`
Standalone base image built from `ubuntu:22.04` with Python 3.12, Clang 18,
system libraries, and tt-lang Python dependencies (pydantic, torch, numpy,
pytest). Small and fast to build; serves as the filesystem base for `dist`
and `ird`.

### `tt-lang-dist-ubuntu-22-04`
Distribution image for end users with pre-built tt-lang, ready to `import ttl`.

**Contents:** LLVM + tt-metal toolchain + installed tt-lang + examples + SSH +
text editors

### `tt-lang-ird-ubuntu-22-04`
Interactive Research & Development image. Contains the toolchain but *not*
tt-lang -- developers clone and build tt-lang themselves.

**Contents:** LLVM + tt-metal toolchain + dev tools (ssh, sudo, tmux, vim,
black, sphinx)

## Building Images Locally

Images must be built from the repository root. The build script requires
`docker` access (use `sudo` or configure Docker for rootless mode). All
commands below assume you are in the repo root.

### Prerequisites

- Docker installed
- For ird/dist: a built toolchain directory (see [Build Integration](../../docs/sphinx/build.md)).
  Build one with `cmake -G Ninja -B build -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain .`
  followed by `cmake --build build`.

### Quick start: build all images

```bash
sudo .github/containers/build-docker-images.sh --no-push
```

This builds base → ird → dist in order. The `--no-push` flag keeps images
local-only (no `ghcr.io/` tags) so they don't shadow registry images.

### Build a single image

The `ird` and `dist` images require a pre-built toolchain directory passed
via `DOCKER_BUILD_EXTRA_ARGS`. Replace `/opt/ttlang-toolchain` with your
toolchain path (produced by `cmake -DTTLANG_TOOLCHAIN_DIR=...`):

```bash
# IRD: toolchain only (for developers)
sudo DOCKER_BUILD_EXTRA_ARGS="--build-context ird-toolchain=/opt/ttlang-toolchain" .github/containers/build-docker-images.sh --no-push --image-type ird

# Dist: full tt-lang install (for end users)
sudo DOCKER_BUILD_EXTRA_ARGS="--build-context dist-toolchain=/opt/ttlang-toolchain" .github/containers/build-docker-images.sh --no-push --image-type dist
```

The base image is used automatically from the registry if not built locally.
To build it locally instead:
```bash
sudo .github/containers/build-docker-images.sh --image-type base --no-push
```

### How it works

The `build-docker-images.sh` script:
1. Determines the Docker tag from `git describe` (e.g. `v0.1.6-47-g0ad37bac`)
2. Runs `docker build` for each image type with appropriate tags
3. For `ird`/`dist`, the Dockerfile COPYs the toolchain from a build context

The toolchain (LLVM + tt-metal + Python venv) must be built separately
before building `ird`/`dist` images. On CI this is done by
`scripts/build-and-install.sh`; locally you build it with
`cmake -DTTLANG_TOOLCHAIN_DIR=/path/to/prefix` and then pass the path
via `DOCKER_BUILD_EXTRA_ARGS` as shown above.

The Dockerfile declares placeholder stages (`FROM scratch AS ird-toolchain`)
that are overridden at build time by `--build-context ird-toolchain=/path`.
Without this override, the COPY gets an empty context and the image will be
missing the toolchain.

### Build options

| Flag | Effect |
|---|---|
| `--no-push` | Local tags only; don't push to registry or create `ghcr.io/` tags |
| `--no-cache` | Rebuild from scratch (no Docker layer cache) |
| `--image-type TYPE` | Build only `base`, `dist`, or `ird` |
| `--check-only` | Check if images exist in registry without building |

### Using a locally built image

After building, use the image with `docker-test.sh`:
```bash
DOCKER_IMAGE=tt-lang-ird-ubuntu-22-04:<version-tag> scripts/docker-test.sh mlir
```

Or run interactively:
```bash
sudo docker run -it --rm --device=/dev/tenstorrent/0:/dev/tenstorrent/0 -v /dev/hugepages:/dev/hugepages -v /dev/hugepages-1G:/dev/hugepages-1G tt-lang-ird-ubuntu-22-04:<version-tag> bash
```

## Image Architecture

```
ubuntu:22.04
     |
     v
tt-lang-base-ubuntu-22-04
  (Python 3.12, clang, system libs, Python deps)
     |
     +---------------------+
     |                     |
    ird                   dist
 (toolchain only,      (full tt-lang
  + dev tools)          build+install)
```

Toolchain building (LLVM + tt-metal) happens outside Docker on CI runners.
The pre-built toolchains are injected into the Dockerfile via `--build-context`
arguments. The Dockerfile itself is purely a packaging step -- it COPYs the
pre-built toolchains into `ird` and `dist` images. The base image is
parameterized via `ARG BASE_IMAGE` so local builds resolve against local tags.

## CI Job Flow

```
check-if-images-already-exist (ubuntu-latest)
  |-- if all images exist: all build jobs skipped, outputs existing image names
  |-- if any missing: sets docker-image='' to trigger builds

                        |
                build-images (ubuntu-22.04)
                  1. Build base image (Dockerfile.base)
                  2. Build toolchains (LLVM + tt-metal) on host
                  3. docker build --target ird (with --build-context)
                  4. docker build --target dist (with --build-context)
                  5. Push all images
```

## Docker Testing (Local)

Run tests inside a Docker container using a host build without rebuilding:

```bash
scripts/docker-test.sh all                       # all test suites
scripts/docker-test.sh mlir                      # MLIR lit tests only
scripts/docker-test.sh -- pytest test/me2e/ -k test_name  # arbitrary command
```

See `scripts/docker-test.sh --help` for options (`--build-dir`, `--image`).

## Hardware Access

To access Tenstorrent hardware from containers:

```bash
docker run -it \
    --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    <image> <command>
```

## Image Sizes (Approximate)

- `tt-lang-base`: ~1.7 GB
- `tt-lang-dist`: ~6-7 GB (LLVM + tt-metal + tt-lang)
- `tt-lang-ird`: ~5-6 GB (LLVM + tt-metal + dev tools)

## Files

- `Dockerfile.base` -- base image from ubuntu:22.04 with Python and system deps
- `Dockerfile` -- multi-stage build (`ird` and `dist` targets, with separate build stages)
- `scripts/build-and-install.sh` -- cmake configure/build/install with mode flags (`--toolchain-only`, `--force-rebuild`, `--test-toolchain`, etc. Used by CI and local toolchain builds. See '--help' for usage.)
- `entrypoint.sh` -- activates tt-lang environment on container start
- `activate-install.sh` -- environment activation for installed tt-lang (used in containers)
- `build-docker-images.sh` -- build/push script with `--image-type` filter
- `cleanup-toolchain.sh` -- normalizes toolchain venv (lib64 symlink fix), strips LLVM binaries, and optionally removes headers/static libs for dist
- `get-docker-tag.sh` -- generates deterministic Docker tags from submodule SHAs and file hashes
- `test-docker-smoke.sh` -- quick smoke test for container functionality
- `CONTAINER_README.md` -- welcome message shown inside dist container
- `IRD_README.md` -- welcome message shown inside IRD container
- `../scripts/normalize-toolchain-install.sh` -- replaces symlinks with actual files for portable toolchain installs
