# tt-lang Docker Containers

This directory contains Dockerfiles for building tt-lang container images.

## Images

### `tt-lang-base-ubuntu-22-04`
Standalone base image built from `ubuntu:22.04` with Python 3.12, LLVM
toolchain, system libraries, and tt-lang Python dependencies (pydantic, torch,
numpy, pytest). Small and fast to build; serves as the filesystem base for
`dist` and `ird`.

### `tt-lang-dist-ubuntu-22-04`
Distribution image for end users with pre-built tt-lang, ready to `import ttl`.

**Contents:** LLVM + tt-metal + tt-mlir toolchain + installed tt-lang + examples
+ SSH + text editors

### `tt-lang-ird-ubuntu-22-04`
Interactive Research & Development image. Contains the toolchain but *not*
tt-lang -- developers clone and build tt-lang themselves.

**Contents:** LLVM + tt-metal + tt-mlir toolchain + dev tools (ssh, sudo, tmux,
vim, black, sphinx)

## Build Scripts

### `build-docker-images.sh`
Orchestrates building images with proper tagging and optional registry push.

```bash
# Build all images locally (no push)
.github/containers/build-docker-images.sh --no-push

# Build a single image type
.github/containers/build-docker-images.sh --image-type base --no-push
.github/containers/build-docker-images.sh --image-type dist --no-push
.github/containers/build-docker-images.sh --image-type ird  --no-push

# Build and push to registry
.github/containers/build-docker-images.sh

# Check if images exist without building
.github/containers/build-docker-images.sh --check-only
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
arguments. The Dockerfile itself is purely a packaging step — it COPYs the
pre-built toolchains into `ird` and `dist` images.

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
                        |
                  set-latest-tag (on push to main only)
                    skopeo copy :$TAG -> :latest for base, dist, ird
```

## Image Sizes (Approximate)

- `tt-lang-base`: ~1.7 GB
- `tt-lang-dist`: ~6-7 GB (LLVM + tt-metal + tt-mlir + tt-lang)
- `tt-lang-ird`: ~5-6 GB (LLVM + tt-metal + tt-mlir + dev tools)

## Hardware Access

To access Tenstorrent hardware from containers:

```bash
docker run -it \
    --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    <image> <command>
```

## Files

- `Dockerfile.base` -- base image from ubuntu:22.04 with Python and system deps
- `Dockerfile` -- multi-stage build (`ird` and `dist` targets, with separate build stages)
- `build-and-install.sh` -- cmake configure/build/install; `--toolchain-only` skips tt-lang build
- `entrypoint.sh` -- activates tt-lang environment on container start
- `activate-install.sh` -- environment activation for installed tt-lang (used in containers)
- `build-docker-images.sh` -- build/push script with `--image-type` filter
- `cleanup-toolchain.sh` -- normalizes toolchain venv (lib64 symlink fix), strips LLVM binaries, and optionally removes headers/static libs for dist
- `get-docker-tag.sh` -- generates deterministic Docker tags from submodule SHAs and file hashes
- `test-docker-smoke.sh` -- quick smoke test for container functionality
- `CONTAINER_README.md` -- welcome message shown inside the container
