# tt-lang Docker Containers

This directory contains Dockerfiles for building tt-lang container images.

## Images

### `tt-lang-base-ubuntu-22-04`
Base image extending `tt-mlir-base-ubuntu-22-04` with tt-lang Python
dependencies (pydantic, torch, numpy, pytest). Small and fast to build; serves
as the filesystem base for `dist` and `ird`.

### `tt-lang-dist-ubuntu-22-04`
Distribution image for end users with pre-built tt-lang, ready to `import ttl`.

**Contents:** tt-mlir toolchain + installed tt-lang + examples + SSH + text
editors

### `tt-lang-ird-ubuntu-22-04`
Interactive Research & Development image. Contains the tt-mlir toolchain but
*not* tt-lang — developers clone and build tt-lang themselves.

**Contents:** tt-mlir toolchain + dev tools (ssh, sudo, tmux, vim, black,
sphinx)

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
tt-mlir-base-ubuntu-22-04 (upstream)     tt-mlir-ci-ubuntu-22-04 (upstream)
             |                                    |
             v                           +--------+--------+
  tt-lang-base-ubuntu-22-04              |                 |
      (Python deps)            build-toolchain           build
             |                (configure only;      (full tt-lang
             |                 installs tt-mlir)     build+install)
             |                        |                    |
             +----------+-------------+                    |
             |          |                                  |
            ird        ...                                 |
     (toolchain only,                                      |
      + dev tools)                                         |
             +---------------------------------------------+
             |
            dist
     (full tt-lang)
```

`dist` and `ird` use separate build stages. `build-toolchain` only runs cmake
configure (which builds tt-mlir via FetchContent) without building tt-lang.
`build` does the full configure + build + install. Docker only executes stages
in the dependency chain of the requested target, so `--target ird` never builds
tt-lang and `--target dist` never runs `build-toolchain`.

## CI Job Flow

Each large-runner job builds a single Dockerfile target on a fresh runner with
its own Docker daemon. This prevents layer cache accumulation across targets,
which was the cause of disk exhaustion when all targets built on one runner.

```
check-if-images-already-exist (ubuntu-latest)
  └─ if all images exist: all build jobs skipped, outputs existing image names
  └─ if any missing: sets docker-image='' to trigger builds

                        ↓
                build-image-base (ubuntu-latest)
                  docker build Dockerfile.base
                  push tt-lang-base-ubuntu-22-04:$TAG
                        ↓
         ┌──────────────┴──────────────────────┐
         ↓                                     ↓
build-image-ird                        build-image-dist
(mlir-large-runner-lang)               (mlir-large-runner-lang)
  FRESH runner + Docker daemon           SEPARATE fresh runner + Docker daemon
  docker build --target ird              docker build --target dist
    build-toolchain + ird stages           build + dist stages
  push tt-lang-ird-ubuntu-22-04:$TAG    push tt-lang-dist-ubuntu-22-04:$TAG
         └──────────────┬────────────────────┘
                        ↓ (on push to main only)
                  set-latest-tag
                    skopeo copy :$TAG → :latest for base, dist, ird
```

## Image Sizes (Approximate)

- `tt-lang-base`: ~1.7 GB
- `tt-lang-dist`: ~6–7 GB (tt-mlir + tt-lang)
- `tt-lang-ird`: ~5–6 GB (tt-mlir + dev tools)

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

- `Dockerfile.base` — base image with Python dependencies
- `Dockerfile` — multi-stage build (`ird` and `dist` targets, with separate build stages)
- `build-and-install.sh` — cmake configure/build/install; `--toolchain-only` skips tt-lang build
- `entrypoint.sh` — activates tt-lang environment on container start
- `activate-install.sh` — environment activation for installed tt-lang (used in containers)
- `build-docker-images.sh` — build/push script with `--image-type` filter
- `build-docker-local.sh` — build all images locally for testing
- `cleanup-toolchain.sh` — normalizes toolchain venv (e.g. lib64 symlink fix)
- `test-docker-smoke.sh` — quick smoke test for container functionality
- `CONTAINER_README.md` — welcome message shown inside the container
