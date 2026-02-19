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
             |                                         |
             v                                         v
  tt-lang-base-ubuntu-22-04                       build stage
      (Python deps)                            (compiles tt-lang)
             |                                         |
             +--------------------+--------------------+
                                  |
                       +----------+----------+
                       |                     |
                      dist                  ird
               (full tt-lang)       (toolchain only,
                                     + dev tools)
```

Both `dist` and `ird` are final stages in `Dockerfile` sharing the same `build`
stage. `dist` keeps all tt-lang artifacts; `ird` strips them and adds dev
tooling.

## CI Job Flow

Each large-runner job builds a single Dockerfile target on a fresh runner with
its own Docker daemon. This prevents layer cache accumulation across targets,
which was the cause of disk exhaustion when all targets built on one runner.

```
check-if-images-already-exist (ubuntu-latest)
  └─ if all images exist: all build jobs skipped, outputs existing image names
  └─ if any missing: sets docker-image='' to trigger builds

         ┌─────────────────────────────────────┐
         ↓                                     ↓
build-toolchain-if-needed              build-image-base
(ubuntu-latest)                        (ubuntu-latest)
  build/restore LLVM toolchain           docker build Dockerfile.base
  populate actions cache                 push tt-lang-base-ubuntu-22-04:$TAG
         └──────────────┬────────────────────┘
                        ↓ (both complete)
         ┌──────────────┴──────────────────────┐
         ↓                                     ↓
build-image-dist                       build-image-ird
(mlir-large-runner-lang)               (mlir-large-runner-lang)
  FRESH runner + Docker daemon           SEPARATE fresh runner + Docker daemon
  docker build --target dist             docker build --target ird
    build stage + dist stage only          build stage + ird stage only
  push tt-lang-dist-ubuntu-22-04:$TAG    push tt-lang-ird-ubuntu-22-04:$TAG
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
- `Dockerfile` — multi-stage build (`dist` and `ird` targets)
- `entrypoint.sh` — activates tt-lang environment on container start
- `build-docker-images.sh` — build/push script with `--image-type` filter
- `cleanup-toolchain.sh` — removes unused LLVM tools to reduce image size
- `CONTAINER_README.md` — welcome message shown inside the container
