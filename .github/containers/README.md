# tt-lang Docker Containers

This directory contains Dockerfiles for building tt-lang container images.

## Images

### `tt-lang-base-ubuntu-22-04`
Base image that extends tt-mlir's base image with tt-lang Python dependencies (pydantic, torch, numpy, dev tools).

**Build:**
```bash
docker build --build-arg MLIR_TAG=latest \
    -t tt-lang-base:local \
    -f .github/containers/Dockerfile.base .
```

### `tt-lang-ci-ubuntu-22-04` (ci target) / `tt-lang-dist-ubuntu-22-04` (alias)
CI and distribution image for end users with pre-built tt-lang, ready to `import ttlang`.
The same image is tagged as both `ci` (for CI workflows) and `dist` (for clarity).

**Build:**
```bash
docker build --build-arg FROM_TAG=local --build-arg MLIR_TAG=latest \
    --target dist \
    -t tt-lang:local \
    -f .github/containers/Dockerfile.dist .
```

**Usage:**
```bash
# Run without hardware
docker run -it tt-lang:local python -c "import ttlang"

# Run with Tenstorrent hardware
docker run -it \
    --device /dev/tenstorrent \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    tt-lang:local python my_kernel.py
```

### `tt-lang-ird-ubuntu-22-04` (ird target)
Interactive Research & Development image with toolchain and debugging tools (gdb, vim, tmux) for building tt-lang from source.

**Build:**
```bash
docker build --build-arg FROM_TAG=local --build-arg MLIR_TAG=latest \
    --target dev \
    -t tt-lang-dev:local \
    -f .github/containers/Dockerfile.dist .
```

**Usage:**
```bash
# Interactive development
docker run -it -v $(pwd):/workspace tt-lang-dev:local

# Build tt-lang from source
docker run -it -v $(pwd):/workspace tt-lang-dev:local \
    bash -c "cd /workspace && cmake -GNinja -Bbuild && cmake --build build"
```

## Build Scripts

### `.github/scripts/build-docker-local.sh`
Build all images locally for testing.

```bash
.github/scripts/build-docker-local.sh
```

### `.github/scripts/build-docker-images.sh`
Orchestrates building all images with proper tagging and optional registry push.

```bash
# Build locally (no push)
.github/scripts/build-docker-images.sh --no-push

# Build and push to registry
.github/scripts/build-docker-images.sh

# Check if images exist without building
.github/scripts/build-docker-images.sh --check-only
```

### `.github/scripts/test-docker-smoke.sh`
Quick smoke test to verify container functionality.

```bash
.github/scripts/test-docker-smoke.sh
```

### `.github/scripts/get-docker-tag.sh`
Generates deterministic Docker tags from file hashes and tt-mlir version.

```bash
.github/scripts/get-docker-tag.sh <MLIR_DOCKER_TAG>
```

## Hardware Access

To access Tenstorrent hardware from containers, use:

```bash
docker run -it \
    --device /dev/tenstorrent \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    <image> <command>
```

To isolate a specific card:
```bash
# Map card 3 to card 0 inside container
docker run -it \
    --device=/dev/tenstorrent/3:/dev/tenstorrent/0 \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    <image> <command>
```

## Architecture

```
tt-mlir-base-ubuntu-22-04 (upstream)
         |
         v
tt-lang-base-ubuntu-22-04
    (adds Python deps)
         |
    +----+----+
    |         |
    v         v
  dist       dev
(pre-built) (toolchain + tools)
```

## Build Strategy

The dist/dev images use a multi-stage build:

1. **Build stage**: Uses tt-mlir CI image to build tt-lang via FetchContent (builds tt-mlir from pinned commit in `third-party/tt-mlir.commit`)
2. **Dist stage**: Copies pre-built tt-lang artifacts for immediate use
3. **Dev stage**: Copies tt-mlir toolchain and adds development tools

## Files

- `Dockerfile.base` - Base image with Python dependencies
- `Dockerfile.dist` - Multi-stage build (dist/dev targets)
- `entrypoint.sh` - Container entrypoint that activates environments
- `.dockerignore` - Excludes build directories from Docker context
