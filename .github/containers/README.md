# tt-lang Docker Containers

This directory contains Dockerfiles for building tt-lang container images.

## Images

### `tt-lang-base-ubuntu-22-04`
Base image that extends tt-mlir's base image with tt-lang Python dependencies (pydantic, torch, numpy, pytest).

### `tt-lang-ci-ubuntu-22-04`
CI image with tt-mlir toolchain only (no tt-lang). Used by CI workflows to build and test tt-lang from source.

**Contents:** tt-mlir toolchain, Python venv, build tools

### `tt-lang-dist-ubuntu-22-04`
Distribution image for end users with pre-built tt-lang, ready to `import ttl`.

**Contents:** tt-mlir + installed tt-lang + examples + tests

### `tt-lang-ird-ubuntu-22-04`
Interactive Research & Development image with dev tools for building tt-lang from source.

**Contents:** tt-mlir toolchain + dev tools (ssh, tmux, vim, black, sphinx)

## Build Scripts

### `.github/containers/build-docker-images.sh`
Orchestrates building all images with proper tagging and optional registry push.

**Important:** This script now requires a pre-built toolchain from CI cache.

```bash
# Build locally (requires pre-built toolchain)
.github/containers/build-docker-images.sh --ttmlir-toolchain=/path/to/toolchain --no-push

# Build and push to registry
.github/containers/build-docker-images.sh --ttmlir-toolchain=/path/to/toolchain

# Check if images exist without building
.github/containers/build-docker-images.sh --ttmlir-toolchain=/path/to/toolchain --check-only
```

### `.github/containers/test-docker-smoke.sh`
Quick smoke test to verify container functionality.

```bash
.github/containers/test-docker-smoke.sh
```

### `.github/containers/get-docker-tag.sh`
Generates deterministic Docker tags from file hashes and tt-mlir version.

```bash
.github/containers/get-docker-tag.sh <MLIR_DOCKER_TAG>
```

## Hardware Access

To access Tenstorrent hardware from containers, use:

```bash
docker run -it \
    --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    <image> <command>
```

## Architecture

Container builds use a pre-built toolchain from a dedicated cache workflow:

```
call-build-ttmlir-toolchain.yml (dedicated cache builder)
         |
         | builds LLVM + tt-mlir (~3-4 hours)
         | saves to cache
         v
[GitHub Actions Cache: Linux-ttlang-toolchain-v1-{commit}]
         |
         v
Container Build (call-build-docker.yml)
         |
         | restores cache (FAILS if not present)
         v
    Docker Build
         |
         v
tt-lang-base-ubuntu-22-04
    (adds Python deps)
         |
         v
      build stage
    (builds tt-lang only)
    (uses pre-built toolchain)
         |
         v
        ci
   (toolchain only)
      /    \
     /      \
   dist    ird
(+ttlang) (+devtools)
```

## Build Strategy

The container build uses a **dedicated cache workflow** (GitHub Actions best practice):

1. **Toolchain workflow** (`call-build-ttmlir-toolchain.yml`) builds LLVM + tt-mlir
2. **Container workflow** restores the toolchain from cache (fails if not present)
3. **Docker build** uses the pre-built toolchain to build only tt-lang

This approach:
- Follows GitHub Actions best practice for caching
- Eliminates duplicate LLVM/tt-mlir builds
- Reduces container build time from 3-4 hours to 10-15 minutes
- Ensures CI and containers use the same toolchain
- Weekly schedule keeps cache warm (prevents 7-day eviction)

### Multi-stage Dockerfile

1. **Base stage**: tt-mlir base + Python deps (pydantic, torch, pytest)
2. **Build stage**: Copies pre-built toolchain, builds tt-lang only
3. **CI stage**: Copies tt-mlir toolchain only (no tt-lang)
4. **Dist stage**: Extends CI, installs tt-lang
5. **IRD stage**: Extends CI, adds dev tools

### Build Context Requirements

The Dockerfile expects a build context named `ttmlir-toolchain` containing:
- `lib/cmake/llvm/LLVMConfig.cmake` - LLVM toolchain
- `lib/cmake/mlir/MLIRConfig.cmake` - MLIR toolchain
- `lib/cmake/ttmlir/TTMLIRConfig.cmake` - tt-mlir
- `venv/bin/python3` - Python virtual environment
- `python_packages/ttrt/runtime/ttnn/` - TTNN runtime

This is passed via `--build-context ttmlir-toolchain=<dir>` to docker build.

## Image Sizes (Approximate)

- `tt-lang-base`: ~1.7GB
- `tt-lang-ci`: ~4-5GB (tt-mlir toolchain)
- `tt-lang-dist`: ~6-7GB (tt-mlir + tt-lang)
- `tt-lang-ird`: ~5-6GB (tt-mlir + dev tools)

## Files

- `Dockerfile.base` - Base image with Python dependencies
- `Dockerfile` - Multi-stage build (ci/dist/ird targets)
- `entrypoint.sh` - Container entrypoint that activates environments
- `activate-install.sh` - Environment activation for installed tt-lang
- `CONTAINER_README.md` - Welcome message shown to users inside container
- `cleanup-toolchain.sh` - Removes unnecessary LLVM tools to reduce size
- `build-and-install.sh` - Builds tt-lang using pre-built toolchain (from-installed mode)
- `.dockerignore` - Excludes build directories from Docker context

## Related Documentation

- [CI Workflows](../CI_WORKFLOWS.md) - Detailed documentation of CI/CD workflows
