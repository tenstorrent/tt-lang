# tt-lang Docker Containers

This directory contains the Dockerfile and scripts for building tt-lang container images.

## Images

### `tt-lang-dev-ubuntu-22-04`
Development image with tt-mlir toolchain and dev tools. For developers who want to build tt-lang from source.

**Contents:** Ubuntu 22.04 + clang-17 + Python 3.11 + tt-mlir toolchain + dev tools (vim, tmux, ssh)

### `tt-lang-user-ubuntu-22-04`
User image with pre-built tt-lang, ready to `import ttl`. Extends dev image.

**Contents:** dev image + installed tt-lang + examples

## Build Scripts

### `.github/containers/build-docker-images.sh`
Orchestrates building all images with proper tagging and optional registry push.

**Important:** Requires both a pre-built toolchain AND pre-built tt-lang from CI.

```bash
# Build locally (requires pre-built toolchain and ttlang-install)
.github/containers/build-docker-images.sh \
    --ttmlir-toolchain=/path/to/toolchain \
    --ttlang-install=/path/to/ttlang-install \
    --no-push

# Build and push to registry
.github/containers/build-docker-images.sh \
    --ttmlir-toolchain=/path/to/toolchain \
    --ttlang-install=/path/to/ttlang-install

# Check if images exist without building
.github/containers/build-docker-images.sh \
    --ttmlir-toolchain=/path/to/toolchain \
    --ttlang-install=/path/to/ttlang-install \
    --check-only
```

### `.github/containers/build-docker-local.sh`
Simplified script for local testing.

```bash
.github/containers/build-docker-local.sh \
    --ttmlir-toolchain=/path/to/toolchain \
    --ttlang-install=/path/to/ttlang-install
```

## Hardware Access

To access Tenstorrent hardware from containers:

```bash
docker run -it \
    --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    <image> <command>
```

## Architecture

Container builds use a pre-built toolchain from a dedicated cache workflow, and tt-lang is built outside Docker for space efficiency:

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
         | 1. Restore toolchain cache (FAILS if not present)
         | 2. Build tt-lang using build-ttlang action
         | 3. Create ttlang-install (toolchain + tt-lang)
         v
    Docker Build (two build contexts)
         |
         +-- ttmlir-toolchain (for dev image)
         +-- ttlang-install (for user image)
         |
         v
      base stage
    (Ubuntu 22.04 +
     build deps +
     Python deps)
         |
         v
       dev
    (toolchain +
     dev tools)
         |
         v
       user
    (dev + tt-lang)
```

**Why build tt-lang outside Docker?**
- Space efficiency: Build artifacts don't end up in Docker layers
- Faster builds: Can reuse ccache from CI
- Smaller images: Only the installed files are copied into the image

## Multi-stage Dockerfile

The self-contained `Dockerfile` uses multi-stage builds for efficiency:

1. **Base stage** (`base`): Ubuntu 22.04 + clang-17 + Python 3.11 + pip dependencies
2. **Dev stage** (`dev`): Base + toolchain + dev tools (for developers)
3. **User stage** (`user`): Dev + pre-built tt-lang (for end users)

Layer optimization:
- Dependencies ordered from least to most frequently changing
- apt-get and pip installs combined into single layers
- `user` extends `dev` to reuse layers
- tt-lang built outside Docker, only installed files copied in

## Build Context Requirements

The Dockerfile expects two build contexts:

### `ttmlir-toolchain` (for dev image)
Pre-built LLVM + tt-mlir toolchain:
- `lib/cmake/llvm/LLVMConfig.cmake` - LLVM toolchain
- `lib/cmake/mlir/MLIRConfig.cmake` - MLIR toolchain
- `lib/cmake/ttmlir/TTMLIRConfig.cmake` - tt-mlir
- `venv/bin/python3` - Python virtual environment
- `python_packages/ttrt/runtime/ttnn/` - TTNN runtime

### `ttlang-install` (for user image)
Toolchain + pre-built tt-lang:
- Everything from `ttmlir-toolchain`
- `python_packages/ttl/` - tt-lang Python package
- `python_packages/pykernel/` - pykernel Python package
- `python_packages/sim/` - simulator Python package
- `env/activate` - tt-lang environment activation script
- `examples/` - Example scripts
- `test/` - Test files

Both passed via `--build-context` to docker build.

## Image Sizes (Approximate)

- `tt-lang-dev`: ~4-5GB (tt-mlir toolchain + dev tools)
- `tt-lang-user`: ~5-6GB (dev + tt-lang)

## Files

- `Dockerfile` - Self-contained multi-stage build (base/dev/user targets)
- `entrypoint.sh` - Container entrypoint that activates environments
- `CONTAINER_README.md` - Welcome message shown to users inside container
- `cleanup-toolchain.sh` - Removes unnecessary LLVM tools to reduce size
- `build-docker-images.sh` - Main build orchestration script
- `build-docker-local.sh` - Local testing script
- `push-docker-images.sh` - Push locally built images to registry

## Related Documentation

- [CI Workflows](../CI_WORKFLOWS.md) - Detailed documentation of CI/CD workflows
