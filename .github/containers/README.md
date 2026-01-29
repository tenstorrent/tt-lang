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

**Important:** Requires a pre-built toolchain from CI cache.

```bash
# Build locally (requires pre-built toolchain)
.github/containers/build-docker-images.sh --ttmlir-toolchain=/path/to/toolchain --no-push

# Build and push to registry
.github/containers/build-docker-images.sh --ttmlir-toolchain=/path/to/toolchain

# Check if images exist without building
.github/containers/build-docker-images.sh --ttmlir-toolchain=/path/to/toolchain --check-only
```

### `.github/containers/build-docker-local.sh`
Simplified script for local testing.

```bash
.github/containers/build-docker-local.sh --ttmlir-toolchain=/path/to/toolchain
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
      base stage
    (Ubuntu 22.04 +
     build deps +
     Python deps)
         |
         v
      build stage
    (builds tt-lang
     using cached
     toolchain)
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

## Multi-stage Dockerfile

The self-contained `Dockerfile` uses multi-stage builds for efficiency:

1. **Base stage** (`base`): Ubuntu 22.04 + clang-17 + Python 3.11 + pip dependencies
2. **Build stage** (`build`): Copies pre-built toolchain, builds tt-lang (intermediate, not exported)
3. **Dev stage** (`dev`): Base + toolchain + dev tools (for developers)
4. **User stage** (`user`): Dev + pre-built tt-lang (for end users)

Layer optimization:
- Dependencies ordered from least to most frequently changing
- apt-get and pip installs combined into single layers
- `user` extends `dev` to reuse layers
- Build context (toolchain) copied after stable dependencies

## Build Context Requirements

The Dockerfile expects a build context named `ttmlir-toolchain` containing:
- `lib/cmake/llvm/LLVMConfig.cmake` - LLVM toolchain
- `lib/cmake/mlir/MLIRConfig.cmake` - MLIR toolchain
- `lib/cmake/ttmlir/TTMLIRConfig.cmake` - tt-mlir
- `venv/bin/python3` - Python virtual environment
- `python_packages/ttrt/runtime/ttnn/` - TTNN runtime

Passed via `--build-context ttmlir-toolchain=<dir>` to docker build.

## Image Sizes (Approximate)

- `tt-lang-dev`: ~4-5GB (tt-mlir toolchain + dev tools)
- `tt-lang-user`: ~5-6GB (dev + tt-lang)

## Files

- `Dockerfile` - Self-contained multi-stage build (base/build/dev/user targets)
- `entrypoint.sh` - Container entrypoint that activates environments
- `activate-install.sh` - Environment activation for installed tt-lang
- `CONTAINER_README.md` - Welcome message shown to users inside container
- `cleanup-toolchain.sh` - Removes unnecessary LLVM tools to reduce size
- `build-and-install.sh` - Builds tt-lang using pre-built toolchain
- `build-docker-images.sh` - Main build orchestration script
- `build-docker-local.sh` - Local testing script
- `push-docker-images.sh` - Push locally built images to registry

## Related Documentation

- [CI Workflows](../CI_WORKFLOWS.md) - Detailed documentation of CI/CD workflows
