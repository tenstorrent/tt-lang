#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build and optionally push tt-lang Docker images
#
# Usage:
#   ./build-docker-images.sh [MLIR_SHA] --ttmlir-toolchain=<dir> --ttlang-install=<dir> [options]
#
# Arguments:
#   MLIR_SHA              - tt-mlir commit SHA (defaults to third-party/tt-mlir.commit)
#   --ttmlir-toolchain    - REQUIRED: Path to pre-built toolchain (LLVM + tt-mlir)
#   --ttlang-install      - REQUIRED: Path to toolchain + tt-lang installed
#   --check-only          - Only check if images exist, don't build
#   --no-push             - Build locally but don't push to registry
#   --no-cache            - Build from scratch without using Docker cache
#
# Must be run from the repository root directory

set -e

# Parse arguments
MLIR_SHA=""
CHECK_ONLY=false
NO_PUSH=false
NO_CACHE=false
TTMLIR_TOOLCHAIN_DIR=""
TTLANG_INSTALL_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ttmlir-toolchain=*)
            TTMLIR_TOOLCHAIN_DIR="${1#*=}"
            shift
            ;;
        --ttlang-install=*)
            TTLANG_INSTALL_DIR="${1#*=}"
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --no-push)
            NO_PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        *)
            if [ -z "$MLIR_SHA" ]; then
                MLIR_SHA="$1"
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "ERROR: --ttmlir-toolchain=<dir> is required"
    echo ""
    echo "Usage: $0 [MLIR_SHA] --ttmlir-toolchain=<dir> --ttlang-install=<dir> [options]"
    exit 1
fi

if [ -z "$TTLANG_INSTALL_DIR" ]; then
    echo "ERROR: --ttlang-install=<dir> is required"
    echo ""
    echo "Usage: $0 [MLIR_SHA] --ttmlir-toolchain=<dir> --ttlang-install=<dir> [options]"
    exit 1
fi

if [ ! -d "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "ERROR: Toolchain directory does not exist: $TTMLIR_TOOLCHAIN_DIR"
    exit 1
fi

if [ ! -d "$TTLANG_INSTALL_DIR" ]; then
    echo "ERROR: tt-lang install directory does not exist: $TTLANG_INSTALL_DIR"
    exit 1
fi

# Validate toolchain contents
if [ ! -f "$TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir/TTMLIRConfig.cmake" ]; then
    echo "ERROR: Invalid toolchain - TTMLIRConfig.cmake not found"
    exit 1
fi

# Validate ttlang-install contents
if [ ! -d "$TTLANG_INSTALL_DIR/python_packages/ttl" ]; then
    echo "ERROR: Invalid ttlang-install - ttl Python package not found"
    exit 1
fi

# Default to pinned tt-mlir commit if not specified
if [ -z "$MLIR_SHA" ]; then
    MLIR_SHA=$(cat third-party/tt-mlir.commit | tr -d '[:space:]')
fi

REPO=tenstorrent/tt-lang

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "ERROR: There are uncommitted changes in the repository."
    echo "Please commit or stash your changes before building Docker images."
    git status --short
    exit 1
fi

echo "=== tt-lang Docker Image Builder ==="
echo "tt-mlir SHA: $MLIR_SHA"
echo "Toolchain: $TTMLIR_TOOLCHAIN_DIR"
echo "tt-lang install: $TTLANG_INSTALL_DIR"
echo "Check only: $CHECK_ONLY"
echo "No push: $NO_PUSH"
echo ""

# Get version from git tags
TTLANG_VERSION=$(git describe --tags --match "v[0-9]*" --always --dirty 2>/dev/null || echo "v0.0.0-unknown")
DOCKER_TAG=$(echo "$TTLANG_VERSION" | sed 's/[\/:]/-/g')
echo "tt-lang version: $TTLANG_VERSION"
echo "Docker tag: $DOCKER_TAG"
echo ""

# Build function
build_image() {
    local name=$1
    local dockerfile=$2
    local target=$3

    local local_image="$name:$DOCKER_TAG"
    local registry_image="ghcr.io/$REPO/$name:$DOCKER_TAG"

    echo "--- Processing: $name ---"

    # Check if image already exists in registry
    if [ "$NO_PUSH" = false ]; then
        if docker manifest inspect "$registry_image" > /dev/null 2>&1; then
            echo "Image already exists: $registry_image"
            if [ "$CHECK_ONLY" = true ]; then
                return 0
            fi
            echo "  Skipping build (image exists)"
            return 0
        fi

        if [ "$CHECK_ONLY" = true ]; then
            echo "Image does not exist: $registry_image"
            return 2
        fi
    fi

    echo "Building: $registry_image"

    local target_arg=""
    if [ -n "$target" ]; then
        target_arg="--target $target"
    fi

    local cache_arg=""
    if [ "$NO_CACHE" = true ]; then
        cache_arg="--no-cache"
    fi

    # Pass both build contexts
    docker build \
        --progress=plain \
        --build-context ttmlir-toolchain="$TTMLIR_TOOLCHAIN_DIR" \
        --build-context ttlang-install="$TTLANG_INSTALL_DIR" \
        $cache_arg \
        $target_arg \
        -t "$registry_image" \
        -t "$local_image" \
        -t "$name:latest" \
        -t "ghcr.io/$REPO/$name:latest" \
        -f "$dockerfile" .

    if [ "$NO_PUSH" = false ]; then
        echo "Pushing: $registry_image"
        docker push "$registry_image"
        docker push "ghcr.io/$REPO/$name:latest"
    fi

    echo "Disk space after $name:"
    df -h | head -2
    echo ""
}

# Build images
DOCKERFILE=".github/containers/Dockerfile"

build_image "tt-lang-dev-ubuntu-22-04" "$DOCKERFILE" dev
build_image "tt-lang-user-ubuntu-22-04" "$DOCKERFILE" user

# Final cleanup
echo "Performing final Docker cleanup..."
docker builder prune -af 2>/dev/null || true
docker system prune -af --volumes 2>/dev/null || true
echo ""

echo "=== Build Complete ==="
echo ""

if [ "$NO_PUSH" = false ]; then
    echo "Images built and pushed:"
else
    echo "Local images built:"
fi
echo "  - tt-lang-dev-ubuntu-22-04:$DOCKER_TAG (tt-mlir toolchain + dev tools)"
echo "  - tt-lang-user-ubuntu-22-04:$DOCKER_TAG (dev + pre-built tt-lang)"

USER_IMAGE="ghcr.io/$REPO/tt-lang-user-ubuntu-22-04:$DOCKER_TAG"
[ "$NO_PUSH" = true ] && USER_IMAGE="tt-lang-user-ubuntu-22-04:$DOCKER_TAG"
echo "$USER_IMAGE" > .docker-image-name
echo ""
echo "USER_IMAGE_NAME:"
echo "$USER_IMAGE"
