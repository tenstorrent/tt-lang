#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build and optionally push tt-lang Docker images
#
# Usage:
#   ./build-docker-images.sh [MLIR_SHA] --ttmlir-toolchain=<dir> [--check-only] [--no-push] [--no-cache]
#
# Arguments:
#   MLIR_SHA              - tt-mlir commit SHA (defaults to third-party/tt-mlir.commit)
#   --ttmlir-toolchain    - REQUIRED: Path to pre-built toolchain (LLVM + tt-mlir)
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

while [[ $# -gt 0 ]]; do
    case $1 in
        --ttmlir-toolchain=*)
            TTMLIR_TOOLCHAIN_DIR="${1#*=}"
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

# Validate required toolchain argument
if [ -z "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "ERROR: --ttmlir-toolchain=<dir> is required"
    echo ""
    echo "Usage: $0 [MLIR_SHA] --ttmlir-toolchain=<dir> [--check-only] [--no-push] [--no-cache]"
    echo ""
    echo "The toolchain must contain pre-built LLVM + tt-mlir from CI cache."
    echo "Container builds require CI to have run first for this tt-mlir commit."
    exit 1
fi

if [ ! -d "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "ERROR: Toolchain directory does not exist: $TTMLIR_TOOLCHAIN_DIR"
    exit 1
fi

# Validate toolchain contents
if [ ! -f "$TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir/TTMLIRConfig.cmake" ]; then
    echo "ERROR: Invalid toolchain - TTMLIRConfig.cmake not found"
    echo "Expected at: $TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir/TTMLIRConfig.cmake"
    exit 1
fi

if [ ! -f "$TTMLIR_TOOLCHAIN_DIR/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "ERROR: Invalid toolchain - MLIRConfig.cmake not found (LLVM toolchain missing)"
    echo "Expected at: $TTMLIR_TOOLCHAIN_DIR/lib/cmake/mlir/MLIRConfig.cmake"
    exit 1
fi

# Default to pinned tt-mlir commit if not specified
if [ -z "$MLIR_SHA" ]; then
    MLIR_SHA=$(cat third-party/tt-mlir.commit | tr -d '[:space:]')
fi

REPO=tenstorrent/tt-lang
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
echo "Check only: $CHECK_ONLY"
echo "No push: $NO_PUSH"
echo "No cache: $NO_CACHE"
echo ""

# Get version from git tags (e.g., v0.1.0 or v0.1.0-5-gabc1234 for dev builds)
TTLANG_VERSION=$(git describe --tags --match "v[0-9]*" --always --dirty 2>/dev/null || echo "v0.0.0-unknown")
echo "tt-lang version: $TTLANG_VERSION"
echo "tt-mlir SHA: ${MLIR_SHA:0:12}"

# Docker tag is the git version (sanitized for Docker - replace / and : with -)
DOCKER_TAG=$(echo "$TTLANG_VERSION" | sed 's/[\/:]/-/g')
echo "Docker tag: $DOCKER_TAG"
echo ""

# Note: Using pre-built toolchain from CI cache
echo "Note: Using pre-built toolchain from: $TTMLIR_TOOLCHAIN_DIR"
echo ""

# Build function
build_image() {
    local name=$1
    local dockerfile=$2
    local target=$3

    # Always use registry path for image references (Dockerfile expects this)
    # Simplified names are just aliases for user convenience
    local local_image="$name:$DOCKER_TAG"
    local registry_image="ghcr.io/$REPO/$name:$DOCKER_TAG"

    echo "--- Processing: $name ---"

    # Check if image already exists in registry (only when not using --no-push)
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

    if [ "$NO_PUSH" = false ]; then
        echo "Building: $registry_image"
    else
        echo "Building: $local_image (also tagged as $registry_image for Dockerfile references)"
    fi

    local target_arg=""
    if [ -n "$target" ]; then
        target_arg="--target $target"
    fi

    # Build options
    local cache_arg=""
    if [ "$NO_CACHE" = true ]; then
        cache_arg="--no-cache"
    fi

    # Always tag with registry path (required for Dockerfile FROM references)
    # Also add simplified name and latest tags
    # Pass pre-built toolchain as build context
    docker build \
        --progress=plain \
        --build-context ttmlir-toolchain="$TTMLIR_TOOLCHAIN_DIR" \
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
    else
        echo "Skipping push (--no-push specified)"
    fi

    echo "Disk space after $name:"
    df -h | head -2

    echo "Done: $name"
    echo ""
}

# Build images (all from same Dockerfile with different targets)
# Order matters: dev first (user extends dev)
DOCKERFILE=".github/containers/Dockerfile"

build_image "tt-lang-dev-ubuntu-22-04" "$DOCKERFILE" dev
build_image "tt-lang-user-ubuntu-22-04" "$DOCKERFILE" user


# Final cleanup of all unused Docker resources
echo "Performing final Docker cleanup..."
docker builder prune -af 2>/dev/null || true
docker system prune -af --volumes 2>/dev/null || true
echo "Final disk space:"
df -h | head -2
echo ""

echo "=== Build Complete ==="
echo ""

if [ "$NO_PUSH" = false ]; then
    echo "Images built and pushed:"
    echo "  - ghcr.io/$REPO/tt-lang-dev-ubuntu-22-04:$DOCKER_TAG (tt-mlir toolchain + dev tools)"
    echo "  - ghcr.io/$REPO/tt-lang-user-ubuntu-22-04:$DOCKER_TAG (dev + pre-built tt-lang)"

    # Write user image name to file for workflow consumption
    DIST_IMAGE="ghcr.io/$REPO/tt-lang-user-ubuntu-22-04:$DOCKER_TAG"
    echo "$DIST_IMAGE" > .docker-image-name
    echo ""
    echo "DIST_IMAGE_NAME:"
    echo "$DIST_IMAGE"
else
    echo "Local images built:"
    echo "  - tt-lang-dev-ubuntu-22-04:$DOCKER_TAG (tt-mlir toolchain + dev tools)"
    echo "  - tt-lang-user-ubuntu-22-04:$DOCKER_TAG (dev + pre-built tt-lang)"

    # Write local user image name to file for workflow consumption
    DIST_IMAGE="tt-lang-user-ubuntu-22-04:$DOCKER_TAG"
    echo "$DIST_IMAGE" > .docker-image-name
    echo ""
    echo "DIST_IMAGE_NAME:"
    echo "$DIST_IMAGE"
fi
echo ""
