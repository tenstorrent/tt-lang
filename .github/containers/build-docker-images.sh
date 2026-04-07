#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build and optionally push tt-lang Docker images
#
# Usage:
#   ./build-docker-images.sh [--check-only] [--no-push] [--no-cache] [--image-type <base|dist|ird>]
#
# Arguments:
#   --check-only      - Only check if images exist, don't build
#   --no-push         - Build locally but don't push to registry
#   --no-cache        - Build from scratch without using Docker cache
#   --image-type TYPE - Build only the specified image type (base, dist, or ird)
#                       Default (no flag) builds all three
#
# Must be run from the repository root directory

set -e

# Parse arguments
CHECK_ONLY=false
NO_PUSH=false
NO_CACHE=false
IMAGE_TYPE=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --image-type)
            IMAGE_TYPE="$2"
            shift 2
            ;;
        *)
            echo "WARNING: Unknown argument: $1" >&2
            shift
            ;;
    esac
done

# Validate --image-type if provided
if [ -n "$IMAGE_TYPE" ] && [ "$IMAGE_TYPE" != "base" ] && [ "$IMAGE_TYPE" != "dist" ] && [ "$IMAGE_TYPE" != "ird" ]; then
    echo "ERROR: Invalid --image-type '$IMAGE_TYPE'. Must be one of: base, dist, ird"
    exit 1
fi

REPO=tenstorrent/tt-lang
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for uncommitted changes (skip in CI). Exclude third-party/ since
# submodule patching during cmake configure makes it dirty.
DIRTY_FILES=$(git diff --name-only HEAD -- . ':!third-party')
if [ -z "$CI" ] && [ -n "$DIRTY_FILES" ]; then
echo "ERROR: Uncommitted changes:"
echo "$DIRTY_FILES"
exit 1
fi

echo "=== tt-lang Docker Image Builder ==="
echo "Check only: $CHECK_ONLY"
echo "No push: $NO_PUSH"
echo "No cache: $NO_CACHE"
[ -n "$IMAGE_TYPE" ] && echo "Image type: $IMAGE_TYPE"
echo ""

# Get version from git tags (e.g., v0.1.0 or v0.1.0-5-gabc1234 for dev builds)
TTLANG_VERSION=$(git describe --tags --match "v[0-9]*" --always 2>/dev/null || true)
if [ -z "$TTLANG_VERSION" ]; then
    echo "ERROR: Could not determine version from git tags."
    echo "Ensure the checkout includes tags (fetch-tags: true) and sufficient history."
    exit 1
fi
echo "tt-lang version: $TTLANG_VERSION"

# Docker tag uses the nearest version tag (e.g., v0.1.8) so rebuilds overwrite
# the same tag rather than creating a new one per commit.
DOCKER_TAG=$(git describe --tags --match "v[0-9]*" --abbrev=0 2>/dev/null | sed 's/[\/:]/-/g')
echo "Docker tag: $DOCKER_TAG"
echo ""

echo "Note: tt-lang builds LLVM, tt-metal, and tt-mlir from submodules"
echo ""

# Compute image names up front (used by both build and check-only paths).
if [ "$NO_PUSH" = false ]; then
    BASE_IMAGE="ghcr.io/$REPO/tt-lang-base-ubuntu-22-04:$DOCKER_TAG"
    DIST_IMAGE="ghcr.io/$REPO/tt-lang-dist-ubuntu-22-04:$DOCKER_TAG"
    IRD_IMAGE="ghcr.io/$REPO/tt-lang-ird-ubuntu-22-04:$DOCKER_TAG"
else
    BASE_IMAGE="tt-lang-base-ubuntu-22-04:$DOCKER_TAG"
    DIST_IMAGE="tt-lang-dist-ubuntu-22-04:$DOCKER_TAG"
    IRD_IMAGE="tt-lang-ird-ubuntu-22-04:$DOCKER_TAG"
fi

# Write image names to files for workflow consumption (avoids fragile log parsing).
echo "$BASE_IMAGE" > .docker-image-base
echo "$DIST_IMAGE" > .docker-image-name
echo "$IRD_IMAGE"  > .docker-image-ird

# Extract tt-metal submodule SHA for Dockerfile.base build arg.
TT_METAL_SHA=$(git ls-tree HEAD third-party/tt-metal 2>/dev/null | awk '{print $3}')

# Build function
build_image() {
    local name=$1
    local dockerfile=$2
    local target=$3

    # Always use registry path for image references (Dockerfile expects this)
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
        echo "Building: $local_image (local only)"
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

    # When building locally (--no-push), only use local tags to avoid
    # shadowing registry images in the local Docker cache.
    # DOCKER_BUILD_EXTRA_ARGS allows callers to inject additional args
    # (e.g. --build-context for cache injection)
    local tag_args=()
    if [ "$NO_PUSH" = false ]; then
        tag_args+=(-t "$registry_image" -t "ghcr.io/$REPO/$name:latest")
    fi
    tag_args+=(-t "$local_image" -t "$name:latest")

    # Pass BASE_IMAGE so Dockerfile FROM references resolve correctly.
    # For local builds, prefer the local base image but fall back to the
    # registry image if no local build exists.
    local base_image_arg=""
    if [ "$NO_PUSH" = false ]; then
        base_image_arg="--build-arg BASE_IMAGE=ghcr.io/$REPO/tt-lang-base-ubuntu-22-04:latest"
    elif docker image inspect tt-lang-base-ubuntu-22-04:latest > /dev/null 2>&1; then
        base_image_arg="--build-arg BASE_IMAGE=tt-lang-base-ubuntu-22-04:latest"
    else
        base_image_arg="--build-arg BASE_IMAGE=ghcr.io/$REPO/tt-lang-base-ubuntu-22-04:latest"
    fi

    docker build \
        --progress=plain \
        $cache_arg \
        $target_arg \
        $base_image_arg \
        ${DOCKER_BUILD_EXTRA_ARGS:-} \
        "${tag_args[@]}" \
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

DOCKERFILE=".github/containers/Dockerfile"

# Build images -- filtered by --image-type if specified, otherwise build all three
if [[ -z "$IMAGE_TYPE" || "$IMAGE_TYPE" == "base" ]]; then
    build_image "tt-lang-base-ubuntu-22-04" .github/containers/Dockerfile.base ""
fi
if [[ -z "$IMAGE_TYPE" || "$IMAGE_TYPE" == "dist" ]]; then
    build_image "tt-lang-dist-ubuntu-22-04" "$DOCKERFILE" dist
fi
if [[ -z "$IMAGE_TYPE" || "$IMAGE_TYPE" == "ird" ]]; then
    build_image "tt-lang-ird-ubuntu-22-04"  "$DOCKERFILE" ird
fi



# Primary output image for this run
PUSH_LABEL=$( [ "$NO_PUSH" = false ] && echo "built and pushed" || echo "built locally" )
case "$IMAGE_TYPE" in
    base) OUTPUT_IMAGE="$BASE_IMAGE" ;;
    ird)  OUTPUT_IMAGE="$IRD_IMAGE"  ;;
    *)    OUTPUT_IMAGE="$DIST_IMAGE" ;;  # dist or all
esac

echo "=== Build Complete ==="
echo ""
if [ -n "$IMAGE_TYPE" ]; then
    echo "Image $PUSH_LABEL: $OUTPUT_IMAGE"
else
    echo "Images $PUSH_LABEL:"
    echo "  - $BASE_IMAGE"
    echo "  - $DIST_IMAGE (pre-built tt-lang)"
    echo "  - $IRD_IMAGE (dev tools)"
fi

# Write image names to files for workflow consumption (avoids fragile log parsing).
echo "$BASE_IMAGE" > .docker-image-base
echo "$DIST_IMAGE" > .docker-image-name
echo "$IRD_IMAGE"  > .docker-image-ird

echo ""
echo "$OUTPUT_IMAGE"
