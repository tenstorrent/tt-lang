#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build and optionally push tt-lang Docker images
#
# Usage:
#   ./build-docker-images.sh [MLIR_SHA] [--check-only] [--no-push]
#
# Arguments:
#   MLIR_SHA     - tt-mlir commit SHA (defaults to third-party/tt-mlir.commit)
#   --check-only - Only check if images exist, don't build
#   --no-push    - Build locally but don't push to registry
#
# Must be run from the repository root directory

set -e

# Parse arguments
MLIR_SHA=""
CHECK_ONLY=false
NO_PUSH=false

for arg in "$@"; do
    case $arg in
        --check-only)
            CHECK_ONLY=true
            ;;
        --no-push)
            NO_PUSH=true
            ;;
        *)
            if [ -z "$MLIR_SHA" ]; then
                MLIR_SHA="$arg"
            fi
            ;;
    esac
done

# Default to pinned tt-mlir commit if not specified
if [ -z "$MLIR_SHA" ]; then
    MLIR_SHA=$(cat third-party/tt-mlir.commit | tr -d '[:space:]')
fi

REPO=tenstorrent/tt-lang
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== tt-lang Docker Image Builder ==="
echo "tt-mlir SHA: $MLIR_SHA"
echo "Check only: $CHECK_ONLY"
echo "No push: $NO_PUSH"
echo ""

# Get tt-mlir Docker tag
# We need to fetch tt-mlir to use its get-docker-tag.sh script
TMP_DIR=".tmp/tt-mlir"
if [ ! -d "$TMP_DIR" ]; then
    echo "Cloning tt-mlir to get Docker tag..."
    git clone --depth 1 https://github.com/tenstorrent/tt-mlir.git "$TMP_DIR"
fi

echo "Checking out tt-mlir at $MLIR_SHA..."
(cd "$TMP_DIR" && git fetch origin "$MLIR_SHA" && git checkout "$MLIR_SHA" --quiet)

MLIR_TAG=$("$TMP_DIR/.github/get-docker-tag.sh")
echo "tt-mlir Docker tag: $MLIR_TAG"

# Get tt-lang Docker tag
DOCKER_TAG=$("$SCRIPT_DIR/get-docker-tag.sh" "$MLIR_TAG")
echo "tt-lang Docker tag: $DOCKER_TAG"
echo ""

# Note: tt-lang builds tt-mlir via FetchContent, so we don't require
# a pre-existing tt-mlir Docker image (unlike tt-xla which layers on top)
echo "Note: tt-lang uses FetchContent to build tt-mlir from source"
echo ""

# Build function
build_image() {
    local name=$1
    local dockerfile=$2
    local target=$3
    local image="ghcr.io/$REPO/$name:$DOCKER_TAG"

    echo "--- Processing: $name ---"

    # Check if image already exists
    if docker manifest inspect "$image" > /dev/null 2>&1; then
        echo "✓ Image already exists: $image"
        if [ "$CHECK_ONLY" = true ]; then
            return 0
        fi
        echo "  Skipping build (image exists)"
        return 0
    fi

    if [ "$CHECK_ONLY" = true ]; then
        echo "✗ Image does not exist: $image"
        return 2
    fi

    echo "Building: $image"

    local target_arg=""
    if [ -n "$target" ]; then
        target_arg="--target $target"
    fi

    docker build \
        --progress=plain \
        $target_arg \
        --build-arg FROM_TAG="$DOCKER_TAG" \
        --build-arg MLIR_TAG="$MLIR_TAG" \
        -t "$image" \
        -t "ghcr.io/$REPO/$name:latest" \
        -f "$dockerfile" .

    if [ "$NO_PUSH" = false ]; then
        echo "Pushing: $image"
        docker push "$image"
        docker push "ghcr.io/$REPO/$name:latest"
    else
        echo "Skipping push (--no-push specified)"
    fi

    echo "✓ Done: $name"
    echo ""
}

# Build images in dependency order
build_image "tt-lang-base-ubuntu-22-04" .github/containers/Dockerfile.base ""
build_image "tt-lang-ci-ubuntu-22-04" .github/containers/Dockerfile ci
build_image "tt-lang-dist-ubuntu-22-04" .github/containers/Dockerfile dist
build_image "tt-lang-ird-ubuntu-22-04" .github/containers/Dockerfile ird

echo "=== Build Complete ==="
echo ""
echo "Images built:"
echo "  - ghcr.io/$REPO/tt-lang-base-ubuntu-22-04:$DOCKER_TAG"
echo "  - ghcr.io/$REPO/tt-lang-ci-ubuntu-22-04:$DOCKER_TAG (tt-mlir toolchain)"
echo "  - ghcr.io/$REPO/tt-lang-dist-ubuntu-22-04:$DOCKER_TAG (pre-built tt-lang)"
echo "  - ghcr.io/$REPO/tt-lang-ird-ubuntu-22-04:$DOCKER_TAG (dev tools)"
echo ""
echo "DIST_IMAGE_NAME:"
echo "ghcr.io/$REPO/tt-lang-dist-ubuntu-22-04:$DOCKER_TAG"
