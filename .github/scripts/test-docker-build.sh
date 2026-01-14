#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Quick test script for Docker builds
# Run from repository root: .github/scripts/test-docker-build.sh

set -e

echo "=== tt-lang Docker Build Test ==="
echo ""

# Use the CI image tag
MLIR_TAG="latest"
echo "Using tt-mlir CI image tag: $MLIR_TAG"
echo ""

# Pull the base tt-mlir images
echo "--- Pulling tt-mlir images ---"
docker pull ghcr.io/tenstorrent/tt-mlir/tt-mlir-base-ubuntu-22-04:${MLIR_TAG}
docker pull ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-22-04:${MLIR_TAG}
echo ""

# Build base image
echo "--- Building tt-lang-base ---"
docker build \
    --build-arg MLIR_TAG=${MLIR_TAG} \
    -t tt-lang-base:local \
    -f .github/containers/Dockerfile.base .
echo "✓ Base image built"
echo ""

# Build dist image (pre-built tt-lang)
echo "--- Building tt-lang dist image ---"
docker build \
    --build-arg FROM_TAG=local \
    --build-arg MLIR_TAG=${MLIR_TAG} \
    --target dist \
    -t tt-lang:local \
    -f .github/containers/Dockerfile.dist .
echo "✓ Dist image built"
echo ""

# Build dev image (development)
echo "--- Building tt-lang dev image ---"
docker build \
    --build-arg FROM_TAG=local \
    --build-arg MLIR_TAG=${MLIR_TAG} \
    --target dev \
    -t tt-lang-dev:local \
    -f .github/containers/Dockerfile.dist .
echo "✓ Dev image built"
echo ""

echo "=== Build Complete ==="
echo ""
echo "Images created:"
echo "  - tt-lang-base:local"
echo "  - tt-lang:local (dist)"
echo "  - tt-lang-dev:local (dev)"
echo ""
echo "Test the dist image:"
echo "  docker run -it tt-lang:local python -c \"import ttlang\""
echo ""
echo "Test the dev image:"
echo "  docker run -it tt-lang-dev:local gdb --version"
