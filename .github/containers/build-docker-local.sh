#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build Docker images locally for testing
# Run from repository root: .github/containers/build-docker-local.sh

set -e

echo "=== tt-lang Docker Build Test ==="
echo ""

# Use the CI image tag
MLIR_TAG="latest"
echo "Using tt-mlir CI image tag: $MLIR_TAG"
echo ""

# Pull the base tt-mlir images
echo "--- Pulling tt-mlir images ---"
sudo docker pull ghcr.io/tenstorrent/tt-mlir/tt-mlir-base-ubuntu-22-04:${MLIR_TAG}
sudo docker pull ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-22-04:${MLIR_TAG}
echo ""

# Build base image
echo "--- Building tt-lang-base ---"
sudo docker build \
    --build-arg MLIR_TAG=${MLIR_TAG} \
    -t tt-lang-base:local \
    -f .github/containers/Dockerfile.base .

# Tag with full registry path so dist/dev builds can find it locally
sudo docker tag tt-lang-base:local ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:local

echo "✓ Base image built"
echo ""

# Build CI image (pre-built tt-lang for users and CI)
echo "--- Building tt-lang CI image ---"
sudo docker build \
    --build-arg FROM_TAG=local \
    --build-arg MLIR_TAG=${MLIR_TAG} \
    --target ci \
    -t tt-lang-ci:local \
    -t tt-lang-dist:local \
    -f .github/containers/Dockerfile.dist .
echo "✓ CI image built (also tagged as dist)"
echo ""

# Build IRD image (interactive development)
echo "--- Building tt-lang IRD image ---"
sudo docker build \
    --build-arg FROM_TAG=local \
    --build-arg MLIR_TAG=${MLIR_TAG} \
    --target ird \
    -t tt-lang-ird:local \
    -f .github/containers/Dockerfile.dist .
echo "✓ IRD image built"
echo ""

echo "=== Build Complete ==="
echo ""
echo "Images created:"
echo "  - tt-lang-base:local"
echo "  - tt-lang-ci:local (also tagged as tt-lang-dist:local)"
echo "  - tt-lang-ird:local"
echo ""
echo "Test the CI/dist image:"
echo "  sudo docker run -it tt-lang-ci:local python -c \"import ttl\""
echo "  sudo docker run -it tt-lang-dist:local python -c \"import ttl\""
echo ""
echo "Test the IRD image:"
echo "  sudo docker run -it tt-lang-ird:local gdb --version"
