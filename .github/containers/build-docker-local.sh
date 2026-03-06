#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build Docker images locally for testing
# Run from repository root: .github/containers/build-docker-local.sh

set -e

echo "=== tt-lang Docker Build Test ==="
echo ""

# Build base image and tag with registry path so dist/ird Dockerfile FROM resolves locally
echo "--- Building tt-lang-base ---"
sudo docker build \
    -t tt-lang-base-ubuntu-22-04:latest \
    -t ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:latest \
    -f .github/containers/Dockerfile.base .

echo "Base image built"
echo ""

# Build Dist image (pre-built tt-lang for users)
echo "--- Building tt-lang Dist image ---"
sudo docker build \
    --target dist \
    -t tt-lang-dist-ubuntu-22-04:latest \
    -f .github/containers/Dockerfile .
echo "Dist image built"
echo ""

# Build IRD image (development tools)
echo "--- Building tt-lang IRD image ---"
sudo docker build \
    --target ird \
    -t tt-lang-ird-ubuntu-22-04:latest \
    -f .github/containers/Dockerfile .
echo "IRD image built"
echo ""

echo "=== Build Complete ==="
echo ""
echo "Images created:"
echo "  - tt-lang-base-ubuntu-22-04:latest"
echo "  - tt-lang-dist-ubuntu-22-04:latest"
echo "  - tt-lang-ird-ubuntu-22-04:latest"
echo ""
echo "Test the dist image:"
echo "  sudo docker run -it tt-lang-dist-ubuntu-22-04:latest python -c \"import ttl\""
