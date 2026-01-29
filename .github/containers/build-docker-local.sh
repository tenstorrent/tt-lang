#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build Docker images locally for testing
#
# Usage: .github/containers/build-docker-local.sh --ttmlir-toolchain=<dir>
#
# Run from repository root

set -e

TTMLIR_TOOLCHAIN_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ttmlir-toolchain=*)
            TTMLIR_TOOLCHAIN_DIR="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "ERROR: --ttmlir-toolchain=<dir> is required"
    echo ""
    echo "Usage: $0 --ttmlir-toolchain=<dir>"
    echo ""
    echo "The toolchain must contain pre-built LLVM + tt-mlir."
    exit 1
fi

if [ ! -d "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "ERROR: Toolchain directory does not exist: $TTMLIR_TOOLCHAIN_DIR"
    exit 1
fi

echo "=== tt-lang Docker Build Test ==="
echo "Toolchain: $TTMLIR_TOOLCHAIN_DIR"
echo ""

DOCKERFILE=".github/containers/Dockerfile"

# Build CI image
echo "--- Building tt-lang CI image ---"
docker build \
    --build-context ttmlir-toolchain="$TTMLIR_TOOLCHAIN_DIR" \
    --target ci \
    -t tt-lang-ci:local \
    -f "$DOCKERFILE" .
echo "✓ CI image built"
echo ""

# Build Dist image
echo "--- Building tt-lang Dist image ---"
docker build \
    --build-context ttmlir-toolchain="$TTMLIR_TOOLCHAIN_DIR" \
    --target user \
    -t tt-lang-user:local \
    -f "$DOCKERFILE" .
echo "✓ Dist image built"
echo ""

# Build IRD image
echo "--- Building tt-lang IRD image ---"
docker build \
    --build-context ttmlir-toolchain="$TTMLIR_TOOLCHAIN_DIR" \
    --target dev \
    -t tt-lang-dev:local \
    -f "$DOCKERFILE" .
echo "✓ IRD image built"
echo ""

echo "=== Build Complete ==="
echo ""
echo "Images created:"
echo "  - tt-lang-ci:local (tt-mlir toolchain)"
echo "  - tt-lang-user:local (pre-built tt-lang)"
echo "  - tt-lang-dev:local (dev tools)"
echo ""
echo "Test the user image:"
echo "  docker run -it tt-lang-user:local python -c \"import ttl\""
