#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build Docker images locally for testing
#
# Usage: .github/containers/build-docker-local.sh \
#            --ttmlir-toolchain=<dir> --ttlang-install=<dir>
#
# Run from repository root

set -e

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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "ERROR: --ttmlir-toolchain=<dir> is required"
    echo ""
    echo "Usage: $0 --ttmlir-toolchain=<dir> --ttlang-install=<dir>"
    echo ""
    echo "The toolchain must contain pre-built LLVM + tt-mlir."
    exit 1
fi

if [ -z "$TTLANG_INSTALL_DIR" ]; then
    echo "ERROR: --ttlang-install=<dir> is required"
    echo ""
    echo "Usage: $0 --ttmlir-toolchain=<dir> --ttlang-install=<dir>"
    echo ""
    echo "The ttlang-install directory must contain toolchain + pre-built tt-lang."
    exit 1
fi

if [ ! -d "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "ERROR: Toolchain directory does not exist: $TTMLIR_TOOLCHAIN_DIR"
    exit 1
fi

if [ ! -d "$TTLANG_INSTALL_DIR" ]; then
    echo "ERROR: ttlang-install directory does not exist: $TTLANG_INSTALL_DIR"
    exit 1
fi

echo "=== tt-lang Docker Build Test ==="
echo "Toolchain: $TTMLIR_TOOLCHAIN_DIR"
echo "tt-lang install: $TTLANG_INSTALL_DIR"
echo ""

DOCKERFILE=".github/containers/Dockerfile"

# Build Dev image
echo "--- Building tt-lang Dev image ---"
docker build \
    --build-context ttmlir-toolchain="$TTMLIR_TOOLCHAIN_DIR" \
    --build-context ttlang-install="$TTLANG_INSTALL_DIR" \
    --target dev \
    -t tt-lang-dev:local \
    -f "$DOCKERFILE" .
echo "✓ Dev image built"
echo ""

# Build User image
echo "--- Building tt-lang User image ---"
docker build \
    --build-context ttmlir-toolchain="$TTMLIR_TOOLCHAIN_DIR" \
    --build-context ttlang-install="$TTLANG_INSTALL_DIR" \
    --target user \
    -t tt-lang-user:local \
    -f "$DOCKERFILE" .
echo "✓ User image built"
echo ""

echo "=== Build Complete ==="
echo ""
echo "Images created:"
echo "  - tt-lang-dev:local (tt-mlir toolchain + dev tools)"
echo "  - tt-lang-user:local (dev + pre-built tt-lang)"
echo ""
echo "Test the user image:"
echo "  docker run -it tt-lang-user:local python -c \"import ttl\""
