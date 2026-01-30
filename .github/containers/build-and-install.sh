#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Configure, build, install, and cleanup tt-lang in a single script.
# This is called from Dockerfile to keep everything in one layer,
# avoiding Docker layer bloat from the large build directory.
#
# Usage:
#   ./build-and-install.sh [--ttmlir-src-dir=<path>]
#
# Options:
#   --ttmlir-src-dir=<path>  Use existing tt-mlir source directory instead of fetching

set -e

TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"
TTMLIR_SRC_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ttmlir-src-dir=*)
            TTMLIR_SRC_DIR="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=== Configuring tt-lang ==="
echo "TTMLIR_TOOLCHAIN_DIR: $TTMLIR_TOOLCHAIN_DIR"
echo "TTMLIR_SRC_DIR: ${TTMLIR_SRC_DIR:-<will fetch>}"

# Build cmake args
CMAKE_ARGS=(
    -G Ninja -B build
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER=clang++-17
    -DCMAKE_C_COMPILER=clang-17
    -DTTMLIR_CMAKE_BUILD_TYPE=Release
    -DTTMLIR_INSTALL_PREFIX=$TTMLIR_TOOLCHAIN_DIR
    -DTTLANG_ENABLE_PERF_TRACE=ON
    -DTTLANG_ENABLE_BINDINGS_PYTHON=ON
)

if [ -n "$TTMLIR_SRC_DIR" ]; then
    # Use provided source directory
    CMAKE_ARGS+=(-DTTMLIR_SRC_DIR="$TTMLIR_SRC_DIR")
else
    # Fetch via git tag
    TTMLIR_COMMIT=$(cat third-party/tt-mlir.commit | tr -d '[:space:]')
    CMAKE_ARGS+=(-DTTMLIR_GIT_TAG="$TTMLIR_COMMIT")
fi

cmake "${CMAKE_ARGS[@]}"

echo "=== Disk space after configure ==="
df -BM

echo "=== Building tt-lang ==="
source build/env/activate
cmake --build build

echo "=== Disk space after build ==="
df -BM

echo "=== Installing tt-lang ==="
cmake --install build --prefix "$TTMLIR_TOOLCHAIN_DIR"

echo "=== Copying Python packages ==="
mkdir -p "$TTMLIR_TOOLCHAIN_DIR/python_packages/ttrt/runtime"
cp -prL build/_deps/tt-mlir-build/python_packages/ttrt/runtime/* \
    "$TTMLIR_TOOLCHAIN_DIR/python_packages/ttrt/runtime/" 2>/dev/null || true

if [ -d "build/_deps/tt-mlir-build/python_packages/ttmlir" ]; then
    cp -prL build/_deps/tt-mlir-build/python_packages/ttmlir/* \
        "$TTMLIR_TOOLCHAIN_DIR/python_packages/ttmlir/" 2>/dev/null || true
fi

echo "=== Normalizing toolchain ==="
# Try temp location first (Docker), then repo location (CI)
if [ -f /tmp/normalize-ttmlir-install.sh ]; then
    bash /tmp/normalize-ttmlir-install.sh "$TTMLIR_TOOLCHAIN_DIR"
    rm -f /tmp/normalize-ttmlir-install.sh
elif [ -f .github/scripts/normalize-ttmlir-install.sh ]; then
    bash .github/scripts/normalize-ttmlir-install.sh "$TTMLIR_TOOLCHAIN_DIR"
fi

# Clean up Python cache files
find "$TTMLIR_TOOLCHAIN_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TTMLIR_TOOLCHAIN_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true

echo "=== Removing build directory ==="
rm -rf build

echo "=== Disk space after cleanup ==="
df -BM

echo "=== Build complete ==="
