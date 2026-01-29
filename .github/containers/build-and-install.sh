#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build and install tt-lang using pre-built toolchain from CI cache.
# This script is called from Dockerfile to keep everything in one layer,
# avoiding Docker layer bloat from the large build directory.
#
# The toolchain at TTMLIR_TOOLCHAIN_DIR must contain:
#   - LLVM/MLIR (lib/cmake/llvm, lib/cmake/mlir)
#   - tt-mlir (lib/cmake/ttmlir)
#   - Python venv (venv/)
#   - ttrt runtime (python_packages/ttrt/runtime/ttnn)
#
# This is the "from-installed" mode - we use -DTTMLIR_DIR to point to
# the pre-installed tt-mlir rather than using FetchContent.

set -e

TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"

# Verify toolchain exists and is complete
if [ ! -f "$TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir/TTMLIRConfig.cmake" ]; then
    echo "ERROR: Pre-built toolchain not found at $TTMLIR_TOOLCHAIN_DIR"
    echo "Container build requires cached toolchain from CI"
    echo "Expected: $TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir/TTMLIRConfig.cmake"
    exit 1
fi

if [ ! -f "$TTMLIR_TOOLCHAIN_DIR/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "ERROR: LLVM toolchain incomplete - MLIRConfig.cmake not found"
    echo "Expected: $TTMLIR_TOOLCHAIN_DIR/lib/cmake/mlir/MLIRConfig.cmake"
    exit 1
fi

if [ ! -x "$TTMLIR_TOOLCHAIN_DIR/venv/bin/python3" ]; then
    echo "ERROR: Python venv not found in toolchain"
    echo "Expected: $TTMLIR_TOOLCHAIN_DIR/venv/bin/python3"
    exit 1
fi

echo "=== Using pre-built toolchain from: $TTMLIR_TOOLCHAIN_DIR ==="

echo "=== Configuring tt-lang (from-installed mode) ==="
cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_C_COMPILER=clang-17 \
    -DTTMLIR_DIR=$TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir \
    -DTTLANG_ENABLE_PERF_TRACE=ON \
    -DTTLANG_ENABLE_BINDINGS_PYTHON=ON

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
cp -prL build/python_packages/ttl "$TTMLIR_TOOLCHAIN_DIR/python_packages/" 2>/dev/null || true
cp -prL build/python_packages/pykernel "$TTMLIR_TOOLCHAIN_DIR/python_packages/" 2>/dev/null || true
cp -prL build/python_packages/sim "$TTMLIR_TOOLCHAIN_DIR/python_packages/" 2>/dev/null || true

# Copy examples and tests
cp -r examples "$TTMLIR_TOOLCHAIN_DIR/" 2>/dev/null || true
cp -r test "$TTMLIR_TOOLCHAIN_DIR/" 2>/dev/null || true

# Copy env/activate script
mkdir -p "$TTMLIR_TOOLCHAIN_DIR/env"
cp build/env/activate "$TTMLIR_TOOLCHAIN_DIR/env/"

# Clean up Python cache files
find "$TTMLIR_TOOLCHAIN_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TTMLIR_TOOLCHAIN_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true

echo "=== Removing build directory ==="
rm -rf build

echo "=== Disk space after cleanup ==="
df -BM

echo "=== Build complete ==="
