#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build, install, and cleanup tt-lang in a single script.
# This is called from Dockerfile to keep everything in one layer,
# avoiding Docker layer bloat from the large build directory.

set -e

TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"

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

echo "=== Normalizing and cleaning up toolchain ==="
bash /tmp/normalize-ttmlir-install.sh "$TTMLIR_TOOLCHAIN_DIR"
bash /tmp/cleanup-toolchain.sh "$TTMLIR_TOOLCHAIN_DIR"

# Clean up Python cache files
find "$TTMLIR_TOOLCHAIN_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TTMLIR_TOOLCHAIN_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true

# Clean up temp scripts
rm -f /tmp/normalize-ttmlir-install.sh /tmp/cleanup-toolchain.sh

echo "=== Removing build directory ==="
rm -rf build

echo "=== Disk space after cleanup ==="
df -BM

echo "=== Build complete ==="
