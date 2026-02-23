#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Configure, build, install, and cleanup tt-lang in a single script.
# This is called from Dockerfile to keep everything in one layer,
# avoiding Docker layer bloat from the large build directory.
#
# Usage:
#   build-and-install.sh [--toolchain-only]
#
# Options:
#   --toolchain-only  Only configure (which builds and installs tt-mlir via
#                     FetchContent) without building or installing tt-lang.
#                     Used by the IRD container target.

set -e

TOOLCHAIN_ONLY=false
if [ "$1" = "--toolchain-only" ]; then
    TOOLCHAIN_ONLY=true
fi

TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"

echo "=== Configuring tt-lang ==="
if [ "$TOOLCHAIN_ONLY" = true ]; then
    echo "    (toolchain-only mode: will skip tt-lang build and install)"
fi
TTMLIR_COMMIT=$(cat third-party/tt-mlir.commit | tr -d '[:space:]')
cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_C_COMPILER=clang-17 \
    -DTTMLIR_CMAKE_BUILD_TYPE=Release \
    -DTTMLIR_INSTALL_PREFIX=$TTMLIR_TOOLCHAIN_DIR \
    -DTTMLIR_GIT_TAG=$TTMLIR_COMMIT \
    -DTTLANG_ENABLE_PERF_TRACE=ON \
    -DTTLANG_ENABLE_BINDINGS_PYTHON=ON

echo "=== Disk space after configure ==="
df -BM

source build/env/activate

echo "=== Installing Python runtime dependencies into toolchain venv ==="
# requirements.txt is also installed into system Python in tt-lang-base, but
# the toolchain venv is isolated and does not inherit system site-packages.
pip install -r requirements.txt --no-cache-dir

if [ "$TOOLCHAIN_ONLY" = false ]; then
    echo "=== Building tt-lang ==="
    cmake --build build

    echo "=== Disk space after build ==="
    df -BM

    echo "=== Installing tt-lang ==="
    cmake --install build --prefix "$TTMLIR_TOOLCHAIN_DIR"
fi

echo "=== Copying Python packages ==="
mkdir -p "$TTMLIR_TOOLCHAIN_DIR/python_packages/ttrt/runtime"
cp -prL build/_deps/tt-mlir-build/python_packages/ttrt/runtime/* \
    "$TTMLIR_TOOLCHAIN_DIR/python_packages/ttrt/runtime/" 2>/dev/null || true

if [ -d "build/_deps/tt-mlir-build/python_packages/ttmlir" ]; then
    cp -prL build/_deps/tt-mlir-build/python_packages/ttmlir/* \
        "$TTMLIR_TOOLCHAIN_DIR/python_packages/ttmlir/" 2>/dev/null || true
fi

# Copy Tracy profiler tools (tt-mlir does not install them)
PROFILER_BIN="build/_deps/tt-mlir-src/third_party/tt-metal/src/tt-metal/build/tools/profiler/bin"
if [ -d "$PROFILER_BIN" ]; then
    mkdir -p "$TTMLIR_TOOLCHAIN_DIR/bin"
    cp -p "$PROFILER_BIN/capture-release" "$TTMLIR_TOOLCHAIN_DIR/bin/" 2>/dev/null || true
    cp -p "$PROFILER_BIN/csvexport-release" "$TTMLIR_TOOLCHAIN_DIR/bin/" 2>/dev/null || true
    echo "Copied Tracy profiler tools"
fi

# Copy Tracy Python module (tt-mlir does not install it)
TT_METAL_SRC="build/_deps/tt-mlir-src/third_party/tt-metal/src/tt-metal"
if [ -d "$TT_METAL_SRC/tools/tracy" ]; then
    mkdir -p "$TTMLIR_TOOLCHAIN_DIR/python_packages/tracy"
    cp -pr "$TT_METAL_SRC/tools/tracy/"*.py "$TTMLIR_TOOLCHAIN_DIR/python_packages/tracy/" 2>/dev/null || true
    echo "Copied Tracy Python module"
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

if [ "$TOOLCHAIN_ONLY" = true ]; then
    echo "=== Toolchain build complete ==="
else
    echo "=== Build complete ==="
fi
