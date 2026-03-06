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
#   --toolchain-only  Only configure (which builds LLVM and tt-metal from
#                     submodules) without building or installing tt-lang.
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
cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_INSTALL_DIR=$TTMLIR_TOOLCHAIN_DIR \
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

echo "=== Copying tt-metal runtime libraries ==="
# Copy tt-metal runtime shared libraries
if [ -d "third-party/tt-metal/build/lib" ]; then
    mkdir -p "$TTMLIR_TOOLCHAIN_DIR/lib"
    cp -prL third-party/tt-metal/build/lib/*.so* "$TTMLIR_TOOLCHAIN_DIR/lib/" 2>/dev/null || true
    echo "Copied tt-metal runtime libraries"
fi

# Copy ttnn shared libraries
for so_dir in third-party/tt-metal/build/ttnn third-party/tt-metal/build/tt_metal; do
    if [ -d "$so_dir" ]; then
        mkdir -p "$TTMLIR_TOOLCHAIN_DIR/lib"
        find "$so_dir" -name "*.so" -exec cp -pL {} "$TTMLIR_TOOLCHAIN_DIR/lib/" \; 2>/dev/null || true
    fi
done

# Copy ttnn Python package
if [ -d "third-party/tt-metal/ttnn/ttnn" ]; then
    mkdir -p "$TTMLIR_TOOLCHAIN_DIR/python_packages/ttnn"
    cp -prL third-party/tt-metal/ttnn/ttnn/* "$TTMLIR_TOOLCHAIN_DIR/python_packages/ttnn/" 2>/dev/null || true
    echo "Copied ttnn Python package"
fi

# Copy Tracy profiler tools
TRACY_BIN="third-party/tt-metal/build/tools/profiler/bin"
if [ -d "$TRACY_BIN" ]; then
    mkdir -p "$TTMLIR_TOOLCHAIN_DIR/bin"
    cp -p "$TRACY_BIN/capture-release" "$TTMLIR_TOOLCHAIN_DIR/bin/" 2>/dev/null || true
    cp -p "$TRACY_BIN/csvexport-release" "$TTMLIR_TOOLCHAIN_DIR/bin/" 2>/dev/null || true
    echo "Copied Tracy profiler tools"
fi

# Copy Tracy Python module
if [ -d "third-party/tt-metal/tools/tracy" ]; then
    mkdir -p "$TTMLIR_TOOLCHAIN_DIR/python_packages/tracy"
    cp -pr third-party/tt-metal/tools/tracy/*.py "$TTMLIR_TOOLCHAIN_DIR/python_packages/tracy/" 2>/dev/null || true
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

echo "=== Removing build directories ==="
rm -rf build third-party/tt-metal/build

echo "=== Disk space after cleanup ==="
df -BM

if [ "$TOOLCHAIN_ONLY" = true ]; then
    echo "=== Toolchain build complete ==="
else
    echo "=== Build complete ==="
fi
