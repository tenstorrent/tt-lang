#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Configure, build, install, and cleanup tt-lang.
#
# Usage:
#   build-and-install.sh [OPTIONS]
#
# Modes (mutually exclusive):
#   (default)              Full pipeline: configure + build + install + finalize
#   --toolchain-only       Configure only (LLVM + tt-metal) + finalize; no tt-lang build
#   --configure-only       Configure only; keep build dirs for downstream stages
#   --build-and-finalize   Build tt-lang + install + finalize (assumes configure already ran)
#   --finalize-only        Copy runtime libs + normalize + cleanup (assumes configure already ran)
#
# Options:
#   --llvm-cache DIR       Pre-built LLVM install to copy into LLVM_INSTALL_DIR.
#                          cmake skips the LLVM build if MLIRConfig.cmake exists.
#   --ttmetal-cache DIR    Pre-built tt-metal build to copy into tt-metal/build/.
#                          cmake skips the tt-metal build if _ttnn.so exists.
#
# Multi-stage Docker usage:
#   The configure/build-and-finalize/finalize-only modes support a multi-stage
#   Dockerfile where LLVM + tt-metal are built once in a "configure" stage,
#   then "build" and "build-toolchain" stages extend it.

set -e

MODE="full"
LLVM_CACHE=""
TTMETAL_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --toolchain-only)
            MODE="toolchain-only"
            shift
            ;;
        --configure-only)
            MODE="configure-only"
            shift
            ;;
        --build-and-finalize)
            MODE="build-and-finalize"
            shift
            ;;
        --finalize-only)
            MODE="finalize-only"
            shift
            ;;
        --llvm-cache)
            LLVM_CACHE="$2"
            shift 2
            ;;
        --ttmetal-cache)
            TTMETAL_CACHE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"

# ---- Phase: Configure (restore caches + cmake configure + pip install) ----
do_configure() {
    # Restore LLVM cache if provided and non-empty
    if [ -n "$LLVM_CACHE" ] && [ -d "$LLVM_CACHE/lib/cmake/mlir" ]; then
        echo "=== Restoring LLVM cache from $LLVM_CACHE ==="
        mkdir -p "$TTMLIR_TOOLCHAIN_DIR"
        cp -a "$LLVM_CACHE"/. "$TTMLIR_TOOLCHAIN_DIR"/
        echo "LLVM cache restored to $TTMLIR_TOOLCHAIN_DIR"
    fi

    # Restore tt-metal cache if provided and non-empty
    if [ -n "$TTMETAL_CACHE" ] && [ -d "$TTMETAL_CACHE/ttnn" ]; then
        echo "=== Restoring tt-metal cache from $TTMETAL_CACHE ==="
        mkdir -p third-party/tt-metal/build
        cp -a "$TTMETAL_CACHE"/. third-party/tt-metal/build/
        echo "tt-metal cache restored to third-party/tt-metal/build/"
    fi

    echo "=== Configuring tt-lang ==="
    cmake -G Ninja -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_INSTALL_DIR=$TTMLIR_TOOLCHAIN_DIR \
        -DTTLANG_ENABLE_PERF_TRACE=ON \
        -DTTLANG_ENABLE_BINDINGS_PYTHON=ON

    echo "=== Disk space after configure ==="
    df -BM

    source build/env/activate

    echo "=== Installing Python runtime dependencies into toolchain venv ==="
    pip install -r requirements.txt --no-cache-dir
}

# ---- Phase: Build + Install tt-lang ----
do_build_and_install() {
    source build/env/activate

    echo "=== Building tt-lang ==="
    cmake --build build

    echo "=== Disk space after build ==="
    df -BM

    echo "=== Installing tt-lang ==="
    cmake --install build --prefix "$TTMLIR_TOOLCHAIN_DIR"
}

# ---- Phase: Finalize (copy runtime libs, normalize, cleanup) ----
do_finalize() {
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
}

# ---- Dispatch based on mode ----
case "$MODE" in
    full)
        do_configure
        do_build_and_install
        do_finalize
        echo "=== Build complete ==="
        ;;
    toolchain-only)
        do_configure
        do_finalize
        echo "=== Toolchain build complete ==="
        ;;
    configure-only)
        do_configure
        echo "=== Configure complete (build dirs preserved) ==="
        ;;
    build-and-finalize)
        do_build_and_install
        do_finalize
        echo "=== Build and finalize complete ==="
        ;;
    finalize-only)
        do_finalize
        echo "=== Finalize complete ==="
        ;;
esac
