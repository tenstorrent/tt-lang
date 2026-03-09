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
#   --configure-only       Configure only; keep build dirs for downstream steps
#   --copy-runtime-libs    Copy tt-metal runtime libs/packages into toolchain dir
#   --build-and-install    Build tt-lang + install (assumes configure already ran)
#   --finalize             Normalize toolchain + cleanup build dirs
#
# Typical multi-stage usage (build outside Docker, copy results in):
#   1. build-and-install.sh --configure-only        # Build LLVM + tt-metal
#   2. build-and-install.sh --copy-runtime-libs      # Copy libs into toolchain
#   3. cp -a toolchain/ ird-toolchain/               # Save ird toolchain
#   4. build-and-install.sh --build-and-install       # Build + install tt-lang
#   5. build-and-install.sh --finalize                # Normalize + cleanup

set -e

# When running inside a Docker container with volume-mounted repos, git
# will refuse to operate due to ownership mismatch ("dubious ownership").
# Mark all directories as safe so that cmake's git operations (patch
# application, SHA verification) work correctly.
git config --global --add safe.directory '*'

MODE="full"

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
        --copy-runtime-libs)
            MODE="copy-runtime-libs"
            shift
            ;;
        --build-and-install)
            MODE="build-and-install"
            shift
            ;;
        --finalize)
            MODE="finalize"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

TTLANG_TOOLCHAIN_DIR="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"
TTMETAL_BUILD_DIR="$TTLANG_TOOLCHAIN_DIR/tt-metal"

# ---- Phase: Configure (cmake configure + pip install) ----
do_configure() {
    echo "=== Configuring tt-lang ==="
    cmake -G Ninja -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_INSTALL_DIR=$TTLANG_TOOLCHAIN_DIR \
        -DTTMETAL_BUILD_DIR=$TTMETAL_BUILD_DIR \
        -DTTLANG_PYTHON_VENV=$TTLANG_TOOLCHAIN_DIR/venv \
        -DTTLANG_ENABLE_PERF_TRACE=ON \
        -DTTLANG_ENABLE_BINDINGS_PYTHON=ON

    echo "=== Disk space after configure ==="
    df -BM

    source build/env/activate

    echo "=== Installing Python runtime dependencies into toolchain venv ==="
    pip install -r requirements.txt --no-cache-dir
}

# ---- Phase: Copy tt-metal runtime libs into toolchain ----
do_copy_runtime_libs() {
    echo "=== Copying tt-metal runtime libraries ==="
    # Copy tt-metal runtime shared libraries
    if [ -d "$TTMETAL_BUILD_DIR/lib" ]; then
        mkdir -p "$TTLANG_TOOLCHAIN_DIR/lib"
        cp -prL "$TTMETAL_BUILD_DIR"/lib/*.so* "$TTLANG_TOOLCHAIN_DIR/lib/" 2>/dev/null || true
        echo "Copied tt-metal runtime libraries"
    fi

    # Copy ttnn shared libraries
    for so_dir in "$TTMETAL_BUILD_DIR/ttnn" "$TTMETAL_BUILD_DIR/tt_metal"; do
        if [ -d "$so_dir" ]; then
            mkdir -p "$TTLANG_TOOLCHAIN_DIR/lib"
            find "$so_dir" -name "*.so" -exec cp -pL {} "$TTLANG_TOOLCHAIN_DIR/lib/" \; 2>/dev/null || true
        fi
    done

    # Copy ttnn Python package
    if [ -d "third-party/tt-metal/ttnn/ttnn" ]; then
        mkdir -p "$TTLANG_TOOLCHAIN_DIR/python_packages/ttnn"
        cp -prL third-party/tt-metal/ttnn/ttnn/* "$TTLANG_TOOLCHAIN_DIR/python_packages/ttnn/" 2>/dev/null || true
        echo "Copied ttnn Python package"
    fi

    # Copy Tracy profiler tools
    TRACY_BIN="$TTMETAL_BUILD_DIR/tools/profiler/bin"
    if [ -d "$TRACY_BIN" ]; then
        mkdir -p "$TTLANG_TOOLCHAIN_DIR/bin"
        cp -p "$TRACY_BIN/capture-release" "$TTLANG_TOOLCHAIN_DIR/bin/" 2>/dev/null || true
        cp -p "$TRACY_BIN/csvexport-release" "$TTLANG_TOOLCHAIN_DIR/bin/" 2>/dev/null || true
        echo "Copied Tracy profiler tools"
    fi

    # Copy Tracy Python module
    if [ -d "third-party/tt-metal/tools/tracy" ]; then
        mkdir -p "$TTLANG_TOOLCHAIN_DIR/python_packages/tracy"
        cp -pr third-party/tt-metal/tools/tracy/*.py "$TTLANG_TOOLCHAIN_DIR/python_packages/tracy/" 2>/dev/null || true
        echo "Copied Tracy Python module"
    fi
}

# ---- Phase: Build + Install tt-lang ----
do_build_and_install() {
    source build/env/activate

    echo "=== Building tt-lang ==="
    cmake --build build

    echo "=== Disk space after build ==="
    df -BM

    echo "=== Installing tt-lang ==="
    cmake --install build --prefix "$TTLANG_TOOLCHAIN_DIR"
}

# ---- Phase: Finalize (normalize toolchain + cleanup) ----
do_finalize() {
    echo "=== Normalizing and cleaning up toolchain ==="
    if [ -f /tmp/normalize-ttlang-install.sh ]; then
        bash /tmp/normalize-ttlang-install.sh "$TTLANG_TOOLCHAIN_DIR"
    elif [ -f .github/scripts/normalize-ttlang-install.sh ]; then
        bash .github/scripts/normalize-ttlang-install.sh "$TTLANG_TOOLCHAIN_DIR"
    fi

    if [ -f /tmp/cleanup-toolchain.sh ]; then
        bash /tmp/cleanup-toolchain.sh "$TTLANG_TOOLCHAIN_DIR"
    elif [ -f .github/containers/cleanup-toolchain.sh ]; then
        bash .github/containers/cleanup-toolchain.sh "$TTLANG_TOOLCHAIN_DIR"
    fi

    # Clean up temp scripts
    rm -f /tmp/normalize-ttlang-install.sh /tmp/cleanup-toolchain.sh

    echo "=== Removing build directories ==="
    rm -rf build

    echo "=== Disk space after cleanup ==="
    df -BM
}

# ---- Dispatch based on mode ----
case "$MODE" in
    full)
        do_configure
        do_build_and_install
        do_copy_runtime_libs
        do_finalize
        echo "=== Build complete ==="
        ;;
    toolchain-only)
        do_configure
        do_copy_runtime_libs
        do_finalize
        echo "=== Toolchain build complete ==="
        ;;
    configure-only)
        do_configure
        echo "=== Configure complete (build dirs preserved) ==="
        ;;
    copy-runtime-libs)
        do_copy_runtime_libs
        echo "=== Runtime libs copied ==="
        ;;
    build-and-install)
        do_build_and_install
        echo "=== Build and install complete ==="
        ;;
    finalize)
        do_finalize
        echo "=== Finalize complete ==="
        ;;
esac
