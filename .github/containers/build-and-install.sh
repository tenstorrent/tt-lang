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
#   (default)              Full pipeline: configure + install tt-metal + build + install + finalize
#   --toolchain-only       Configure only (LLVM + tt-metal) + finalize; no tt-lang build
#   --configure-only       Configure only; keep build dirs for downstream steps
#   --install-ttmetal      Install tt-metal artifacts from build dir into toolchain
#   --build-and-install    Build tt-lang + install (assumes configure already ran)
#   --finalize             Normalize toolchain + cleanup
#   --test-toolchain       Build in a separate dir using the installed toolchain, run tests
#
# Options:
#   --force-rebuild        Force toolchain rebuild (LLVM + tt-metal) even if cached
#   --remove-build-dir     Remove CMAKE_BINARY_DIR after finalize (for Docker builds)
#
# Typical multi-stage usage (build outside Docker, copy results in):
#   1. build-and-install.sh --configure-only               # Build LLVM + tt-metal
#   2. build-and-install.sh --install-ttmetal              # Install tt-metal into toolchain
#   3. cp -a toolchain/ ird-toolchain/                     # Save ird toolchain
#   4. build-and-install.sh --build-and-install            # Build + install tt-lang
#   5. build-and-install.sh --finalize --remove-build-dir  # Normalize + cleanup

set -e

# When running inside a Docker container with volume-mounted repos, git
# will refuse to operate due to ownership mismatch ("dubious ownership").
# Mark all directories as safe so that cmake's git operations (patch
# application, SHA verification) work correctly.
git config --global --add safe.directory '*'

MODE="full"
REMOVE_BUILD_DIR=false
FORCE_REBUILD=false

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
        --install-ttmetal)
            MODE="install-ttmetal"
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
        --test-toolchain)
            MODE="test-toolchain"
            shift
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --remove-build-dir)
            REMOVE_BUILD_DIR=true
            shift
            ;;
        *)
            echo "WARNING: Unknown argument: $1" >&2
            shift
            ;;
    esac
done

TTLANG_TOOLCHAIN_DIR="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"
CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR:-build}"

# ---- Configure (cmake configure + pip install) ----
do_configure() {
    echo "=== Configuring tt-lang ==="
    # Use the pre-built toolchain if it already contains LLVM.
    local _use_toolchain=OFF
    if [ -f "$TTLANG_TOOLCHAIN_DIR/lib/cmake/mlir/MLIRConfig.cmake" ]; then
        _use_toolchain=ON
    fi

    local _force_rebuild=OFF
    local _build_toolchain=OFF
    if [ "$FORCE_REBUILD" = true ]; then
        _force_rebuild=ON
        _build_toolchain=ON
        _use_toolchain=OFF
    fi

    cmake -G Ninja -B "$CMAKE_BINARY_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTTLANG_USE_TOOLCHAIN=$_use_toolchain \
        -DTTLANG_TOOLCHAIN_DIR=$TTLANG_TOOLCHAIN_DIR \
        -DTTLANG_PYTHON_VENV=$TTLANG_TOOLCHAIN_DIR/venv \
        -DTTLANG_ENABLE_PERF_TRACE=ON \
        -DTTLANG_FORCE_TOOLCHAIN_REBUILD=$_force_rebuild \
        -DTTLANG_BUILD_TOOLCHAIN=$_build_toolchain

    echo "=== Disk space after configure ==="
    df -BM

    source "$CMAKE_BINARY_DIR/env/activate"

    echo "=== Installing Python runtime dependencies into toolchain venv ==="
    pip install -r requirements.txt --no-cache-dir
}

# ---- Install tt-metal artifacts into toolchain ----
do_install_ttmetal() {
    echo "=== Installing tt-metal into toolchain ==="
    bash scripts/install-ttmetal.sh \
        third-party/tt-metal \
        "$CMAKE_BINARY_DIR/tt-metal" \
        "$TTLANG_TOOLCHAIN_DIR/tt-metal"
}

# ---- Build + install tt-lang ----
do_build_and_install() {
    source "$CMAKE_BINARY_DIR/env/activate"

    echo "=== Building tt-lang ==="
    cmake --build "$CMAKE_BINARY_DIR"

    echo "=== Disk space after build ==="
    df -BM

    echo "=== Installing tt-lang ==="
    cmake --install "$CMAKE_BINARY_DIR" --prefix "$TTLANG_TOOLCHAIN_DIR"
}

# ---- Finalize (normalize toolchain + cleanup) ----
do_finalize() {
    echo "=== Normalizing and cleaning up toolchain ==="
    if [ -f /tmp/normalize-toolchain-install.sh ]; then
        bash /tmp/normalize-toolchain-install.sh "$TTLANG_TOOLCHAIN_DIR"
    elif [ -f .github/scripts/normalize-toolchain-install.sh ]; then
        bash .github/scripts/normalize-toolchain-install.sh "$TTLANG_TOOLCHAIN_DIR"
    fi

    if [ -f /tmp/cleanup-toolchain.sh ]; then
        bash /tmp/cleanup-toolchain.sh "$TTLANG_TOOLCHAIN_DIR"
    elif [ -f .github/containers/cleanup-toolchain.sh ]; then
        bash .github/containers/cleanup-toolchain.sh "$TTLANG_TOOLCHAIN_DIR"
    fi

    # Clean up temp scripts
    rm -f /tmp/normalize-toolchain-install.sh /tmp/cleanup-toolchain.sh

    if [ "$REMOVE_BUILD_DIR" = true ]; then
        echo "=== Removing build directory: $CMAKE_BINARY_DIR ==="
        rm -rf "$CMAKE_BINARY_DIR"
    fi

    echo "=== Disk space after cleanup ==="
    df -BM
}

# ---- Test toolchain (separate build using installed toolchain) ----
do_test_toolchain() {
    local test_build_dir="${CMAKE_BINARY_DIR}-toolchain-test"

    echo "=== Testing toolchain from ${TTLANG_TOOLCHAIN_DIR} ==="
    echo "=== Test build dir: ${test_build_dir} ==="

    rm -rf "$test_build_dir"

    cmake -G Ninja -B "$test_build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTTLANG_TOOLCHAIN_DIR="$TTLANG_TOOLCHAIN_DIR" \
        -DTTLANG_USE_TOOLCHAIN=ON

    cmake --build "$test_build_dir"

    if [ -n "${TT_VISIBLE_DEVICES:-}" ]; then
        if ! tt-smi -r "$TT_VISIBLE_DEVICES"; then
            echo "WARNING: tt-smi -r $TT_VISIBLE_DEVICES failed" >&2
        fi
    fi

    cmake --build "$test_build_dir" --target check-ttlang

    source "$test_build_dir/env/activate"
    python examples/elementwise-tutorial/step_4_multinode_grid_auto.py

    rm -rf "$test_build_dir"
}

# ---- Dispatch based on mode ----
case "$MODE" in
    full)
        do_configure
        do_install_ttmetal
        do_build_and_install
        do_finalize
        echo "=== Build complete ==="
        ;;
    toolchain-only)
        do_configure
        do_install_ttmetal
        do_finalize
        echo "=== Toolchain build complete ==="
        ;;
    configure-only)
        do_configure
        echo "=== Configure complete (build dirs preserved) ==="
        ;;
    install-ttmetal)
        do_install_ttmetal
        echo "=== tt-metal installed into toolchain ==="
        ;;
    build-and-install)
        do_build_and_install
        echo "=== Build and install complete ==="
        ;;
    finalize)
        do_finalize
        echo "=== Finalize complete ==="
        ;;
    test-toolchain)
        do_test_toolchain
        echo "=== Toolchain test complete ==="
        ;;
esac
