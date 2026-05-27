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
#   --llvm-toolchain-only  Configure only (LLVM) + finalize using an external tt-metal; no tt-lang build
#   --configure-only       Configure only; keep build dirs for downstream steps
#   --install-ttmetal      Install tt-metal artifacts from build dir into toolchain
#   --build-and-install    Build tt-lang + install (assumes configure already ran)
#   --finalize             Normalize toolchain + cleanup
#   --test-toolchain       Build in a separate dir using the installed toolchain, run tests
#
# Options:
#   --force-rebuild               Force toolchain rebuild (LLVM + tt-metal) even if cached
#   --rebuild-ttmetal             Rebuild tt-metal from submodule while keeping LLVM
#                                 from the toolchain (sets -DTTLANG_USE_TOOLCHAIN_TTMETAL=OFF)
#   --remove-build-dir            Remove CMAKE_BINARY_DIR after finalize (for Docker builds)
#   --accept-ttmetal-mismatch     Pass -DTTLANG_ACCEPT_TTMETAL_MISMATCH=ON to cmake
#                                 configure to bypass the tt-metal SHA verification
#   --external-tt-metal-dir <path>
#                                 Use an existing tt-metal source or install tree
#   --external-tt-metal-build-dir <path>
#                                 Use an existing native tt-metal build directory
#   --python <path>               Python interpreter to use for the toolchain venv
#                                 (forwarded as -DPython3_EXECUTABLE=<path>)
#   --python-venv <path>          Existing Python venv to use for configure/build
#                                 (forwarded as -DTTLANG_PYTHON_VENV=<path>)
#   --ttnn-dep-mode <mode>        Wheel ttnn dependency mode for setup.py
#                                 (pypi, external, or bundled)
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
# application, SHA verification) work correctly. Gated on a container
# indicator so direct invocations from a developer host do not mutate
# the user's global git config.
if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
    git config --global --add safe.directory '*'
fi

MODE="full"
REMOVE_BUILD_DIR=false
FORCE_REBUILD=false
REBUILD_TTMETAL=false
ACCEPT_TTMETAL_MISMATCH=false
PYTHON_EXECUTABLE=""
PYTHON_VENV=""
EXTERNAL_TT_METAL_DIR=""
EXTERNAL_TT_METAL_BUILD_DIR=""
TTNN_DEP_MODE="${TTLANG_TTNN_DEP_MODE:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --toolchain-only)
            MODE="toolchain-only"
            shift
            ;;
        --llvm-toolchain-only)
            MODE="llvm-toolchain-only"
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
        --rebuild-ttmetal)
            REBUILD_TTMETAL=true
            shift
            ;;
        --remove-build-dir)
            REMOVE_BUILD_DIR=true
            shift
            ;;
        --accept-ttmetal-mismatch)
            ACCEPT_TTMETAL_MISMATCH=true
            shift
            ;;
        --external-tt-metal-dir)
            if [ $# -lt 2 ]; then
                echo "ERROR: --external-tt-metal-dir requires a path" >&2
                exit 1
            fi
            EXTERNAL_TT_METAL_DIR="$2"
            shift 2
            ;;
        --external-tt-metal-build-dir)
            if [ $# -lt 2 ]; then
                echo "ERROR: --external-tt-metal-build-dir requires a path" >&2
                exit 1
            fi
            EXTERNAL_TT_METAL_BUILD_DIR="$2"
            shift 2
            ;;
        --python)
            if [ $# -lt 2 ]; then
                echo "ERROR: --python requires a path" >&2
                exit 1
            fi
            PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        --python-venv)
            if [ $# -lt 2 ]; then
                echo "ERROR: --python-venv requires a path" >&2
                exit 1
            fi
            PYTHON_VENV="$2"
            shift 2
            ;;
        --ttnn-dep-mode)
            if [ $# -lt 2 ]; then
                echo "ERROR: --ttnn-dep-mode requires a mode" >&2
                exit 1
            fi
            TTNN_DEP_MODE="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -n "$EXTERNAL_TT_METAL_BUILD_DIR" ] && [ -z "$EXTERNAL_TT_METAL_DIR" ]; then
    echo "ERROR: --external-tt-metal-build-dir requires --external-tt-metal-dir" >&2
    exit 1
fi

if [ "$MODE" = "llvm-toolchain-only" ] && [ -z "$EXTERNAL_TT_METAL_DIR" ]; then
    echo "ERROR: --llvm-toolchain-only requires --external-tt-metal-dir" >&2
    exit 1
fi

if [ -n "$TTNN_DEP_MODE" ]; then
    case "$TTNN_DEP_MODE" in
        pypi | external | bundled)
            export TTLANG_TTNN_DEP_MODE="$TTNN_DEP_MODE"
            ;;
        *)
            echo "ERROR: --ttnn-dep-mode must be one of: pypi, external, bundled" >&2
            exit 1
            ;;
    esac
fi

TTLANG_TOOLCHAIN_DIR="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"
if [ "$MODE" = "toolchain-only" ] || [ "$MODE" = "llvm-toolchain-only" ]; then
    CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR:-build-toolchain}"
else
    CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR:-build}"
fi

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

    if [ "$MODE" = "llvm-toolchain-only" ]; then
        _build_toolchain=OFF
    fi

    # Per-component override: rebuild tt-metal from submodule while still
    # consuming LLVM from the toolchain. Defaults to TTLANG_USE_TOOLCHAIN
    # unless --rebuild-ttmetal sets it to OFF.
    local _use_toolchain_ttmetal=$_use_toolchain
    if [ "$REBUILD_TTMETAL" = true ]; then
        _use_toolchain_ttmetal=OFF
    fi

    local _accept_ttmetal_mismatch=OFF
    if [ "$ACCEPT_TTMETAL_MISMATCH" = true ]; then
        _accept_ttmetal_mismatch=ON
    fi

    local _python_args=()
    if [ -n "$PYTHON_EXECUTABLE" ]; then
        _python_args+=("-DPython3_EXECUTABLE=$PYTHON_EXECUTABLE")
    fi
    if [ -n "$PYTHON_VENV" ]; then
        _python_args+=("-DTTLANG_PYTHON_VENV=$PYTHON_VENV")
    fi

    local _external_ttmetal_args=()
    if [ -n "$EXTERNAL_TT_METAL_DIR" ]; then
        _external_ttmetal_args+=("-DTTLANG_EXTERNAL_TT_METAL_DIR=$EXTERNAL_TT_METAL_DIR")
    fi
    if [ -n "$EXTERNAL_TT_METAL_BUILD_DIR" ]; then
        _external_ttmetal_args+=("-DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR=$EXTERNAL_TT_METAL_BUILD_DIR")
    fi

    cmake -G Ninja -B "$CMAKE_BINARY_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTTLANG_USE_TOOLCHAIN=$_use_toolchain \
        -DTTLANG_USE_TOOLCHAIN_TTMETAL=$_use_toolchain_ttmetal \
        -DTTLANG_TOOLCHAIN_DIR=$TTLANG_TOOLCHAIN_DIR \
        -DTTLANG_PYTHON_VENV=$TTLANG_TOOLCHAIN_DIR/venv \
        -DTTLANG_ENABLE_PERF_TRACE=ON \
        -DTTLANG_FORCE_TOOLCHAIN_REBUILD=$_force_rebuild \
        -DTTLANG_BUILD_TOOLCHAIN=$_build_toolchain \
        -DTTLANG_ACCEPT_TTMETAL_MISMATCH=$_accept_ttmetal_mismatch \
        "${_external_ttmetal_args[@]}" \
        "${_python_args[@]}"

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
    python examples/elementwise-tutorial/step_4_multinode_grid_full.py

    rm -rf "$test_build_dir"
}

# ---- Dispatch based on mode ----
case "$MODE" in
    full)
        do_configure
        # Only install tt-metal into the toolchain when we actually built it
        # from submodule. With TTLANG_USE_TOOLCHAIN_TTMETAL=ON (the default
        # when a toolchain copy exists), BuildTTMetal.cmake skips the build
        # and there is nothing in $CMAKE_BINARY_DIR/tt-metal to install.
        if [ "$REBUILD_TTMETAL" = true ] || [ "$FORCE_REBUILD" = true ]; then
            do_install_ttmetal
        fi
        do_build_and_install
        do_finalize
        echo "=== Build complete ==="
        ;;
    toolchain-only)
        do_configure
        if [ "$REBUILD_TTMETAL" = true ] || [ "$FORCE_REBUILD" = true ]; then
            do_install_ttmetal
        else
            echo "=== Skipping tt-metal install: not rebuilt (pass --rebuild-ttmetal or --force-rebuild to install) ==="
        fi
        do_finalize
        echo "=== Toolchain build complete ==="
        ;;
    llvm-toolchain-only)
        do_configure
        do_finalize
        echo "=== LLVM toolchain build complete ==="
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
