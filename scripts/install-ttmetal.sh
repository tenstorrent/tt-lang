#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Install tt-metal build artifacts into a toolchain prefix.
#
# Copies only the shared libraries, Python packages, profiler tools, and
# runtime artifacts needed at configure/run time.  Object files, CMake
# caches, CPM source caches, and other build intermediates are excluded.
#
# Usage:
#   install-ttmetal.sh <tt-metal-source-dir> <tt-metal-build-dir> <install-dir>

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <tt-metal-source-dir> <tt-metal-build-dir> <install-dir>"
    exit 1
fi

SRC="$1"
BUILD="$2"
INSTALL="$3"

echo "=== Installing tt-metal artifacts ==="
echo "  Source:  $SRC"
echo "  Build:   $BUILD"
echo "  Install: $INSTALL"

# --- Shared libraries from lib/ ---
if [ -d "$BUILD/lib" ]; then
    mkdir -p "$INSTALL/lib"
    cp -pL "$BUILD"/lib/*.so* "$INSTALL/lib/" 2>/dev/null || true
    echo "Installed lib/*.so*"
fi

# --- ttnn shared libraries ---
if [ -d "$BUILD/ttnn" ]; then
    mkdir -p "$INSTALL/ttnn"
    find "$BUILD/ttnn" -maxdepth 1 -name "*.so" -exec cp -pL {} "$INSTALL/ttnn/" \;
    echo "Installed ttnn/*.so"
fi

# --- tt_metal shared libraries ---
if [ -d "$BUILD/tt_metal" ]; then
    so_files=$(find "$BUILD/tt_metal" -maxdepth 1 -name "*.so" 2>/dev/null)
    if [ -n "$so_files" ]; then
        mkdir -p "$INSTALL/tt_metal"
        echo "$so_files" | while read -r f; do cp -pL "$f" "$INSTALL/tt_metal/"; done
        echo "Installed tt_metal/*.so"
    fi
fi

# --- Tracy profiler tools ---
TRACY_BIN="$BUILD/tools/profiler/bin"
if [ -d "$TRACY_BIN" ]; then
    mkdir -p "$INSTALL/tools/profiler/bin"
    cp -p "$TRACY_BIN/capture-release" "$INSTALL/tools/profiler/bin/" 2>/dev/null || true
    cp -p "$TRACY_BIN/csvexport-release" "$INSTALL/tools/profiler/bin/" 2>/dev/null || true
    echo "Installed Tracy profiler tools"
fi

# --- ttnn Python package ---
if [ -d "$SRC/ttnn/ttnn" ]; then
    mkdir -p "$INSTALL/python_packages/ttnn"
    cp -prL "$SRC/ttnn/ttnn/"* "$INSTALL/python_packages/ttnn/" 2>/dev/null || true
    echo "Installed ttnn Python package"
fi

# --- Tracy Python module ---
if [ -d "$SRC/tools/tracy" ]; then
    mkdir -p "$INSTALL/python_packages/tracy"
    cp -pr "$SRC/tools/tracy/"*.py "$INSTALL/python_packages/tracy/" 2>/dev/null || true
    echo "Installed Tracy Python module"
fi

# --- Runtime artifacts (linker scripts, LLK headers, SoC/core descriptors) ---
# These are copied into the build dir during cmake configure by
# copy-ttmetal-runtime-artifacts.sh.  Re-copy from whichever location has them.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COPY_SCRIPT="$SCRIPT_DIR/copy-ttmetal-runtime-artifacts.sh"

if [ -x "$COPY_SCRIPT" ]; then
    # The copy script expects <src> <dest>.  Runtime artifacts live in the
    # build dir (copied there during configure) or in the source tree.
    # Try the build dir first, fall back to source.
    if [ -d "$BUILD/runtime/hw" ]; then
        bash "$COPY_SCRIPT" "$BUILD" "$INSTALL"
    else
        bash "$COPY_SCRIPT" "$SRC" "$INSTALL"
    fi
else
    echo "WARNING: copy-ttmetal-runtime-artifacts.sh not found at $COPY_SCRIPT"
fi

# --- JIT source trees (headers and firmware .cc files) ---
# The JIT build system resolves these via TT_METAL_HOME at device runtime.
# The toolchain must contain the full tt_metal/ and ttnn/cpp/ subtrees (~99 MB).
if [ -d "$SRC/tt_metal" ]; then
    echo "Installing tt-metal JIT source tree..."
    cp -a "$SRC/tt_metal" "$INSTALL/"
    mkdir -p "$INSTALL/ttnn"
    cp -a "$SRC/ttnn/cpp" "$INSTALL/ttnn/"
    echo "Installed JIT source tree"
fi

echo "=== tt-metal install complete ==="
du -sh "$INSTALL" 2>/dev/null || true
