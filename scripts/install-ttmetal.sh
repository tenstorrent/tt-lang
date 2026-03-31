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

# --- Shared libraries ---
# Flatten all .so files from the build tree into lib/.  This covers libraries
# from subdirectories like tt_stl/, _deps/fmt-build/, tt_metal/third_party/umd/,
# etc. without hard-coding each path.  CMakeFiles/ directories are excluded
# (build intermediates only).
mkdir -p "$INSTALL/lib"
find "$BUILD" \( -name "*.so" -o -name "*.so.*" \) -not -path "*/CMakeFiles/*" \
    -exec cp -pL {} "$INSTALL/lib/" \;
echo "Installed shared libraries into lib/"

# --- Tracy profiler tools ---
TRACY_BIN="$BUILD/tools/profiler/bin"
if [ -d "$TRACY_BIN" ]; then
    mkdir -p "$INSTALL/tools/profiler/bin"
    cp -p "$TRACY_BIN/capture-release" "$INSTALL/tools/profiler/bin/" 2>/dev/null || true
    cp -p "$TRACY_BIN/csvexport-release" "$INSTALL/tools/profiler/bin/" 2>/dev/null || true
    echo "Installed Tracy profiler tools"
fi

# --- ttnn Python package ---
# Preserve the two-level ttnn/ttnn/ layout from the source tree so that
# PYTHONPATH points to the outer ttnn/ directory.  This avoids ttnn's
# types.py shadowing the stdlib types module.
if [ -d "$SRC/ttnn/ttnn" ]; then
    rm -rf "$INSTALL/python_packages/ttnn"
    mkdir -p "$INSTALL/python_packages/ttnn/ttnn"
    cp -prL "$SRC/ttnn/ttnn/"* "$INSTALL/python_packages/ttnn/ttnn/" 2>/dev/null || true
    echo "Installed ttnn Python package"
fi

# --- Tracy Python module ---
# Mirrors source layout: tools/ contains the tracy package, added to
# PYTHONPATH as a separate entry.
if [ -d "$SRC/tools/tracy" ]; then
    rm -rf "$INSTALL/python_packages/tools"
    mkdir -p "$INSTALL/python_packages/tools/tracy"
    cp -pr "$SRC/tools/tracy/"*.py "$INSTALL/python_packages/tools/tracy/" 2>/dev/null || true
    echo "Installed Tracy Python module"
fi

# --- Runtime artifacts (linker scripts, LLK headers, SoC/core descriptors, sfpi) ---
# Some artifacts are build-generated (runtime/hw), others live only in the
# source tree (runtime/sfpi).  Copy from the build dir first for
# build-generated artifacts, then fill in anything missing from source.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COPY_SCRIPT="$SCRIPT_DIR/copy-ttmetal-runtime-artifacts.sh"

if [ -x "$COPY_SCRIPT" ]; then
    if [ -d "$BUILD/runtime/hw" ]; then
        bash "$COPY_SCRIPT" "$BUILD" "$INSTALL"
    fi
    bash "$COPY_SCRIPT" --restore "$SRC" "$INSTALL"
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
