#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Cleanup the toolchain directory to reduce Docker image size.
# Run BEFORE packaging into Docker images (not inside the Dockerfile).
#
# Usage:
#   cleanup-toolchain.sh <toolchain-dir>            # conservative (ird-safe)
#   cleanup-toolchain.sh <toolchain-dir> --dist     # aggressive (dist only)

set -e

TOOLCHAIN_DIR="${1:?Usage: $0 <toolchain-dir> [--dist]}"
DIST_MODE=false
[ "${2:-}" = "--dist" ] && DIST_MODE=true

echo "Cleaning up toolchain at: $TOOLCHAIN_DIR (dist=$DIST_MODE)"
du -sh "$TOOLCHAIN_DIR" 2>/dev/null || true

# ---- venv/lib64 dedup ----
if [ -d "$TOOLCHAIN_DIR/venv/lib64" ] && [ ! -L "$TOOLCHAIN_DIR/venv/lib64" ]; then
    echo "Removing duplicate venv/lib64 directory"
    rm -rf "$TOOLCHAIN_DIR/venv/lib64"
    ln -s lib "$TOOLCHAIN_DIR/venv/lib64"
fi

# ---- Strip unnecessary LLVM binaries ----
# tt-lang needs these binaries to actually function:
#   FileCheck, not, count (lit testing), llvm-lit, mlir-opt, mlir-translate
#   (debugging), llvm-tblgen, mlir-tblgen, mlir-linalg-ods-yaml-gen (cmake
#   configure / tablegen).
# Everything else is replaced with empty executable stubs so that cmake's
# imported-target existence checks still pass.
KEEP_BINS=(
    FileCheck
    not
    count
    llvm-lit
    llvm-tblgen
    mlir-linalg-ods-yaml-gen
    mlir-opt
    mlir-tblgen
    mlir-translate
    ttlang-sim
)

if [ -d "$TOOLCHAIN_DIR/bin" ]; then
    echo "Replacing unnecessary LLVM binaries with stubs..."
    _before=$(du -sm "$TOOLCHAIN_DIR/bin" | cut -f1)

    for f in "$TOOLCHAIN_DIR/bin"/*; do
        [ -f "$f" ] || continue
        _name=$(basename "$f")
        _keep=false
        for _k in "${KEEP_BINS[@]}"; do
            if [ "$_name" = "$_k" ]; then
                _keep=true
                break
            fi
        done
        if [ "$_keep" = false ]; then
            rm -f "$f"
            touch "$f"
            chmod +x "$f"
        fi
    done

    # Strip debug symbols from kept binaries
    for _k in "${KEEP_BINS[@]}"; do
        [ -f "$TOOLCHAIN_DIR/bin/$_k" ] && strip --strip-unneeded "$TOOLCHAIN_DIR/bin/$_k" 2>/dev/null || true
    done

    _after=$(du -sm "$TOOLCHAIN_DIR/bin" | cut -f1)
    echo "  bin/: ${_before}M -> ${_after}M (saved $((_before - _after))M)"
fi

# ---- Dist-only: remove static libs, LLVM source ----
if [ "$DIST_MODE" = true ]; then
    if [ -d "$TOOLCHAIN_DIR/src" ]; then
        echo "Removing src/ (not needed at runtime)"
        rm -rf "$TOOLCHAIN_DIR/src"
    fi

    if [ -d "$TOOLCHAIN_DIR/lib" ]; then
        echo "Removing static libraries (.a) from lib/..."
        _before=$(du -sm "$TOOLCHAIN_DIR/lib" | cut -f1)
        find "$TOOLCHAIN_DIR/lib" -name '*.a' -delete
        _after=$(du -sm "$TOOLCHAIN_DIR/lib" | cut -f1)
        echo "  lib/: ${_before}M -> ${_after}M (saved $((_before - _after))M)"
    fi
fi

# ---- Python cache cleanup ----
echo "Removing Python cache files..."
find "$TOOLCHAIN_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TOOLCHAIN_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true

echo ""
du -sh "$TOOLCHAIN_DIR" 2>/dev/null || true
echo "Toolchain cleanup complete"
