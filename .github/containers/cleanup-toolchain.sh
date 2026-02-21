#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Post-install fixups for the tt-mlir toolchain directory
# (e.g. deduplicating venv lib64)

set -e

TOOLCHAIN_DIR="${1:?Usage: $0 <toolchain-dir>}"

echo "Cleaning up toolchain at: $TOOLCHAIN_DIR"

# Remove duplicate lib64 directory in venv (replace with symlink to lib)
if [ -d "$TOOLCHAIN_DIR/venv/lib64" ] && [ -d "$TOOLCHAIN_DIR/venv/lib" ]; then
    echo "Removing duplicate venv/lib64 directory"
    rm -rf "$TOOLCHAIN_DIR/venv/lib64"
    ln -s lib "$TOOLCHAIN_DIR/venv/lib64"
fi

echo "Toolchain cleanup complete"
