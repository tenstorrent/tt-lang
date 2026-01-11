#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Normalize tt-mlir-install by replacing symlinks with actual files.
# This makes the installation self-contained for caching and artifact archiving.
#
# Usage: normalize-ttmlir-install.sh <tt-mlir-install-dir>

set -euo pipefail

INSTALL_DIR="${1:?Usage: $0 <tt-mlir-install-dir>}"

if [ ! -d "$INSTALL_DIR" ]; then
    echo "Error: Directory '$INSTALL_DIR' does not exist"
    exit 1
fi

echo "Normalizing tt-mlir installation at: $INSTALL_DIR"

# Find all symlinks first (before we start modifying the filesystem)
mapfile -t symlinks < <(find "$INSTALL_DIR" -type l)

echo "Found ${#symlinks[@]} symlinks to process"

# Replace each symlink with its target
for link in "${symlinks[@]}"; do
    target=$(readlink -f "$link")
    if [ -e "$target" ]; then
        # Remove symlink and copy the target
        rm "$link"
        if [ -d "$target" ]; then
            cp -r "$target" "$link"
        else
            cp "$target" "$link"
        fi
        echo "  Copied: $link"
    else
        echo "  Warning: Broken symlink (target missing): $link -> $target"
    fi
done

echo "Normalization complete."
