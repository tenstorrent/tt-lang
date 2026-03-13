#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Normalize toolchain installation by replacing symlinks with actual files.
# This makes the installation self-contained for caching and artifact archiving.
#
# Usage: normalize-toolchain-install.sh <install-dir>

set -euo pipefail

INSTALL_DIR="${1:?Usage: $0 <install-dir>}"

if [ ! -d "$INSTALL_DIR" ]; then
    echo "Error: Directory '$INSTALL_DIR' does not exist"
    exit 1
fi

echo "Normalizing toolchain installation at: $INSTALL_DIR"

ABS_INSTALL_DIR=$(cd "$INSTALL_DIR" && pwd)

# The cmake install creates a python_packages/ directory that contains the
# canonical copies of all Python packages. It also creates top-level symlinks
# (e.g., ttrt/ -> python_packages/ttrt/) for build-tree compatibility.
#
# We want to keep only python_packages/ and remove the duplicate top-level
# symlinks, then normalize any remaining symlinks that point outside the
# install dir.

# Pass 1: Remove top-level symlinks that point into python_packages/ (duplicates).
for link in "$INSTALL_DIR"/*/; do
    link="${link%/}"  # strip trailing slash
    [ -L "$link" ] || continue
    target=$(readlink -f "$link")
    case "$target" in
        "$ABS_INSTALL_DIR/python_packages"*)
            echo "  Removing duplicate symlink: $link -> $target"
            rm "$link"
            ;;
    esac
done

# Pass 2: Replace remaining symlinks with actual files.
mapfile -t symlinks < <(find "$INSTALL_DIR" -type l)

echo "Found ${#symlinks[@]} symlinks to normalize"

for link in "${symlinks[@]}"; do
    target=$(readlink -f "$link" 2>/dev/null) || true
    if [ -n "$target" ] && [ -e "$target" ]; then
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

# Ensure venv has a 'python' symlink (some venvs only create python3).
if [ -d "$INSTALL_DIR/venv/bin" ] && [ ! -e "$INSTALL_DIR/venv/bin/python" ]; then
    ln -s python3 "$INSTALL_DIR/venv/bin/python"
    echo "  Created python -> python3 symlink in venv"
fi

echo "Normalization complete."
