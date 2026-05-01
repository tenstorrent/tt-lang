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

# Make Python shebangs relocatable: replace the absolute-path shebang with an
# sh/Python polyglot that execs python3 via a path computed relative to the
# script's own directory. Caller passes the relative path to python3.
rewrite_python_shebang() {
    local script="$1"
    local rel_python="$2"
    local first second
    first=$(sed -n '1p' "$script" 2>/dev/null) || return 0
    second=$(sed -n '2p' "$script" 2>/dev/null)
    local skip=0
    case "$first" in
        '#!'*python|'#!'*python[0-9]*|'#!'*python[0-9].[0-9]*)
            skip=1
            ;;
        '#!/bin/sh')
            # Re-rewrite an existing polyglot (idempotent + path correction).
            if [ "$second" = '""":"' ]; then
                skip=4
            else
                return 0
            fi
            ;;
        *)
            return 0
            ;;
    esac
    local tmp
    tmp=$(mktemp)
    {
        printf '#!/bin/sh\n""":"\nexec "$(dirname "$(readlink -f "$0")")/%s" "$0" "$@"\n"""\n' "$rel_python"
        tail -n "+$((skip + 1))" "$script"
    } > "$tmp"
    chmod --reference="$script" "$tmp"
    mv "$tmp" "$script"
    echo "  Rewrote shebang: $script"
}

# Each candidate dir maps to the relative path from that dir to the venv's
# python3. venv/bin scripts use a sibling python3; top-level bin scripts must
# walk up and into venv/bin.
echo "Rewriting Python shebangs to be relocatable"
declare -A SHEBANG_DIRS=(
    ["$INSTALL_DIR/venv/bin"]="python3"
    ["$INSTALL_DIR/bin"]="../venv/bin/python3"
)
for dir in "${!SHEBANG_DIRS[@]}"; do
    [ -d "$dir" ] || continue
    rel_python="${SHEBANG_DIRS[$dir]}"
    for script in "$dir"/*; do
        # Skip non-files and binaries (text scripts only).
        [ -f "$script" ] && [ ! -L "$script" ] || continue
        # Cheap filter: must start with "#!".
        head -c 2 "$script" 2>/dev/null | grep -q '^#!' || continue
        rewrite_python_shebang "$script" "$rel_python"
    done
done

echo "Normalization complete."
