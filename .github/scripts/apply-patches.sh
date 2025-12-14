#!/usr/bin/env bash
# Apply patches matching a glob pattern to a target directory.
# Usage: apply-patches.sh <target-dir> <patch-dir> <glob-pattern>
# Example: apply-patches.sh /path/to/tt-mlir /path/to/patches "tt-mlir*.patch"

set -euo pipefail

TARGET_DIR="$1"
PATCH_DIR="$2"
PATTERN="$3"

shopt -s nullglob
patches=("$PATCH_DIR"/$PATTERN)
shopt -u nullglob

for patch in "${patches[@]}"; do
    name=$(basename "$patch")
    if git -C "$TARGET_DIR" apply --check "$patch" 2>/dev/null; then
        echo "Applying: $name"
        git -C "$TARGET_DIR" apply "$patch"
    else
        echo "Skipping (already applied or conflicts): $name"
    fi
done
