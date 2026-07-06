#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Apply every wheel patch in the sibling .github/wheel-patches/ directory to a
# tt-lang checkout before building a wheel. Patches let a wheel built from an
# older tt-lang ref pick up targeted fixes from the current line (e.g. a
# corrected numpy pin) without rebasing the whole checkout.
#
# The patch runner and the patch files come from the workflow commit, not the
# ref being rebuilt: an older ref that needs a patch predates both. So the tree
# to patch is passed with --target-dir and is independent of where this script
# and its patches live. With no --target-dir, the script's own repository root
# is patched.
#
# Patches are discovered, not enumerated: drop a self-contained *.sh in
# .github/wheel-patches/ and it runs. Each patch runs with the target tree as
# its working directory, in sorted filename order.
#
# Usage: apply-wheel-patches.sh [--target-dir <dir>]

set -euo pipefail

target_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-dir)
            [[ $# -ge 2 ]] || { echo "usage: apply-wheel-patches.sh [--target-dir <dir>]" >&2; exit 2; }
            target_dir="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            echo "usage: apply-wheel-patches.sh [--target-dir <dir>]" >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
patches_dir="$repo_root/.github/wheel-patches"

if [ -z "$target_dir" ]; then
    target_dir="$repo_root"
fi
target_dir="$(cd "$target_dir" && pwd)"

if [ ! -d "$patches_dir" ]; then
    echo "No wheel-patches directory at $patches_dir; nothing to apply."
    exit 0
fi

shopt -s nullglob
patches=("$patches_dir"/*.sh)
shopt -u nullglob

if [ "${#patches[@]}" -eq 0 ]; then
    echo "No wheel patches present; nothing to apply."
    exit 0
fi

cd "$target_dir"
for patch in "${patches[@]}"; do
    echo "Applying wheel patch to $target_dir: $(basename "$patch")"
    bash "$patch"
done

echo "Applied ${#patches[@]} wheel patch(es) to $target_dir."
