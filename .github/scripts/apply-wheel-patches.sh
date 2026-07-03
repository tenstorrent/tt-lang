#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Apply every wheel patch present in .github/wheel-patches/ to the checked-out
# tree before building a tt-lang wheel. Patches let a wheel built from an older
# tt-lang ref pick up targeted fixes from the current line (e.g. a corrected
# numpy pin) without rebasing the whole checkout. The set of patches is
# discovered, not enumerated here: drop a self-contained *.sh in that directory
# and it runs. Patches run in sorted filename order, from the repository root.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
patches_dir="$repo_root/.github/wheel-patches"

cd "$repo_root"

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

for patch in "${patches[@]}"; do
    echo "Applying wheel patch: $(basename "$patch")"
    bash "$patch"
done

echo "Applied ${#patches[@]} wheel patch(es)."
