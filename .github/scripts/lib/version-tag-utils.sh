#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

ttlang_compute_version_tag() {
    local path_list_description="$1"
    shift

    if [[ "$#" -eq 0 ]]; then
        echo "ERROR: path list is empty (defined in ${path_list_description})." >&2
        echo "  Without a path list, git diff and git ls-tree would scan the whole tree," >&2
        echo "  producing the hashed form for every commit." >&2
        return 1
    fi

    local repo_root
    repo_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
    if [ -z "$repo_root" ]; then
        echo "ERROR: Not inside a git repository." >&2
        return 1
    fi
    cd "$repo_root"

    # `|| true` keeps `set -e` from killing the script when there are no
    # matching tags; the empty-result case gets a specific diagnostic below.
    local nearest_tag_raw
    nearest_tag_raw=$(git describe --tags --match "v[0-9]*" --abbrev=0 2>/dev/null || true)
    if [ -z "$nearest_tag_raw" ]; then
        echo "ERROR: Could not determine version tag from git tags." >&2
        echo "  Ensure the CI checkout uses fetch-depth: 0 and fetch-tags: true." >&2
        return 1
    fi

    local nearest_tag
    nearest_tag=$(printf '%s' "$nearest_tag_raw" | tr '+' '-')

    if git diff --quiet "$nearest_tag_raw..HEAD" -- "$@"; then
        echo "$nearest_tag"
    else
        local hash
        hash=$(git ls-tree HEAD -- "$@" | sha256sum | cut -c1-8)
        echo "${nearest_tag}-${hash}"
    fi
}
