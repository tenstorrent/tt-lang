#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Print the Docker version tag for the current branch state.
#
# Clean state (no uplift-relevant changes since the nearest version tag):
# the tag name itself, e.g. `vX.Y.Z`. Git tags may carry SemVer build
# metadata after `+` (e.g. vX.Y.Z+rcN); since Docker tags allow only
# [A-Za-z0-9_.-], `+` is translated to `-` (`vX.Y.Z-rcN`).
#
# Uplift state (uplift-relevant paths differ from the nearest tag): append
# `-uplift-<8char>` where the hash is derived from
# `git ls-tree HEAD -- <paths>`. Same submodule SHAs + Dockerfile/requirements
# content -> same hash, so independent PRs uplifting to the same toolchain
# resolve to the same docker tag and share the rebuilt image. The path list
# is defined in .github/scripts/uplift-paths.sh.
#
# Usage: .github/containers/get-version-tag.sh
# Must be run from a git repository with version tags (v[0-9]*) and full
# history (`fetch-depth: 0`, `fetch-tags: true` in CI checkouts).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/uplift-paths.sh
source "${SCRIPT_DIR}/../scripts/uplift-paths.sh"

if [[ ${#UPLIFT_PATHS[@]} -eq 0 ]]; then
    echo "ERROR: UPLIFT_PATHS is empty (defined in ${SCRIPT_DIR}/../scripts/uplift-paths.sh)." >&2
    echo "  Without a path list, git diff and git ls-tree would scan the whole tree," >&2
    echo "  producing the uplift form for every commit." >&2
    exit 1
fi

# Run from the repo root so UPLIFT_PATHS (relative to repo root) resolves
# consistently regardless of the caller's CWD.
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$REPO_ROOT" ]; then
    echo "ERROR: Not inside a git repository." >&2
    exit 1
fi
cd "$REPO_ROOT"

# `|| true` keeps `set -e` from killing the script when there are no matching
# tags; we handle the empty-result case explicitly below.
NEAREST_TAG_RAW=$(git describe --tags --match "v[0-9]*" --abbrev=0 2>/dev/null || true)
if [ -z "$NEAREST_TAG_RAW" ]; then
    echo "ERROR: Could not determine version tag from git tags." >&2
    echo "  Ensure the CI checkout uses fetch-depth: 0 and fetch-tags: true." >&2
    exit 1
fi
NEAREST_TAG=$(printf '%s' "$NEAREST_TAG_RAW" | tr '+' '-')

if git diff --quiet "$NEAREST_TAG_RAW..HEAD" -- "${UPLIFT_PATHS[@]}"; then
    echo "$NEAREST_TAG"
else
    HASH=$(git ls-tree HEAD -- "${UPLIFT_PATHS[@]}" | sha256sum | cut -c1-8)
    echo "${NEAREST_TAG}-uplift-${HASH}"
fi
