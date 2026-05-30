#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Decide whether a build needs a from-source toolchain rebuild ("uplift").
# Writes `uplift=true|false` to $GITHUB_OUTPUT.
#
# Usage: detect-uplift.sh <base-sha> <head-sha>

set -euo pipefail

BASE=${1:?missing base sha}
HEAD=${2:?missing head sha}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/uplift-paths.sh"

# Run from the repo root so UPLIFT_PATHS (relative to repo root) resolves
# consistently regardless of the caller's CWD. Without this, `git diff --
# <paths>` from a subdirectory interprets the paths relative to the
# subdirectory and silently produces an empty result.
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$REPO_ROOT" ]; then
    echo "ERROR: Not inside a git repository." >&2
    exit 1
fi
cd "$REPO_ROOT"

mapfile -t CHANGED < <(git diff --name-only "$BASE" "$HEAD" -- "${UPLIFT_PATHS[@]}")

if [[ ${#CHANGED[@]} -gt 0 ]]; then
    echo "uplift=true" >> "$GITHUB_OUTPUT"
    echo "Uplift detected:"
    printf '  %s\n' "${CHANGED[@]}"
else
    echo "uplift=false" >> "$GITHUB_OUTPUT"
    echo "No uplift-relevant changes."
fi
