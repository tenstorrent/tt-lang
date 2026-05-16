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

CHANGED=$(git diff --name-only "$BASE" "$HEAD" -- "${UPLIFT_PATHS[@]}")

if [[ -n "$CHANGED" ]]; then
    echo "uplift=true" >> "$GITHUB_OUTPUT"
    echo "Uplift detected:"
    printf '  %s\n' $CHANGED
else
    echo "uplift=false" >> "$GITHUB_OUTPUT"
    echo "No uplift-relevant changes."
fi
