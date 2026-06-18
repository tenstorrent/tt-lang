#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Writes wheel_or_container=true|false to $GITHUB_OUTPUT based on whether
# git diff <base>..<head> touches any path in wheel-or-container-paths.sh.
# Missing args or a base that doesn't resolve to a commit emits true.
#
# Usage: wheel-or-container-changed.sh <base-sha> <head-sha>

set -euo pipefail

BASE=${1:-}
HEAD=${2:-}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wheel-or-container-paths.sh"

cd "$(git rev-parse --show-toplevel)"

if [[ -z "$BASE" || -z "$HEAD" ]] || ! git cat-file -e "${BASE}^{commit}" 2>/dev/null; then
    echo "wheel_or_container=true" >> "$GITHUB_OUTPUT"
    echo "No comparable prior commit; running wheel/container coverage."
    exit 0
fi

mapfile -t CHANGED < <(git diff --name-only "$BASE" "$HEAD" -- "${WHEEL_OR_CONTAINER_PATHS[@]}")

if [[ ${#CHANGED[@]} -gt 0 ]]; then
    echo "wheel_or_container=true" >> "$GITHUB_OUTPUT"
    printf 'Changed: %s\n' "${CHANGED[@]}"
else
    echo "wheel_or_container=false" >> "$GITHUB_OUTPUT"
fi
