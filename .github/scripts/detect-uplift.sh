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

PATHS=(
    third-party/tt-metal-version
    third-party/llvm-project
    third-party/tt-mlir
    third-party/tt-metal
    .github/containers/Dockerfile.base
    pyproject.toml
    requirements-runtime.txt
)

CHANGED=$(git diff --name-only "$BASE" "$HEAD" -- "${PATHS[@]}")

if [[ -n "$CHANGED" ]]; then
    echo "uplift=true" >> "$GITHUB_OUTPUT"
    echo "Uplift detected:"
    printf '  %s\n' $CHANGED
else
    echo "uplift=false" >> "$GITHUB_OUTPUT"
    echo "No uplift-relevant changes."
fi
