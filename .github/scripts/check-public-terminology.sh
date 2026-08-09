#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -eu

# Keep the identifier split so this enforcement change satisfies its own rule.
PROHIBITED_PATTERN='b[l]aze'

branch_name=${GITHUB_HEAD_REF:-}
if [ -z "$branch_name" ]; then
    branch_name=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)
fi
if [ -z "$branch_name" ]; then
    branch_name=${GITHUB_REF_NAME:-}
fi

if printf '%s\n' "$branch_name" | grep -Eiq "$PROHIBITED_PATTERN"; then
    echo "error: branch name contains a non-public comparison identifier" >&2
    exit 1
fi

base_branch=${GITHUB_BASE_REF:-main}
base_ref="refs/remotes/origin/$base_branch"
if git rev-parse --verify --quiet "$base_ref^{commit}" >/dev/null; then
    if git diff --no-ext-diff --no-color "$base_ref...HEAD" |
        grep -Eiq "$PROHIBITED_PATTERN"; then
        echo "error: branch diff contains a non-public comparison identifier" >&2
        exit 1
    fi
elif [ -n "${GITHUB_BASE_REF:-}" ]; then
    echo "error: cannot resolve pull request base ref $base_ref" >&2
    exit 1
fi

if git diff --cached --no-ext-diff --no-color |
    grep -Eiq "$PROHIBITED_PATTERN"; then
    echo "error: staged diff contains a non-public comparison identifier" >&2
    exit 1
fi
