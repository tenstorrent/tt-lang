#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Refuse to run unless GITHUB_REF points at a v[0-9]* release tag, then print
# the version (tag with leading 'v' stripped) for downstream steps.
#
# Usage: .github/scripts/require-release-tag.sh
# Reads:  $GITHUB_REF
# Writes: tag_version=<MAJOR.MINOR.PATCH...> to $GITHUB_OUTPUT (if set)
# Stdout: <MAJOR.MINOR.PATCH...>

set -euo pipefail

ref="${GITHUB_REF:-}"
if [[ ! "$ref" =~ ^refs/tags/v[0-9] ]]; then
    echo "This workflow must be dispatched from a v* tag (got '$ref')." >&2
    echo "Create and push a tag like 'v1.1.2', then dispatch from that tag." >&2
    exit 1
fi

tag_version="${ref#refs/tags/v}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "tag_version=${tag_version}" >> "$GITHUB_OUTPUT"
fi
echo "$tag_version"
