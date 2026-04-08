#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Print the Docker version tag derived from the nearest git version tag.
# E.g., if the nearest tag is v0.1.8, prints "v0.1.8".
#
# Usage: .github/containers/get-version-tag.sh
# Must be run from a git repository with version tags (v[0-9]*).

set -e

TAG=$(git describe --tags --match "v[0-9]*" --abbrev=0 2>/dev/null | sed 's/[\/:]/-/g')
if [ -z "$TAG" ]; then
    echo "ERROR: Could not determine version tag from git tags." >&2
    exit 1
fi
echo "$TAG"
