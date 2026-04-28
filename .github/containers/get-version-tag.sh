#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Print the Docker version tag derived from the nearest git version tag.
# E.g., if the nearest tag is v0.1.9, prints "v0.1.9".
#
# Git tags may carry SemVer build metadata after '+' (e.g., v1.0.0+uplift)
# but Docker tags allow only [A-Za-z0-9_.-]; '+' is translated to '-' so the
# Docker image tag for v1.0.0+uplift is v1.0.0-uplift.
#
# Usage: .github/containers/get-version-tag.sh
# Must be run from a git repository with version tags (v[0-9]*).

set -e

TAG=$(git describe --tags --match "v[0-9]*" --abbrev=0 2>/dev/null | sed 's#[/:+]#-#g')
if [ -z "$TAG" ]; then
    echo "ERROR: Could not determine version tag from git tags." >&2
    exit 1
fi
echo "$TAG"
