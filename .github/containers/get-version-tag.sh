#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Print the Docker version tag for the current branch state.
#
# Clean state (no container image content changes since the nearest version tag):
# the tag name itself, e.g. `vX.Y.Z`. Git tags may carry SemVer build
# metadata after `+` (e.g. vX.Y.Z+rcN); since Docker tags allow only
# [A-Za-z0-9_.-], `+` is translated to `-` (`vX.Y.Z-rcN`).
#
# Modified container-input state: append `-<8char>` where the hash is derived from
# `git ls-tree HEAD -- <paths>`. Same submodule SHAs + Dockerfile/requirements
# content -> same hash, so independent PRs with the same container inputs
# resolve to the same Docker tag and share the rebuilt image. The path list is
# defined in .github/scripts/uplift-paths.sh.
#
# Usage: .github/containers/get-version-tag.sh
# Must be run from a git repository with version tags (v[0-9]*) and full
# history (`fetch-depth: 0`, `fetch-tags: true` in CI checkouts).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/uplift-paths.sh
source "${SCRIPT_DIR}/../scripts/uplift-paths.sh"
# shellcheck source=../scripts/lib/version-tag-utils.sh
source "${SCRIPT_DIR}/../scripts/lib/version-tag-utils.sh"

ttlang_compute_version_tag \
    ".github/scripts/uplift-paths.sh" \
    "${UPLIFT_PATHS[@]}"
