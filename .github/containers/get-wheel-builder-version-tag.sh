#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Print the manylinux wheel-builder image tag for the current branch state.
# The tag uses the same deterministic format as get-version-tag.sh but hashes
# the wheel-builder-specific path list.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/wheel-builder-tag-paths.sh
source "${SCRIPT_DIR}/../scripts/wheel-builder-tag-paths.sh"
# shellcheck source=../scripts/lib/version-tag-utils.sh
source "${SCRIPT_DIR}/../scripts/lib/version-tag-utils.sh"

ttlang_compute_version_tag \
    ".github/scripts/wheel-builder-tag-paths.sh" \
    "${WHEEL_BUILDER_TAG_PATHS[@]}"
