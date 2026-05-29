#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify the wheel versions produced by the S3 PyPI publish workflow. External
# mode publishes a tt-lang wheel with a +light local version plus the
# tt-lang-light metapackage; bundled and pypi modes publish all wheels at the
# requested internal version.
#
# Usage: verify-s3-wheel-versions.sh <ttnn_dep_mode> <version_override> <dist_dir>

set -euo pipefail

usage() {
    echo "Usage: $0 <ttnn_dep_mode> <version_override> <dist_dir>" >&2
    exit 2
}

if [[ $# -ne 3 ]]; then
    usage
fi

mode="$1"
version="$2"
dist_dir="$3"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$mode" in
    external)
        "$script_dir/verify-wheel-version.sh" \
            --expect "tt_lang=$version+light" \
            --expect "tt_lang_light=$version" \
            --expect "tt_lang_sim=$version" \
            "$dist_dir"
        ;;
    bundled | pypi)
        "$script_dir/verify-wheel-version.sh" "$version" "$dist_dir"
        ;;
    *)
        echo "Unknown ttnn dependency mode: $mode" >&2
        exit 2
        ;;
esac
