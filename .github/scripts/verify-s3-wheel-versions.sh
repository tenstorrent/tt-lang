#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify the wheel versions produced by the S3 PyPI publish workflow. The light
# variant publishes a tt-lang wheel with a +light local version plus the
# tt-lang-light metapackage; bundled and pypi variants publish all wheels at the
# requested internal version.
#
# Usage: verify-s3-wheel-versions.sh [--no-sim] <wheel_variant> <version_override> <dist_dir>

set -euo pipefail

usage() {
    echo "Usage: $0 [--no-sim] <wheel_variant> <version_override> <dist_dir>" >&2
    exit 2
}

include_sim=1
if [[ "${1:-}" == "--no-sim" ]]; then
    include_sim=0
    shift
fi

if [[ $# -ne 3 ]]; then
    usage
fi

variant="$1"
version="$2"
dist_dir="$3"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$variant" in
    light)
        verify_args=(
            --expect "tt_lang=$version+light"
            --expect "tt_lang_light=$version"
        )
        if [[ "$include_sim" -eq 1 ]]; then
            verify_args+=(--expect "tt_lang_sim=$version")
        fi
        "$script_dir/verify-wheel-version.sh" "${verify_args[@]}" "$dist_dir"
        ;;
    bundled | pypi)
        "$script_dir/verify-wheel-version.sh" "$version" "$dist_dir"
        ;;
    *)
        echo "Unknown wheel variant: $variant" >&2
        exit 2
        ;;
esac
