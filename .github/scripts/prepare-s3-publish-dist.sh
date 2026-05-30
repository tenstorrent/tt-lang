#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify variant-specific wheel artifact directories and copy their wheels into
# a single publish directory. A spec has the form <variant>[:no-sim]=<dist_dir>.
# Use `:no-sim` when the same workflow run publishes bundled wheels and the
# light artifact intentionally omits the duplicate tt-lang-sim wheel.
#
# Usage: prepare-s3-publish-dist.sh <version_override> <publish_dir> <spec>...

set -euo pipefail

usage() {
    echo "Usage: $0 <version_override> <publish_dir> <variant[:no-sim]=dist_dir>..." >&2
    exit 2
}

if [[ $# -lt 3 ]]; then
    usage
fi

version="$1"
publish_dir="$2"
shift 2
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$publish_dir" in
    "" | "." | "/")
        echo "Unsafe publish directory: $publish_dir" >&2
        exit 2
        ;;
esac

rm -rf "$publish_dir"
mkdir -p "$publish_dir"

for spec in "$@"; do
    if [[ "$spec" != *=* ]]; then
        usage
    fi
    variant_spec="${spec%%=*}"
    artifact_dir="${spec#*=}"
    verify_args=()

    if [[ "$variant_spec" == *:no-sim ]]; then
        variant="${variant_spec%:no-sim}"
        if [[ "$variant" != light ]]; then
            echo ":no-sim is only valid for the light variant: $spec" >&2
            exit 2
        fi
        verify_args+=(--no-sim)
    else
        variant="$variant_spec"
    fi

    case "$variant" in
        pypi | light | bundled) ;;
        *)
            echo "Unknown wheel variant: $variant" >&2
            exit 2
            ;;
    esac

    if [[ ! -d "$artifact_dir" ]]; then
        echo "Wheel artifact directory not found: $artifact_dir" >&2
        exit 1
    fi

    "$script_dir/verify-s3-wheel-versions.sh" \
        "${verify_args[@]}" \
        "$variant" \
        "$version" \
        "$artifact_dir"

    shopt -s nullglob
    wheels=("$artifact_dir"/*.whl)
    shopt -u nullglob
    if [[ "${#wheels[@]}" -eq 0 ]]; then
        echo "No wheels found under $artifact_dir" >&2
        exit 1
    fi

    for wheel in "${wheels[@]}"; do
        wheel_name="$(basename "$wheel")"
        target="$publish_dir/$wheel_name"
        if [[ -e "$target" ]]; then
            echo "Duplicate wheel filename across S3 publish artifacts: $wheel_name" >&2
            exit 1
        fi
        cp "$wheel" "$target"
    done
done
