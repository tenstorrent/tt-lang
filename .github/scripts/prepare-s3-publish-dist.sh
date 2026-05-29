#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify mode-specific wheel artifact directories and copy their wheels into a
# single publish directory. A spec has the form <mode>[:no-sim]=<dist_dir>.
# Use `:no-sim` when the same workflow run publishes bundled wheels and the
# external artifact intentionally omits the duplicate tt-lang-sim wheel.
#
# Usage: prepare-s3-publish-dist.sh <version_override> <publish_dir> <spec>...

set -euo pipefail

usage() {
    echo "Usage: $0 <version_override> <publish_dir> <mode[:no-sim]=dist_dir>..." >&2
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
    mode_spec="${spec%%=*}"
    artifact_dir="${spec#*=}"
    verify_args=()

    if [[ "$mode_spec" == *:no-sim ]]; then
        verify_args+=(--no-sim)
        mode="${mode_spec%:no-sim}"
    else
        mode="$mode_spec"
    fi

    case "$mode" in
        pypi | external | bundled) ;;
        *)
            echo "Unknown ttnn dependency mode: $mode" >&2
            exit 2
            ;;
    esac

    if [[ ! -d "$artifact_dir" ]]; then
        echo "Wheel artifact directory not found: $artifact_dir" >&2
        exit 1
    fi

    "$script_dir/verify-s3-wheel-versions.sh" \
        "${verify_args[@]}" \
        "$mode" \
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
