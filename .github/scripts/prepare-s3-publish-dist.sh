#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify variant-specific wheel artifact directories and copy their wheels into
# a single publish directory. A spec has the form <variant>[:no-sim]=<dist_dir>.
# Use `:no-sim` for light artifacts that intentionally omit tt-lang-sim.
#
# Usage: prepare-s3-publish-dist.sh <version_override> <publish_dir> <spec>...

set -eu

usage() {
    echo "Usage: $0 <version_override> <publish_dir> <variant[:no-sim]=dist_dir>..." >&2
    exit 2
}

if [ "$#" -lt 3 ]; then
    usage
fi

version="$1"
publish_dir="$2"
shift 2
script_dir="$(cd "$(dirname "$0")" && pwd)"

case "$publish_dir" in
    "" | "." | "/")
        echo "Unsafe publish directory: $publish_dir" >&2
        exit 2
        ;;
esac

rm -rf "$publish_dir"
mkdir -p "$publish_dir"

for spec in "$@"; do
    case "$spec" in
        *=*) ;;
        *) usage ;;
    esac
    variant_spec="${spec%%=*}"
    artifact_dir="${spec#*=}"
    no_sim=false

    case "$variant_spec" in
        *:no-sim)
            variant="${variant_spec%:no-sim}"
            if [ "$variant" != light ]; then
                echo ":no-sim is only valid for the light variant: $spec" >&2
                exit 2
            fi
            no_sim=true
            ;;
        *)
            variant="$variant_spec"
            ;;
    esac

    case "$variant" in
        pypi | light | bundled) ;;
        *)
            echo "Unknown wheel variant: $variant" >&2
            exit 2
            ;;
    esac

    if [ ! -d "$artifact_dir" ]; then
        echo "Wheel artifact directory not found: $artifact_dir" >&2
        exit 1
    fi

    if [ "$no_sim" = true ]; then
        "$script_dir/verify-s3-wheel-versions.sh" \
            --no-sim \
            "$variant" \
            "$version" \
            "$artifact_dir"
    else
        "$script_dir/verify-s3-wheel-versions.sh" \
            "$variant" \
            "$version" \
            "$artifact_dir"
    fi

    seen_wheel=false
    for wheel in "$artifact_dir"/*.whl; do
        if [ ! -e "$wheel" ]; then
            continue
        fi
        seen_wheel=true
        wheel_name="$(basename "$wheel")"
        target="$publish_dir/$wheel_name"
        if [ -e "$target" ]; then
            echo "Duplicate wheel filename across S3 publish artifacts: $wheel_name" >&2
            exit 1
        fi
        cp "$wheel" "$target"
    done
    if [ "$seen_wheel" = false ]; then
        echo "No wheels found under $artifact_dir" >&2
        exit 1
    fi
done
