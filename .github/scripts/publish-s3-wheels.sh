#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Upload every wheel under <dist_dir> to the top-level tt-lang/ wheel directory.
# tt-lang/<YYYY-MM>/ and tt-lang/releases/ are generated find-links views over
# those top-level wheel objects, not physical copies. With --overwrite, replace
# existing direct wheel objects.
#
# Usage: publish-s3-wheels.sh [--overwrite] [--overwrite-if true|false] --prefix <tt-lang/YYYY-MM|tt-lang/releases> <dist_dir>

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/s3-index.sh
. "$script_dir/lib/s3-index.sh"
# shellcheck source=lib/s3-publish-prefix.sh
. "$script_dir/lib/s3-publish-prefix.sh"

usage() {
    echo "Usage: $0 [--overwrite] [--overwrite-if true|false] --prefix <tt-lang/YYYY-MM|tt-lang/releases> <dist_dir>" >&2
    exit 2
}

overwrite=0
prefix=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite)
            overwrite=1
            shift
            ;;
        --overwrite-if)
            [[ $# -ge 2 ]] || usage
            case "$2" in
                true) overwrite=1 ;;
                false) overwrite=0 ;;
                *) usage ;;
            esac
            shift 2
            ;;
        --prefix)
            [[ $# -ge 2 ]] || usage
            prefix="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -ne 1 || -z "$prefix" ]]; then
    usage
fi
if ! ttlang_s3_valid_publish_prefix "$prefix"; then
    echo "Publish prefix must be tt-lang/<YYYY-MM> or tt-lang/releases: $prefix" >&2
    exit 2
fi

dist_dir="$1"
bucket="${TTLANG_S3_BUCKET:-tenstorrent-pypi}"

assert_object_absent_or_overwrite() {
    local key="$1" head_error
    if [[ "$overwrite" -eq 1 ]]; then
        return 0
    fi
    head_error="$(mktemp)"
    if aws s3api head-object --bucket "$bucket" --key "$key" >/dev/null 2>"$head_error"; then
        rm -f "$head_error"
        echo "S3 object already exists: s3://$bucket/$key (pass --overwrite to replace it)." >&2
        exit 1
    fi
    if grep -Eq 'NoSuchKey|Not Found|404|does not exist' "$head_error"; then
        rm -f "$head_error"
        return 0
    fi
    cat "$head_error" >&2
    rm -f "$head_error"
    exit 1
}

shopt -s nullglob
wheels=("$dist_dir"/*.whl)
if [[ "${#wheels[@]}" -eq 0 ]]; then
    echo "No wheels found under $dist_dir" >&2
    exit 1
fi

top_level="$(dirname "$prefix")"

for wheel in "${wheels[@]}"; do
    wheel_key="$top_level/$(basename "$wheel")"
    assert_object_absent_or_overwrite "$wheel_key"
done

for wheel in "${wheels[@]}"; do
    wheel_key="$top_level/$(basename "$wheel")"
    aws s3 cp "$wheel" "s3://$bucket/$wheel_key" \
        --content-type "application/octet-stream"
done

if [[ "$prefix" == "tt-lang/releases" ]]; then
    s3_regenerate_release_view "$bucket"
else
    s3_regenerate_month_view "$bucket" "$prefix"
fi
s3_regenerate_index --directories-only --hidden-stable-wheels "$bucket" "$top_level"
