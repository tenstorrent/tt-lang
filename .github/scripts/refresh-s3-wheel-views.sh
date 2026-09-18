#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "Usage: $0 --months YYYY-MM[,YYYY-MM...]" >&2
    exit 2
}

months=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --months)
            [[ $# -ge 2 ]] || usage
            months="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done
[[ -n "$months" ]] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/s3-index.sh
. "$script_dir/lib/s3-index.sh"
# shellcheck source=lib/s3-publish-prefix.sh
. "$script_dir/lib/s3-publish-prefix.sh"
bucket="${TTLANG_S3_BUCKET:-tenstorrent-pypi}"

IFS=, read -r -a month_values <<< "$months"
for month in "${month_values[@]}"; do
    ttlang_s3_valid_year_month "$month" || {
        echo "Invalid year-month: $month" >&2
        exit 2
    }
    prefix="tt-lang/$month"
    anchors="$(s3_month_view_anchors "$bucket" "$month")"
    if [[ -n "$anchors" ]]; then
        s3_regenerate_month_view "$bucket" "$prefix"
    else
        aws s3api delete-object --bucket "$bucket" --key "$prefix" >/dev/null
        aws s3api delete-object --bucket "$bucket" --key "$prefix/" >/dev/null
    fi
done

s3_regenerate_index \
    --directories-only \
    --hidden-stable-wheels \
    "$bucket" \
    tt-lang

"${INJECT_S3_INDEX_README:-$script_dir/inject-s3-index-readme.sh}" \
    --bucket "$bucket" \
    --key tt-lang/ \
    --require-existing
