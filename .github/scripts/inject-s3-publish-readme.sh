#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -eu

[ "$#" -eq 2 ] || {
    echo "Usage: $0 <publish-prefix> <dist-dir>" >&2
    exit 2
}

prefix="$1"
dist_dir="$2"
case "$prefix" in
    tt-lang/releases | tt-lang/[0-9][0-9][0-9][0-9]-[0-9][0-9]) ;;
    *)
        echo "Invalid S3 publish prefix: $prefix" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
workflow_root="$(cd "$script_dir/../.." && pwd)"
parent="$(dirname "$prefix")"
"$script_dir/inject-s3-index-readme.sh" \
    --key "$parent/" \
    --readme "$workflow_root/packaging/s3-index/README.md" \
    --require-existing \
    --dist-dir "$dist_dir"
