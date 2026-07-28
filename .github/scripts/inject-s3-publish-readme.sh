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

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/s3-publish-prefix.sh
. "$script_dir/lib/s3-publish-prefix.sh"
if ! ttlang_s3_valid_publish_prefix "$prefix"; then
    echo "Invalid S3 publish prefix: $prefix" >&2
    exit 2
fi

workflow_root="$(cd "$script_dir/../.." && pwd)"
parent="$(dirname "$prefix")"
"$script_dir/inject-s3-index-readme.sh" \
    --key "$parent/" \
    --readme "$workflow_root/packaging/s3-index/README.md" \
    --require-existing \
    --dist-dir "$dist_dir"
