#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Publish a per-tt-metal-SHA wheel set to a browsable slash-key directory under
# <prefix> (e.g. tt-lang/ttmetal/<sha7>), consumed with pip --find-links. Wheels
# and README.txt are uploaded as objects; the directory listing is written to the
# slash-key <prefix>/ and the parent listing is regenerated so the new entry
# shows up when browsing.
#
# Usage: publish-s3-direct-wheels.sh --prefix <prefix> [--readme <path>] <dist_dir>

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
# shellcheck source=lib/s3-index.sh
. "$script_dir/lib/s3-index.sh"

usage() {
    echo "Usage: $0 --prefix <prefix> [--readme <path>] <dist_dir>" >&2
    exit 2
}

bucket="${TTLANG_S3_BUCKET:-tenstorrent-pypi}"
prefix=""
readme="$repo_root/packaging/s3-index/README.md"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) [[ $# -ge 2 ]] || usage; prefix="${2%/}"; shift 2 ;;
        --readme) [[ $# -ge 2 ]] || usage; readme="$2"; shift 2 ;;
        -*) echo "Unknown option: $1" >&2; usage ;;
        *) break ;;
    esac
done

if [[ $# -ne 1 || -z "$prefix" ]]; then
    usage
fi

dist_dir="$1"
shopt -s nullglob
wheels=("$dist_dir"/*.whl)
if [[ "${#wheels[@]}" -eq 0 ]]; then
    echo "No wheels found under $dist_dir" >&2
    exit 1
fi

# Upload the README (as README.txt) and each wheel as plain objects.
aws s3 cp "$readme" "s3://$bucket/$prefix/README.txt" \
    --content-type "text/plain; charset=utf-8"
for wheel in "${wheels[@]}"; do
    aws s3 cp "$wheel" "s3://$bucket/$prefix/$(basename "$wheel")" \
        --content-type "application/octet-stream"
done

# Per-SHA index: hash the local wheels (no re-download) and write the slash-key.
# Wheels upload first, so if this write fails the directory is not browsable
# until re-run. Re-running this script with the local dist restores the hashed
# index; `s3-pypi-ops.sh --operation put-index` instead regenerates from S3
# (s3_child_anchors), which produces a valid but hash-less index.
index_html="$(mktemp)"; trap 'rm -f "$index_html"' EXIT
s3_local_wheel_anchors "$dist_dir" | s3_render_index "tt-lang: $prefix" > "$index_html"
s3_put_index "$bucket" "$prefix" "$index_html"

parent="$(dirname "$prefix")"
if [[ "$parent" != "." && "$parent" != "$prefix" ]]; then
    s3_regenerate_index "$bucket" "$parent"
fi
