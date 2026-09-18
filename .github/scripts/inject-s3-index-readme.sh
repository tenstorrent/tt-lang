#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Restore the README into an S3 PyPI index. If allowed and the target index is
# missing, create one from the wheel dist only for that missing-key case.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: inject-s3-index-readme.sh --key <s3-key> [options]

Options:
  --bucket <bucket>      S3 bucket. Default: tenstorrent-pypi.
  --readme <path>        README markdown path. Default: packaging/s3-index/README.md.
  --dist-dir <path>      Wheel dist dir used to create a missing root index. Default: dist.
  --require-existing     Fail if the target index key does not already exist.
EOF
    exit 2
}

bucket=tenstorrent-pypi
readme=packaging/s3-index/README.md
dist_dir=dist
key=""
create_from_dist=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket)
            [[ $# -ge 2 ]] || usage
            bucket="$2"
            shift 2
            ;;
        --readme)
            [[ $# -ge 2 ]] || usage
            readme="$2"
            shift 2
            ;;
        --dist-dir)
            [[ $# -ge 2 ]] || usage
            dist_dir="$2"
            shift 2
            ;;
        --key)
            [[ $# -ge 2 ]] || usage
            key="$2"
            shift 2
            ;;
        --require-existing)
            create_from_dist=false
            shift
            ;;
        *)
            usage
            ;;
    esac
done

if [[ -z "$key" ]]; then
    usage
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
index_html="$tmpdir/index.html"
aws_error="$tmpdir/aws-get.err"

# The key may end in "/" (the directory listing is a slash-key); s3api uses the
# exact key, whereas `s3 cp` treats a trailing slash as a directory.
if ! aws s3api get-object --bucket "$bucket" --key "$key" "$index_html" >/dev/null 2>"$aws_error"; then
    if grep -Eq 'NoSuchKey|Not Found|404|does not exist' "$aws_error"; then
        if [[ "$create_from_dist" == false ]]; then
            echo "S3 index s3://$bucket/$key does not exist." >&2
            exit 1
        fi
        echo "S3 index s3://$bucket/$key does not exist; creating a root index from $dist_dir." >&2
        rm -f "$index_html"
    else
        cat "$aws_error" >&2
        exit 1
    fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/inject_s3_index_readme.py" \
    --create-from-dist "$dist_dir" \
    "$readme" \
    "$index_html"
aws s3api put-object --bucket "$bucket" --key "$key" --body "$index_html" \
    --content-type "text/html; charset=utf-8" >/dev/null
if [[ "$key" == */ ]]; then
    alias_key="${key%/}"
    if [[ -n "$alias_key" ]]; then
        aws s3api put-object --bucket "$bucket" --key "$alias_key" --body "$index_html" \
            --content-type "text/html; charset=utf-8" >/dev/null
    fi
fi
