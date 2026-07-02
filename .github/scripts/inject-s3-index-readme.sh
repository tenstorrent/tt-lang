#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Restore the README into an S3 PyPI root index. If s3pypi omitted a prefixed
# root index, create one from the wheel dist only for that missing-key case.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: inject-s3-index-readme.sh --key <s3-key> [options]

Options:
  --bucket <bucket>      S3 bucket. Default: tenstorrent-pypi.
  --readme <path>        README markdown path. Default: packaging/s3-index/README.md.
  --dist-dir <path>      Wheel dist dir used to create a missing root index. Default: dist.
EOF
    exit 2
}

bucket=tenstorrent-pypi
readme=packaging/s3-index/README.md
dist_dir=dist
key=""

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
aws_error="$tmpdir/aws-cp.err"

if ! aws s3 cp "s3://$bucket/$key" "$index_html" 2>"$aws_error"; then
    if grep -Eq 'HeadObject.*404|404.*HeadObject|Key ".+" does not exist' "$aws_error"; then
        echo "S3 index s3://$bucket/$key does not exist; creating a root index from $dist_dir." >&2
        rm -f "$index_html"
    else
        cat "$aws_error" >&2
        exit 1
    fi
fi

python3 .github/scripts/inject_s3_index_readme.py \
    --create-from-dist "$dist_dir" \
    "$readme" \
    "$index_html"
aws s3 cp "$index_html" "s3://$bucket/$key" \
    --content-type "text/html; charset=utf-8"
