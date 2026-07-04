#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Upload wheel files directly under an S3 prefix and write an index.html file
# listing in the same directory. This layout is for per-tt-metal-SHA wheel sets
# that are consumed with pip --find-links, not as a PEP 503 package index.
#
# Usage: publish-s3-direct-wheels.sh --prefix <prefix> [--readme <path>] <dist_dir>

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

usage() {
    echo "Usage: $0 --prefix <prefix> [--readme <path>] <dist_dir>" >&2
    exit 2
}

bucket="${TTLANG_S3_BUCKET:-tenstorrent-pypi}"
prefix=""
readme="$repo_root/packaging/s3-index/README.md"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || usage
            prefix="${2%/}"
            shift 2
            ;;
        --readme)
            [[ $# -ge 2 ]] || usage
            readme="$2"
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

dist_dir="$1"
shopt -s nullglob
wheels=("$dist_dir"/*.whl)
if [[ "${#wheels[@]}" -eq 0 ]]; then
    echo "No wheels found under $dist_dir" >&2
    exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
index_html="$tmpdir/index.html"
readme_txt="$tmpdir/README.txt"

python3 - "$dist_dir" "$index_html" <<'PY'
import hashlib
import html
import sys
from pathlib import Path
from urllib.parse import quote

dist_dir = Path(sys.argv[1])
index_path = Path(sys.argv[2])
wheels = sorted(dist_dir.glob("*.whl"))

anchors = ['    <a href="README.txt">README.txt</a><br>']
for wheel in wheels:
    hasher = hashlib.sha256()
    with wheel.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    name = wheel.name
    href = f"{quote(name)}#sha256={digest}"
    anchors.append(f'    <a href="{href}">{html.escape(name)}</a><br>')

index_path.write_text(
    "<!DOCTYPE html>\n"
    "<html>\n"
    "  <head>\n"
    '    <meta charset="UTF-8">\n'
    "    <title>tt-lang per-SHA wheels</title>\n"
    "  </head>\n"
    "  <body>\n"
    + "\n".join(anchors)
    + "\n  </body>\n"
    "</html>\n",
    encoding="utf-8",
)
PY

python3 "$script_dir/inject_s3_index_readme.py" "$readme" "$index_html"
cp "$readme" "$readme_txt"

aws s3 cp "$readme_txt" "s3://$bucket/$prefix/README.txt" \
    --content-type "text/plain; charset=utf-8"

for wheel in "${wheels[@]}"; do
    aws s3 cp "$wheel" "s3://$bucket/$prefix/$(basename "$wheel")" \
        --content-type "application/octet-stream"
done

aws s3 cp "$index_html" "s3://$bucket/$prefix/index.html" \
    --content-type "text/html; charset=utf-8"
