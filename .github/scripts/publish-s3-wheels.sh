#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Upload every wheel under <dist_dir> to the tenstorrent-pypi S3 PyPI index
# via s3pypi. With --overwrite, pass --force to allow replacing an existing
# wheel/version. With --prefix <p>, publish under the S3 key prefix <p> (a
# self-contained simple index at <p>/) instead of the flat root; omitting it
# preserves the flat-root layout.
#
# Usage: publish-s3-wheels.sh [--overwrite] [--prefix <prefix>] <dist_dir>

set -euo pipefail

usage() {
    echo "Usage: $0 [--overwrite] [--prefix <prefix>] <dist_dir>" >&2
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

if [[ $# -ne 1 ]]; then
    usage
fi

dist_dir="$1"

# --prefix maps to s3pypi's key prefix (self-contained simple index under
# <prefix>/); requires an s3pypi version that supports --prefix.
upload_args=(--put-root-index --bucket tenstorrent-pypi)
if [[ -n "$prefix" ]]; then
    upload_args+=(--prefix "$prefix")
fi
if [[ "$overwrite" -eq 1 ]]; then
    upload_args+=(--force)
fi

shopt -s nullglob
wheels=("$dist_dir"/*.whl)
if [[ "${#wheels[@]}" -eq 0 ]]; then
    echo "No wheels found under $dist_dir" >&2
    exit 1
fi

for wheel in "${wheels[@]}"; do
    s3pypi upload "$wheel" "${upload_args[@]}"
done
