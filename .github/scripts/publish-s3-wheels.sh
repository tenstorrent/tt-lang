#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Upload every wheel under <dist_dir> to the tenstorrent-pypi S3 PyPI index
# via s3pypi. With --overwrite, pass --force to allow replacing an existing
# wheel/version.
#
# Usage: publish-s3-wheels.sh [--overwrite] <dist_dir>

set -euo pipefail

overwrite=0
if [[ "${1:-}" == "--overwrite" ]]; then
    overwrite=1
    shift
fi

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 [--overwrite] <dist_dir>" >&2
    exit 2
fi

dist_dir="$1"

upload_args=(--put-root-index --bucket tenstorrent-pypi)
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
