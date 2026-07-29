#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -eu

[ "$#" -eq 3 ] || {
    echo "Usage: $0 <version> <artifact-root> <publish-dir>" >&2
    exit 2
}

version="$1"
artifact_root="$2"
publish_dir="$3"
script_dir="$(cd "$(dirname "$0")" && pwd)"

set --
if [ -d "$artifact_root/bundled" ]; then
    set -- "$@" "bundled=$artifact_root/bundled"
fi
if [ -d "$artifact_root/light" ]; then
    set -- "$@" "light:no-sim=$artifact_root/light"
fi
if [ -d "$artifact_root/pypi" ]; then
    set -- "$@" "pypi=$artifact_root/pypi"
fi
if [ "$#" -eq 0 ]; then
    echo "No selected wheel artifacts found under $artifact_root" >&2
    exit 1
fi

"$script_dir/prepare-s3-publish-dist.sh" "$version" "$publish_dir" "$@"
