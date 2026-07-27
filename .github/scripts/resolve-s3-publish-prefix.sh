#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

[[ $# -eq 1 ]] || {
    echo "Usage: $0 <version>" >&2
    exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/s3-publish-prefix.sh
. "$script_dir/lib/s3-publish-prefix.sh"

version="$1"
public_version="${version%%+*}"
prefix=tt-lang/releases
if [[ "$public_version" =~ \.dev([0-9]+)$ ]]; then
    dev_number="${BASH_REMATCH[1]}"
    if [[ ! "$dev_number" =~ ^([0-9]{4})([0-9]{2})[0-9]{2}$ ]]; then
        echo "S3 dev versions must use an 8-digit YYYYMMDD dev number: $version" >&2
        exit 2
    fi
    prefix="tt-lang/${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
    if ! ttlang_s3_valid_publish_prefix "$prefix"; then
        echo "Invalid calendar month in dev version: $version" >&2
        exit 2
    fi
elif [[ "$public_version" == *".dev"* ]]; then
    echo "Invalid dev version: $version" >&2
    exit 2
fi

echo "prefix=$prefix" >> "${GITHUB_OUTPUT:-/dev/stdout}"
echo "Publish prefix: $prefix"
