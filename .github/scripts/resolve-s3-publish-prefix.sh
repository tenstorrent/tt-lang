#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

[[ $# -eq 1 ]] || {
    echo "Usage: $0 <version>" >&2
    exit 2
}

version="$1"
prefix=tt-lang/releases
if [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.dev([0-9]{4})([0-9]{2})[0-9]{2} ]]; then
    prefix="tt-lang/${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
fi

echo "prefix=$prefix" >> "${GITHUB_OUTPUT:-/dev/stdout}"
echo "Publish prefix: $prefix"
