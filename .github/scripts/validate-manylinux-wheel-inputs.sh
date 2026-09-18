#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -eu

[ "$#" -eq 2 ] || {
    echo "Usage: $0 <pypi|external> <version>" >&2
    exit 2
}

ttnn_dep_mode="$1"
version="$2"
[ -n "$version" ] || {
    echo "version_override is required" >&2
    exit 2
}

case "$ttnn_dep_mode" in
    pypi | external) ;;
    *)
        echo "ttnn_dep_mode must be pypi or external" >&2
        exit 2
        ;;
esac
