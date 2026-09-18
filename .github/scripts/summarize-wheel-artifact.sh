#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -eu

dry_run=false
if [ "${1:-}" = --dry-run ]; then
    dry_run=true
    shift
fi
[ "$#" -eq 1 ] || {
    echo "Usage: $0 [--dry-run] <dist-dir>" >&2
    exit 2
}

dist_dir="$1"
if [ "$dry_run" = true ]; then
    echo "Dry run complete; no upload performed."
fi
echo "Wheel artifact:"
ls -lh "$dist_dir"
