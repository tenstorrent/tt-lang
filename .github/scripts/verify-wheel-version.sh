#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify every wheel in a directory has the expected version field.
#
# Usage: .github/scripts/verify-wheel-version.sh <expected_version> <wheel_dir>
#
# Wheel filenames follow PEP 427:
#   {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
# The second '-'-separated component is the version.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <expected_version> <wheel_dir>" >&2
    exit 2
fi

expected="$1"
wheel_dir="$2"

shopt -s nullglob
wheels=("$wheel_dir"/*.whl)
if [[ ${#wheels[@]} -eq 0 ]]; then
    echo "No wheels found in $wheel_dir" >&2
    exit 1
fi

failed=0
for whl in "${wheels[@]}"; do
    ver=$(basename "$whl" | awk -F- '{print $2}')
    if [[ "$ver" != "$expected" ]]; then
        echo "Wheel version '$ver' does not match expected '$expected' (file: $whl)" >&2
        failed=1
    fi
done

exit "$failed"
