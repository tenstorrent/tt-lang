#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify every wheel in a directory has the expected version field.
#
# Usage: .github/scripts/verify-wheel-version.sh <expected_version> <wheel_dir>
# Usage: .github/scripts/verify-wheel-version.sh \
#          --expect <distribution=expected_version> [...] <wheel_dir>
#
# Wheel filenames follow PEP 427:
#   {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
# The first two '-'-separated components are the distribution and version.

set -euo pipefail

usage() {
    {
        echo "Usage: $0 <expected_version> <wheel_dir>"
        echo "Usage: $0 --expect <distribution=expected_version> [...] <wheel_dir>"
    } >&2
    exit 2
}

if [[ $# -lt 2 ]]; then
    usage
fi

declare -A expected_by_distribution=()
single_expected=""

if [[ "$1" == "--expect" ]]; then
    while [[ $# -gt 0 && "$1" == "--expect" ]]; do
        if [[ $# -lt 2 ]]; then
            usage
        fi
        spec="$2"
        if [[ "$spec" != *=* ]]; then
            usage
        fi
        distribution="${spec%%=*}"
        expected="${spec#*=}"
        if [[ -z "$distribution" || -z "$expected" ]]; then
            usage
        fi
        expected_by_distribution["$distribution"]="$expected"
        shift 2
    done
    if [[ $# -ne 1 ]]; then
        usage
    fi
else
    if [[ $# -ne 2 ]]; then
        usage
    fi
    single_expected="$1"
    shift
fi

wheel_dir="$1"

shopt -s nullglob
wheels=("$wheel_dir"/*.whl)
if [[ ${#wheels[@]} -eq 0 ]]; then
    echo "No wheels found in $wheel_dir" >&2
    exit 1
fi

failed=0
declare -A seen_distributions=()
for whl in "${wheels[@]}"; do
    distribution=$(basename "$whl" | awk -F- '{print $1}')
    ver=$(basename "$whl" | awk -F- '{print $2}')
    seen_distributions["$distribution"]=1
    if [[ -n "$single_expected" ]]; then
        expected="$single_expected"
    elif [[ -n "${expected_by_distribution[$distribution]:-}" ]]; then
        expected="${expected_by_distribution[$distribution]}"
    else
        echo "No expected version configured for distribution '$distribution' (file: $whl)" >&2
        failed=1
        continue
    fi
    if [[ "$ver" != "$expected" ]]; then
        echo "Wheel version '$ver' does not match expected '$expected' (file: $whl)" >&2
        failed=1
    fi
done

if [[ -z "$single_expected" ]]; then
    for distribution in "${!expected_by_distribution[@]}"; do
        if [[ -z "${seen_distributions[$distribution]:-}" ]]; then
            echo "No wheel found for expected distribution '$distribution'" >&2
            failed=1
        fi
    done
fi

exit "$failed"
