#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run every test_*.sh under this directory. Returns 0 only if all tests pass.

set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
files=("$TESTS_DIR"/test_*.sh)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "No tests found in $TESTS_DIR" >&2
    exit 1
fi

overall_rc=0
for f in "${files[@]}"; do
    echo ""
    echo "=== Running $(basename "$f") ==="
    if ! "$f"; then
        overall_rc=1
    fi
done

echo ""
if [[ $overall_rc -eq 0 ]]; then
    echo "All test files passed."
else
    echo "One or more test files reported failures." >&2
fi
exit "$overall_rc"
