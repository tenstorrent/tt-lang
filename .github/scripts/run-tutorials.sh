#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run all tutorial examples. Works both inside the dist container (where
# examples are at /root/examples/) and from a source checkout (pass "." as
# the examples root).
#
# Discovers *.py files under examples/elementwise-tutorial/ and
# examples/tutorial/ directories.
#
# Usage:
#   bash .github/scripts/run-tutorials.sh [examples-root]
#
# Optional argument: root directory containing the examples/ tree.
# Defaults to /root (matching the dist container layout).

set -uo pipefail

# Activate the tt-lang environment if not already active.
if [[ "${TTLANG_ENV_ACTIVATED:-0}" != "1" ]]; then
    ACTIVATE="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}/env/activate"
    if [[ -f "$ACTIVATE" ]]; then
        source "$ACTIVATE"
    fi
fi

ROOT="${1:-/root}"
EXAMPLES_DIR="${ROOT}/examples"

if [[ ! -d "$EXAMPLES_DIR" ]]; then
    echo "ERROR: Examples directory not found: $EXAMPLES_DIR" >&2
    exit 1
fi

# Collect tutorial scripts from both tutorial directories.
collect_tutorials() {
    local dir
    for dir in elementwise-tutorial tutorial; do
        if [[ -d "${EXAMPLES_DIR}/${dir}" ]]; then
            find "${EXAMPLES_DIR}/${dir}" -type f -name "*.py" -print0 \
                | sort -z \
                | tr '\0' '\n'
        fi
    done
}

mapfile -t SCRIPTS < <(collect_tutorials)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
    echo "ERROR: No tutorial scripts found under ${EXAMPLES_DIR}/{elementwise-tutorial,tutorial}/" >&2
    exit 1
fi

echo "=== Tutorial Tests ==="
echo "Examples root: ${EXAMPLES_DIR}"
echo "Found ${#SCRIPTS[@]} tutorial script(s)"
echo ""

declare -a RESULTS=()
N_PASS=0
N_FAIL=0

for script in "${SCRIPTS[@]}"; do
    label="${script#"${ROOT}/"}"
    echo "--- ${label} ---"
    rc=0
    python3 "$script" || rc=$?

    if [[ $rc -eq 0 ]]; then
        RESULTS+=("${label} ... PASS")
        (( N_PASS++ ))
    else
        RESULTS+=("${label} ... FAIL (rc=${rc})")
        (( N_FAIL++ ))
    fi
    echo ""
done

echo "========================================"
echo "  tutorial tests: results"
echo "========================================"
for r in "${RESULTS[@]}"; do
    echo "  ${r}"
done
echo "----------------------------------------"
printf "  PASS: %d  FAIL: %d  Total: %d\n" "${N_PASS}" "${N_FAIL}" "${#SCRIPTS[@]}"
echo "========================================"

if [[ ${N_FAIL} -gt 0 ]]; then
    exit 1
fi
