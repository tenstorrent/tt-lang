#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run all tutorial examples. Works both inside the dist container (where
# examples are at /root/examples/) and from a source checkout (pass "." as
# the examples root).
#
# Discovers *.py files under examples/elementwise-tutorial/,
# examples/matmul-tutorial/, and examples/tutorial/ directories
# (excluding __init__.py, which exists only to mark these as packages
# inside the tt-lang wheel).
#
# Usage:
#   bash .github/scripts/run-tutorials.sh [examples-root]
#
# Optional argument: root directory containing the examples/ tree.
# Defaults to /root (matching the dist container layout).

set -uxo pipefail

# 300s per script accommodates n150 (single-chip, partially harvested) running
# the 8192x8192 matmul tutorials with cold JIT cache; multi-device hardware
# completes each in 20-30s. Override via TUTORIAL_TIMEOUT_SECONDS if needed.
TUTORIAL_TIMEOUT_SECONDS="${TUTORIAL_TIMEOUT_SECONDS:-300}"

# File header tag (within first 80 lines) marking scripts that require >1
# device. Skipped on single-device hardware (e.g. n150).
MULTI_DEVICE_TAG="TTLANG_TUTORIAL_CI: requires-multi-device"

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

# Query device count once. Tutorials tagged as requiring multi-device are
# skipped when this is < 2.
NUM_DEVICES=$(python3 -c 'import ttnn; print(ttnn.GetNumAvailableDevices())' 2>&1 \
              | grep -E '^[0-9]+$' | tail -n1)
NUM_DEVICES="${NUM_DEVICES:-0}"
echo "Available devices: ${NUM_DEVICES}"

file_has_tag() {
    head -n 80 "$1" | grep -Fq "# $2"
}

# Collect tutorial scripts from the three tutorial directories.
collect_tutorials() {
    local dir
    for dir in elementwise-tutorial matmul-tutorial tutorial; do
        if [[ -d "${EXAMPLES_DIR}/${dir}" ]]; then
            find "${EXAMPLES_DIR}/${dir}" -type f -name "*.py" \
                ! -name "__init__.py" -print0 \
                | sort -z \
                | tr '\0' '\n'
        fi
    done
}

mapfile -t SCRIPTS < <(collect_tutorials)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
    echo "ERROR: No tutorial scripts found under ${EXAMPLES_DIR}/{elementwise-tutorial,matmul-tutorial,tutorial}/" >&2
    exit 1
fi

echo "=== Tutorial Tests ==="
echo "Examples root: ${EXAMPLES_DIR}"
echo "Found ${#SCRIPTS[@]} tutorial script(s)"
echo ""

declare -a RESULTS=()
N_PASS=0
N_FAIL=0
N_SKIP=0

for script in "${SCRIPTS[@]}"; do
    label="${script#"${ROOT}/"}"

    if (( NUM_DEVICES < 2 )) && file_has_tag "${script}" "${MULTI_DEVICE_TAG}"; then
        RESULTS+=("${label} ... SKIP (requires multi-device, available=${NUM_DEVICES})")
        (( N_SKIP++ ))
        continue
    fi

    echo "--- ${label} ---"
    rc=0
    timeout --signal=TERM --kill-after=10 "${TUTORIAL_TIMEOUT_SECONDS}" python3 "$script" || rc=$?

    if [[ $rc -eq 0 ]]; then
        RESULTS+=("${label} ... PASS")
        (( N_PASS++ ))
    elif [[ $rc -eq 124 ]]; then
        RESULTS+=("${label} ... HANG (timed out after ${TUTORIAL_TIMEOUT_SECONDS}s)")
        (( N_FAIL++ ))
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
printf "  PASS: %d  FAIL: %d  SKIP: %d  Total: %d\n" "${N_PASS}" "${N_FAIL}" "${N_SKIP}" "${#SCRIPTS[@]}"
echo "========================================"

if [[ ${N_FAIL} -gt 0 ]]; then
    exit 1
fi
