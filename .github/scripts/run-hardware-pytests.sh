#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run the hardware Python pytest suite, parallelizing single-device tests across
# every available chip and running multi_device (fabric mesh) tests serially.
#
# Chip count is the number of digit-named nodes under /dev/tenstorrent (matching
# test/lit.cfg.py). With more than one chip, single-device tests run under
# pytest-xdist (-n <chips>) with each worker restricted to one chip through
# TT_VISIBLE_DEVICES. The multi_device tests then run serially with every chip
# visible. With one chip the whole suite runs serially.
#
# Env: HW_PYTEST_CHIPS overrides the detected chip count.
#
# Usage: run-hardware-pytests.sh <test-dir> <report-prefix>

set -euo pipefail

TEST_DIR="${1:?usage: run-hardware-pytests.sh <test-dir> <report-prefix>}"
REPORT_PREFIX="${2:?usage: run-hardware-pytests.sh <test-dir> <report-prefix>}"

count_chips() {
    local chip_count=0 entry
    for entry in /dev/tenstorrent/*; do
        entry="${entry##*/}"
        case "$entry" in
            '' | *[!0-9]*) ;; # the literal glob (no match) or non-numeric nodes
            *) chip_count=$((chip_count + 1)) ;;
        esac
    done
    printf '%s\n' "$chip_count"
}

chips="${HW_PYTEST_CHIPS:-$(count_chips)}"

case "$chips" in
    '' | *[!0-9]*)
        echo "run-hardware-pytests.sh: chip count must be a non-negative integer, got '${chips}'" >&2
        exit 2
        ;;
esac

# The thread timeout method interrupts C-level device deadlocks; SIGALRM cannot.
common=(-v --tb=long --timeout=300 --timeout-method=thread)

selected_phase_count=0
run_pytest_phase() {
    local phase_rc=0
    python3 -m pytest "$@" || phase_rc=$?
    if [ "$phase_rc" -eq 5 ]; then
        echo "No tests selected for phase: pytest $*"
        return 0
    fi
    selected_phase_count=$((selected_phase_count + 1))
    return "$phase_rc"
}

if [ "$chips" -gt 1 ]; then
    echo "Detected ${chips} chips: single-device tests in parallel (-n ${chips}), multi_device serial"
    rc=0
    TTLANG_PIN_XDIST_WORKERS_TO_DEVICES=1 run_pytest_phase "$TEST_DIR" -m "not multi_device" -n "$chips" \
        "${common[@]}" --junitxml="${REPORT_PREFIX}-parallel.xml" || rc=1
    run_pytest_phase "$TEST_DIR" -m multi_device \
        "${common[@]}" --junitxml="${REPORT_PREFIX}-multidevice.xml" || rc=1
    if [ "$selected_phase_count" -eq 0 ]; then
        echo "No tests selected by either hardware pytest phase" >&2
        exit 5
    fi
    exit "$rc"
fi

echo "Detected ${chips} chip(s): running the full suite serially"
python3 -m pytest "$TEST_DIR" "${common[@]}" --junitxml="${REPORT_PREFIX}.xml"
