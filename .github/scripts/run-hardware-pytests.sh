#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run the hardware Python pytest suite, parallelizing single-device tests across
# every available chip and running multi_device (fabric mesh) tests serially.
#
# Chip count is the number of digit-named nodes under /dev/tenstorrent (matching
# test/lit.cfg.py). With more than one chip, single-device tests run under
# pytest-xdist (-n <chips>); test/python/conftest.py masks each worker to one
# chip via TT_VISIBLE_DEVICES so their device_id=0 opens hit distinct cards. The
# multi_device tests then run serially with every chip visible. With one chip the
# whole suite runs serially (multi_device tests skip themselves).
#
# Env: HW_PYTEST_CHIPS overrides the detected chip count.
#
# Usage: run-hardware-pytests.sh <test-dir> <report-prefix>

set -euo pipefail

TEST_DIR="${1:?usage: run-hardware-pytests.sh <test-dir> <report-prefix>}"
REPORT_PREFIX="${2:?usage: run-hardware-pytests.sh <test-dir> <report-prefix>}"

count_chips() {
    local n=0 entry
    for entry in /dev/tenstorrent/*; do
        entry="${entry##*/}"
        case "$entry" in
            '' | *[!0-9]*) ;; # the literal glob (no match) or non-numeric nodes
            *) n=$((n + 1)) ;;
        esac
    done
    printf '%s\n' "$n"
}

chips="${HW_PYTEST_CHIPS:-$(count_chips)}"

# thread method kills a C-level device deadlock that SIGALRM (signal) cannot.
common=(-v --tb=long --timeout=300 --timeout-method=thread)

if [ "$chips" -gt 1 ]; then
    echo "Detected ${chips} chips: single-device tests in parallel (-n ${chips}), multi_device serial"
    rc=0
    python3 -m pytest "$TEST_DIR" -m "not multi_device" -n "$chips" \
        "${common[@]}" --junitxml="${REPORT_PREFIX}-parallel.xml" || rc=1
    python3 -m pytest "$TEST_DIR" -m multi_device \
        "${common[@]}" --junitxml="${REPORT_PREFIX}-multidevice.xml" || rc=1
    exit "$rc"
fi

echo "Detected ${chips} chip(s): running the full suite serially"
python3 -m pytest "$TEST_DIR" "${common[@]}" --junitxml="${REPORT_PREFIX}.xml"
