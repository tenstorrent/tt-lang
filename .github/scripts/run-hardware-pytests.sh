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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hardware-test-common.sh"

TEST_DIR="${1:?usage: run-hardware-pytests.sh <test-dir> <report-prefix>}"
REPORT_PREFIX="${2:?usage: run-hardware-pytests.sh <test-dir> <report-prefix>}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

chips="$(resolve_tt_chip_count "${HW_PYTEST_CHIPS:-}")" || {
    echo "run-hardware-pytests.sh: invalid ${chips:-chip count}" >&2
    exit 2
}

# The thread timeout method interrupts C-level device deadlocks; SIGALRM cannot.
# --reruns retries a flaky test up to 3 times (pytest-rerunfailures) before
# reporting failure. A worker segfault is a crash, not a rerunnable failure, so
# it is unaffected.
common=(-v --tb=long --timeout=300 --timeout-method=thread --reruns 3)
pytest_config="$(absolute_path "$(dirname "$REPORT_PREFIX")/pytest.ini")"
if [ -f "$pytest_config" ]; then
    common=(-c "$pytest_config" --rootdir="${REPO_ROOT}/test" "${common[@]}")
fi

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
    unset TT_VISIBLE_DEVICES
    cache_root="$(absolute_path "${TT_METAL_CACHE:-${REPORT_PREFIX}-tt-metal-cache}")"
    rc=0
    # Workers are pinned 1:1 to chips by index (see pin_xdist_worker_to_device).
    # Disable xdist's crash-restart: a replacement worker gets the next index
    # (e.g. gw8 on an 8-chip host), which maps to a nonexistent device and turns
    # one crash into a flood of "Invalid device ID" errors.
    TTLANG_PIN_XDIST_WORKERS_TO_DEVICES=1 \
        TTLANG_XDIST_TT_METAL_CACHE_ROOT="$cache_root" \
        run_pytest_phase "$TEST_DIR" -m "not multi_device" -n "$chips" \
        --max-worker-restart=0 "${common[@]}" \
        --junitxml="${REPORT_PREFIX}-parallel.xml" || rc=1
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
