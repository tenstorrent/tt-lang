#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run the hardware Python pytest suite, parallelizing single-device tests across
# available chips and running compile_only and multi_device tests serially.
#
# Chip count is the number of digit-named nodes under /dev/tenstorrent (matching
# test/lit.cfg.py). With more than one chip, single-device tests run under
# pytest-xdist with each worker restricted to one chip through
# TT_VISIBLE_DEVICES. Compile-only tests then run without xdist so concurrent
# compiler subprocesses do not contend for host resources. The multi_device
# tests run serially after that. With one chip the whole suite runs serially.
#
# Env: HW_PYTEST_CHIPS overrides the detected chip count.
# Env: HW_TEST_WORKERS caps xdist concurrency at no more than the chip count.
# Env: HW_PYTEST_TIMEOUT overrides the per-test timeout in seconds (default 300).
# Env: HW_SERIAL_TEST_VISIBLE_DEVICES restricts device visibility for serial
#      phases that need multiple devices. Unset means every chip remains visible.
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

timeout_seconds="${HW_PYTEST_TIMEOUT:-300}"
case "$timeout_seconds" in
    0 | *[!0-9]*)
        echo "run-hardware-pytests.sh: timeout must be a positive integer, got '$timeout_seconds'" >&2
        exit 2
        ;;
esac

workers="$chips"
if [ -n "${HW_TEST_WORKERS:-}" ]; then
    workers="$HW_TEST_WORKERS"
    case "$workers" in
        0 | *[!0-9]*)
            echo "run-hardware-pytests.sh: worker count must be a positive integer, got '$workers'" >&2
            exit 2
            ;;
    esac
    [ "$workers" -le "$chips" ] || {
        echo "run-hardware-pytests.sh: worker count $workers exceeds chip count $chips" >&2
        exit 2
    }
fi

# The thread timeout method terminates a process stuck in a C-level device call;
# SIGALRM cannot interrupt those calls.
# --reruns retries a flaky test up to 3 times (pytest-rerunfailures) before
# reporting failure. The shared pytest plugin makes abnormal xdist worker
# termination fail the session because pytest-rerunfailures otherwise hides it.
# A persistent timeout can consume four timeout intervals across all attempts.
common=(-v --tb=long --timeout="$timeout_seconds" --timeout-method=thread --reruns 3)
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

run_multi_device_phase() {
    if [ -n "${HW_SERIAL_TEST_VISIBLE_DEVICES:-}" ]; then
        export TT_VISIBLE_DEVICES="$HW_SERIAL_TEST_VISIBLE_DEVICES"
        echo "Restricting serial multi_device pytest phase to TT_VISIBLE_DEVICES=$TT_VISIBLE_DEVICES"
    else
        unset TT_VISIBLE_DEVICES
    fi
    run_pytest_phase "$@"
}

if [ "$chips" -gt 1 ]; then
    echo "Detected ${chips} chips: single-device tests in parallel (-n ${workers}), compile_only and multi_device serial"
    unset TT_VISIBLE_DEVICES
    cache_root="$(absolute_path "${TT_METAL_CACHE:-${REPORT_PREFIX}-tt-metal-cache}")"
    rc=0
    # Abnormal worker termination must fail the session. A replacement can hide
    # the original crash and invalidate the crash guard's completeness check.
    TTLANG_PIN_XDIST_WORKERS_TO_DEVICES=1 \
        TTLANG_XDIST_TT_METAL_CACHE_ROOT="$cache_root" \
        run_pytest_phase "$TEST_DIR" -m "not multi_device and not compile_only" -n "$workers" \
        --max-worker-restart=0 "${common[@]}" \
        --junitxml="${REPORT_PREFIX}-parallel.xml" || rc=1
    run_pytest_phase "$TEST_DIR" -m compile_only \
        "${common[@]}" --junitxml="${REPORT_PREFIX}-compile-only.xml" || rc=1
    run_multi_device_phase "$TEST_DIR" -m multi_device \
        "${common[@]}" --junitxml="${REPORT_PREFIX}-multidevice.xml" || rc=1
    if [ "$selected_phase_count" -eq 0 ]; then
        echo "No tests selected by any hardware pytest phase" >&2
        exit 5
    fi
    exit "$rc"
fi

echo "Detected ${chips} chip(s): running the full suite serially"
python3 -m pytest "$TEST_DIR" "${common[@]}" --junitxml="${REPORT_PREFIX}.xml"
