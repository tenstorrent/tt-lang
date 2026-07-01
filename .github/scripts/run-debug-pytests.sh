#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Repeatedly run a chosen subset of pytest tests to reproduce and debug
# intermittent or hardware-specific failures (e.g. an architecture-specific
# segfault). On a failing iteration, capture a native backtrace from a core dump
# if the runner produces one, or by running under gdb.
#
# Env:
#   PYTEST_ARGS   pytest selection, e.g. 'test/python/test_reduce.py -k "fp32"'.
#                 Parsed with shell-like quoting (via xargs); never eval'd.
#   REPEAT        max iterations (default 1).
#   STOP_ON_FAIL  stop at the first failing iteration (default true).
#   UNDER_GDB     run each iteration under gdb to capture a backtrace (default
#                 false; gdb's ptrace can perturb timing-sensitive races, so use
#                 it after a plain run demonstrates reproduction).
#   ARTIFACT_DIR  directory for logs, backtraces, and cores (default
#                 ./debug-artifacts).
#
# Exit status: non-zero if any iteration failed.

set -uo pipefail

: "${PYTEST_ARGS:?PYTEST_ARGS is required (the pytest selection to run)}"
REPEAT="${REPEAT:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-true}"
UNDER_GDB="${UNDER_GDB:-false}"
ARTIFACT_DIR="${ARTIFACT_DIR:-debug-artifacts}"

case "$REPEAT" in
    '' | *[!0-9]*)
        echo "run-debug-pytests.sh: REPEAT must be a positive integer, got '${REPEAT}'" >&2
        exit 2
        ;;
esac
if [ "$REPEAT" -lt 1 ]; then
    echo "run-debug-pytests.sh: REPEAT must be a positive integer, got '${REPEAT}'" >&2
    exit 2
fi
if [ "$UNDER_GDB" = "true" ] && ! command -v gdb >/dev/null 2>&1; then
    echo "run-debug-pytests.sh: UNDER_GDB=true requires gdb" >&2
    exit 2
fi

PYTEST_FLAGS=(-v --tb=long --timeout=300 --timeout-method=thread)

# Split PYTEST_ARGS into an argv array honoring quotes, without eval or python
# (so `-k "a and b"` stays one argument). xargs applies shell-like word/quote
# parsing but, unlike eval, never executes the input.
if ! parsed="$(printf '%s' "$PYTEST_ARGS" | xargs printf '%s\n')"; then
    echo "run-debug-pytests.sh: cannot parse PYTEST_ARGS (unbalanced quotes?)" >&2
    exit 2
fi
mapfile -t SELECTOR <<< "$parsed"

mkdir -p "$ARTIFACT_DIR"
ulimit -c unlimited 2>/dev/null || true
echo "core_pattern=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo '?')"
echo "selection: ${SELECTOR[*]}"

have_gdb=false
command -v gdb >/dev/null 2>&1 && have_gdb=true

failures=0
for ((iteration = 1; iteration <= REPEAT; iteration++)); do
    echo "::group::iteration ${iteration}/${REPEAT}"
    log="${ARTIFACT_DIR}/iter_${iteration}.log"
    if [ "$UNDER_GDB" = "true" ] && [ "$have_gdb" = "true" ]; then
        gdb -batch -nx --return-child-result -ex run -ex 'thread apply all bt' \
            --args python3 -m pytest "${SELECTOR[@]}" "${PYTEST_FLAGS[@]}" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
    else
        python3 -m pytest "${SELECTOR[@]}" "${PYTEST_FLAGS[@]}" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
    fi
    echo "iteration ${iteration} exit=${rc}"
    echo "::endgroup::"

    if [ "$rc" -ne 0 ]; then
        failures=$((failures + 1))
        if [ "$have_gdb" = "true" ]; then
            for core_file in core core.*; do
                [ -e "$core_file" ] || continue
                echo "capturing backtrace from ${core_file}"
                gdb -batch -nx -ex 'thread apply all bt' python3 "$core_file" \
                    > "${ARTIFACT_DIR}/bt_iter_${iteration}.txt" 2>&1 || true
                mv "$core_file" "$ARTIFACT_DIR/" 2>/dev/null || true
            done
        fi
        [ "$STOP_ON_FAIL" = "true" ] && break
    fi
done

echo "completed: ${failures} failing iteration(s) (repeat=${REPEAT})"
[ "$failures" -eq 0 ]
