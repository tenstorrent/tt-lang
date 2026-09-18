#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Reset the runner's cards and verify tt-smi answers afterwards.
#
# A degraded cluster reports as unrelated failures in every device test rather
# than as a device error, so fail here instead. Exabox Blackhole Galaxy workers
# do not provide the IPMI utility required by -glx_reset; the UMD warm reset
# supports these systems and the other CI devices.
#
# Env: TT_RESET_MAX_ATTEMPTS (10), TT_RESET_RETRY_SECONDS (30).
#
# Usage: reset-tt-cards.sh

set -uo pipefail

MAX_ATTEMPTS="${TT_RESET_MAX_ATTEMPTS:-10}"
RETRY_SECONDS="${TT_RESET_RETRY_SECONDS:-30}"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "=== Reset attempt $attempt of $MAX_ATTEMPTS ==="

    if ! tt-smi -r; then
        echo "reset failed on attempt $attempt"
        sleep "$RETRY_SECONDS"
        continue
    fi

    # Links and services settle after the reset returns.
    sleep 5

    if tt-smi --snapshot_no_tty --snapshot; then
        echo "reset and health check passed on attempt $attempt"
        exit 0
    fi

    echo "health check failed on attempt $attempt"
    sleep "$RETRY_SECONDS"
done

echo "::error::Card reset or health check failed after $MAX_ATTEMPTS attempts"
exit 1
