#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run hardware Python lit tests in chip-pinned serial shards. Parallel shard
# count can be capped below the chip count to preserve host CPU capacity.
# Multi-device lit tests run afterward in a serial process.
#
# Env:
#   HW_LIT_CHIPS overrides the detected chip count.
#   HW_TEST_WORKERS caps parallel lit shards at no more than the chip count.
#   HW_LIT_MULTI_DEVICE_FILTER overrides the lit filter for multi-device tests.
#
# Usage: run-hardware-lit.sh <lit-test-dir> <report-prefix>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hardware-test-common.sh"

TEST_DIR="${1:?usage: run-hardware-lit.sh <lit-test-dir> <report-prefix>}"
REPORT_PREFIX="${2:?usage: run-hardware-lit.sh <lit-test-dir> <report-prefix>}"

chips="$(resolve_tt_chip_count "${HW_LIT_CHIPS:-}")" || {
    echo "run-hardware-lit.sh: invalid ${chips:-chip count}" >&2
    exit 2
}

workers="$chips"
if [ -n "${HW_TEST_WORKERS:-}" ]; then
    workers="$HW_TEST_WORKERS"
    case "$workers" in
        0 | *[!0-9]*)
            echo "run-hardware-lit.sh: worker count must be a positive integer, got '$workers'" >&2
            exit 2
            ;;
    esac
    [ "$workers" -le "$chips" ] || {
        echo "run-hardware-lit.sh: worker count $workers exceeds chip count $chips" >&2
        exit 2
    }
fi

lit_common=(-j1 --verbose)
multi_device_filter="${HW_LIT_MULTI_DEVICE_FILTER:-mesh_tensor}"
cache_root="$(absolute_path "${TT_METAL_CACHE:-${REPORT_PREFIX}-tt-metal-cache}")"
lit_exec_root="$(absolute_path "${REPORT_PREFIX}-lit-exec")"

if [ "$chips" -le 1 ]; then
    echo "Detected ${chips} chip(s): running Python lit serially"
    TTLANG_LIT_TEST_EXEC_ROOT="${lit_exec_root}/serial" \
        llvm-lit "$TEST_DIR" "${lit_common[@]}" \
        --xunit-xml-output="${REPORT_PREFIX}.xml"
    exit $?
fi

echo "Detected ${chips} chips: ${workers} Python lit shards in parallel, multi-device serial"

rc=0
pids=()
for ((chip_index = 0; chip_index < workers; chip_index++)); do
    shard_number=$((chip_index + 1))
    (
        export TT_VISIBLE_DEVICES="${chip_index}"
        export TT_METAL_CACHE="${cache_root}/shard-${shard_number}"
        export TTLANG_LIT_TEST_EXEC_ROOT="${lit_exec_root}/shard-${shard_number}"
        mkdir -p "$TT_METAL_CACHE"
        llvm-lit "$TEST_DIR" \
            --num-shards "$workers" \
            --run-shard "$shard_number" \
            --filter-out "$multi_device_filter" \
            --allow-empty-runs \
            "${lit_common[@]}" \
            --xunit-xml-output="${REPORT_PREFIX}-shard-${shard_number}.xml"
    ) &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done

multi_device_cache="${cache_root}/multidevice"
mkdir -p "$multi_device_cache"
env -u TT_VISIBLE_DEVICES \
    TT_METAL_CACHE="$multi_device_cache" \
    TTLANG_LIT_TEST_EXEC_ROOT="${lit_exec_root}/multidevice" \
    llvm-lit "$TEST_DIR" \
    --filter "$multi_device_filter" \
    --allow-empty-runs \
    "${lit_common[@]}" \
    --xunit-xml-output="${REPORT_PREFIX}-multidevice.xml" || rc=1

exit "$rc"
