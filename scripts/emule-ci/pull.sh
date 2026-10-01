#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
umask 077
image="${1:?Usage: pull.sh APPROVED_IMAGE_DIGEST NEW_OUTPUT_DIRECTORY}"
output="${2:?Usage: pull.sh APPROVED_IMAGE_DIGEST NEW_OUTPUT_DIRECTORY}"
if [[ ! "$image" =~ ^ghcr\.io/tenstorrent/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$ ]]; then
    echo 'Expected an approved company GHCR image pinned by sha256 digest.' >&2
    exit 2
fi
test ! -e "$output"
mkdir -p "$output/public" "$output/private"
started=$SECONDS
result=0
docker pull --platform linux/amd64 "$image" > "$output/private/pull.log" 2>&1 &
pull_pid=$!
terminate_pull() {
    kill -TERM "$pull_pid" 2>/dev/null || return 0
    for attempt in {1..20}; do
        kill -0 "$pull_pid" 2>/dev/null || return 0
        sleep 0.1
    done
    kill -KILL "$pull_pid" 2>/dev/null || true
}
trap terminate_pull EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
while kill -0 "$pull_pid" 2>/dev/null; do
    available_kib="$(df -Pk "$output" | awk 'NR == 2 { print $4 }')"
    df -Pk "$output" | awk -v elapsed="$((SECONDS - started))" \
        'NR == 2 { print elapsed, $3, $4 }' >> "$output/public/disk-samples.txt"
    if [ "$((SECONDS - started))" -ge 1200 ]; then
        result=124
        break
    fi
    if [ "$available_kib" -lt 2097152 ]; then
        result=95
        break
    fi
    sleep 2
done
if [ "$result" -ne 0 ]; then
    terminate_pull
fi
wait "$pull_pid" || { status=$?; [ "$result" -ne 0 ] || result=$status; }
trap - INT TERM EXIT
df -Pk "$output" | awk -v elapsed="$((SECONDS - started))" \
    'NR == 2 { print elapsed, $3, $4 }' >> "$output/public/disk-samples.txt"
printf '%s\n' "$((SECONDS - started))" > "$output/public/pull-seconds.txt"
printf '%s\n' "$result" > "$output/public/pull-exit-code.txt"
exit "$result"
