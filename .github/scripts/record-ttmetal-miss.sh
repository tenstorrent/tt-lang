#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Record that no compatible tt-lang was found for a tt-metal SHA, so the nightly
# detector skips re-attempting it until tt-lang HEAD advances. Writes a small
# attempt.json marker under the SHA's 7-char S3 prefix.
#
# Usage:
#   record-ttmetal-miss.sh --ttmetal-sha <sha> --ttlang-head <sha>
#       [--max-age-days <n>] [--date <iso>]
#
# Env:
#   TTLANG_S3_BUCKET  S3 bucket (default tenstorrent-pypi).

set -euo pipefail

S3_BUCKET="${TTLANG_S3_BUCKET:-tenstorrent-pypi}"

usage() {
    echo "Usage: $0 --ttmetal-sha <sha> --ttlang-head <sha> [--max-age-days <n>] [--date <iso>]" >&2
    exit 2
}

ttmetal_sha=""
ttlang_head=""
max_age_days=""
attempt_date=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ttmetal-sha)  [[ $# -ge 2 ]] || usage; ttmetal_sha="$2"; shift 2 ;;
        --ttlang-head)  [[ $# -ge 2 ]] || usage; ttlang_head="$2"; shift 2 ;;
        --max-age-days) [[ $# -ge 2 ]] || usage; max_age_days="$2"; shift 2 ;;
        --date)         [[ $# -ge 2 ]] || usage; attempt_date="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "$ttmetal_sha" || -z "$ttlang_head" ]]; then
    usage
fi

short="${ttmetal_sha:0:7}"
attempt_date="${attempt_date:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

marker="$(printf '{"ttmetal_sha":"%s","ttlang_head":"%s","max_age_days":"%s","attempt_date":"%s","result":"no_compatible"}\n' \
    "$ttmetal_sha" "$ttlang_head" "$max_age_days" "$attempt_date")"

printf '%s' "$marker" | aws s3 cp - "s3://$S3_BUCKET/$short/attempt.json" \
    --content-type "application/json"

echo "Recorded miss for $short (tt-lang HEAD $ttlang_head)" >&2
