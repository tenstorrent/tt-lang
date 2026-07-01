#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

count_tt_chips() {
    local chip_count=0 entry
    for entry in /dev/tenstorrent/*; do
        entry="${entry##*/}"
        case "$entry" in
            '' | *[!0-9]*) ;;
            *) chip_count=$((chip_count + 1)) ;;
        esac
    done
    printf '%s\n' "$chip_count"
}

resolve_tt_chip_count() {
    local override="${1:-}" chips
    chips="${override:-$(count_tt_chips)}"
    case "$chips" in
        '' | *[!0-9]*)
            echo "chip count must be a non-negative integer, got '${chips}'" >&2
            return 2
            ;;
    esac
    printf '%s\n' "$chips"
}

absolute_path() {
    local path="${1:?path is required}"
    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$PWD" "$path" ;;
    esac
}
