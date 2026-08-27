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

tt_device_group() {
    local group_size="${1:?group size is required}"
    local first_device="${2:-0}"
    local device_index device_list=""
    for ((device_index = first_device; device_index < first_device + group_size; device_index++)); do
        if [ -n "$device_list" ]; then
            device_list+=","
        fi
        device_list+="$device_index"
    done
    printf '%s\n' "$device_list"
}

resolve_tt_device_groups() {
    local chip_count="${1:?chip count is required}"
    local device_groups first_group group_size group_start
    for ((group_size = 1; group_size <= chip_count; group_size++)); do
        [ $((chip_count % group_size)) -eq 0 ] || continue
        first_group="$(tt_device_group "$group_size")"
        if ! TT_VISIBLE_DEVICES="$first_group" python3 -c \
            'import sys, ttnn; raise SystemExit(ttnn.get_num_devices() != int(sys.argv[1]))' \
            "$group_size" >/dev/null 2>&1; then
            continue
        fi

        device_groups=""
        for ((group_start = 0; group_start < chip_count; group_start += group_size)); do
            if [ -n "$device_groups" ]; then
                device_groups+=";"
            fi
            device_groups+="$(tt_device_group "$group_size" "$group_start")"
        done
        printf '%s\n' "$device_groups"
        return 0
    done
    return 1
}
