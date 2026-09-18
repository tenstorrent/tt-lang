#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

ttlang_s3_valid_year_month() {
    case "$1" in
        [0-9][0-9][0-9][0-9]-0[1-9] | [0-9][0-9][0-9][0-9]-1[0-2])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

ttlang_s3_valid_publish_prefix() {
    if [ "$1" = "tt-lang/releases" ]; then
        return 0
    fi
    case "$1" in
        tt-lang/*)
            ttlang_s3_valid_year_month "${1#tt-lang/}"
            ;;
        *)
            return 1
            ;;
    esac
}
