#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Sourceable Docker image helpers for workflow and local scripts.

ttlang_docker() {
    ${DOCKER:-docker} "$@"
}

ttlang_image_for_tag() {
    if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
        echo "Usage: ttlang_image_for_tag <image-name> <tag> [registry]" >&2
        return 2
    fi

    ttlang_image_name="$1"
    ttlang_image_tag="$2"
    ttlang_image_registry="${3:-ghcr.io/tenstorrent/tt-lang}"
    ttlang_local_image="${ttlang_image_name}:${ttlang_image_tag}"

    if ttlang_docker image inspect "$ttlang_local_image" >/dev/null 2>&1; then
        printf '%s\n' "$ttlang_local_image"
    else
        printf '%s/%s\n' "$ttlang_image_registry" "$ttlang_local_image"
    fi
}
