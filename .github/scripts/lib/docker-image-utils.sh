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

ttlang_wheel_builder_image() {
    if [ "$#" -ne 2 ]; then
        echo "Usage: ttlang_wheel_builder_image <python-tag> <docker-tag>" >&2
        return 2
    fi
    ttlang_image_for_tag "tt-lang-wheel-manylinux-2-34-$1" "$2"
}

# Validate a comma-separated list of supported Python ABI tags and print them
# one per line. Used to iterate the light-wheel builder ABIs in one place.
ttlang_python_tags() {
    if [ "$#" -ne 1 ] || [ -z "$1" ]; then
        echo "At least one Python tag is required" >&2
        return 2
    fi
    ttlang_pt_count=0
    for ttlang_pt in $(printf '%s\n' "$1" | tr ',' ' '); do
        case "$ttlang_pt" in
            cp310 | cp312) ;;
            *)
                echo "Unsupported Python tag: $ttlang_pt" >&2
                return 2
                ;;
        esac
        ttlang_pt_count=$((ttlang_pt_count + 1))
        printf '%s\n' "$ttlang_pt"
    done
    if [ "$ttlang_pt_count" -eq 0 ]; then
        echo "At least one Python tag is required" >&2
        return 2
    fi
}
