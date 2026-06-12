#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build and optionally push manylinux_2_34 wheel-builder images for S3 light
# wheels.

set -eu

NO_PUSH=false
PYTHON_TAGS=cp310,cp312
DOCKER_TAG=""

usage() {
    cat >&2 <<'EOF'
Usage: build-wheel-manylinux-images.sh [--no-push] [--image-tag <tag>] [--python-tags cp310,cp312]
EOF
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-push)
            NO_PUSH=true
            shift
            ;;
        --python-tags)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            PYTHON_TAGS="$2"
            shift 2
            ;;
        --image-tag)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            DOCKER_TAG="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

repo=tenstorrent/tt-lang
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(git rev-parse --show-toplevel)"
docker_tag="${DOCKER_TAG:-$("$script_dir/get-version-tag.sh")}"
dockerfile="$script_dir/Dockerfile.wheel-manylinux-2-34"

# shellcheck source=/dev/null
. "$repo_root/third-party/tt-metal-version"
: "${TT_METAL_TAG:?third-party/tt-metal-version: TT_METAL_TAG not set}"
tt_metal_short_sha="$(git -C "$repo_root/third-party/tt-metal" rev-parse --short=10 HEAD)"

if [ -z "$PYTHON_TAGS" ]; then
    echo "At least one Python tag is required" >&2
    exit 2
fi

for python_tag in $(printf '%s\n' "$PYTHON_TAGS" | tr ',' ' '); do
    case "$python_tag" in
        cp310 | cp312) ;;
        *)
            echo "Unsupported Python tag: $python_tag" >&2
            exit 2
            ;;
    esac

    image_name="tt-lang-wheel-manylinux-2-34-${python_tag}"
    local_image="${image_name}:${docker_tag}"
    registry_image="ghcr.io/${repo}/${image_name}:${docker_tag}"

    if [ "$NO_PUSH" = true ]; then
        echo "Building local image: $local_image"
        ${DOCKER:-docker} build \
            --progress=plain \
            --build-arg "PYTHON_TAG=$python_tag" \
            --build-arg "TT_METAL_TAG=$TT_METAL_TAG" \
            --build-arg "TT_METAL_SHORT_SHA=$tt_metal_short_sha" \
            -t "$local_image" \
            -f "$dockerfile" \
            "$repo_root"
    else
        echo "Building registry image: $registry_image"
        ${DOCKER:-docker} build \
            --progress=plain \
            --build-arg "PYTHON_TAG=$python_tag" \
            --build-arg "TT_METAL_TAG=$TT_METAL_TAG" \
            --build-arg "TT_METAL_SHORT_SHA=$tt_metal_short_sha" \
            -t "$registry_image" \
            -t "$local_image" \
            -f "$dockerfile" \
            "$repo_root"
    fi

    if [ "$NO_PUSH" != true ]; then
        ${DOCKER:-docker} push "$registry_image"
        if [ "${GITHUB_REF:-}" = "refs/heads/main" ]; then
            latest="${registry_image%:*}:latest"
            ${DOCKER:-docker} tag "$registry_image" "$latest"
            ${DOCKER:-docker} push "$latest"
        fi
    fi
done
