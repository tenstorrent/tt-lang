#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build and optionally push the manylinux_2_34 wheel-builder images shared by
# public PyPI and S3 wheel workflows.

set -eu

NO_PUSH=false
PYTHON_TAGS=cp310,cp312
DOCKER_TAG=""
BUILD_PARALLEL_LEVEL=""
LLVM_CACHE_REF=""
TTMETAL_CACHE_REF=""
WORKFLOW_SOURCE="."

usage() {
    cat >&2 <<'EOF'
Usage: build-wheel-manylinux-images.sh [--no-push] [--image-tag <tag>] [--python-tags cp310,cp312] [--build-parallel-level <jobs>] [--llvm-cache-ref <ref> --ttmetal-cache-ref <ref>] [--workflow-source <dir>]
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
        --build-parallel-level)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            BUILD_PARALLEL_LEVEL="$2"
            shift 2
            ;;
        --llvm-cache-ref)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            LLVM_CACHE_REF="$2"
            shift 2
            ;;
        --ttmetal-cache-ref)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            TTMETAL_CACHE_REF="$2"
            shift 2
            ;;
        --workflow-source)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            WORKFLOW_SOURCE="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

repo="${GITHUB_REPOSITORY:-tenstorrent/tt-lang}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(git rev-parse --show-toplevel)"
docker_tag="${DOCKER_TAG:-$("$script_dir/get-version-tag.sh")}"
dockerfile="$script_dir/Dockerfile.wheel-manylinux-2-34"
case "$WORKFLOW_SOURCE" in
    /* | *../* | */.. | ..)
        echo "--workflow-source must stay within the build context" >&2
        exit 2
        ;;
esac
[ -f "$repo_root/$WORKFLOW_SOURCE/.github/containers/CMakeLists.wheel-toolchain" ] || {
    echo "Workflow source not found: $WORKFLOW_SOURCE" >&2
    exit 2
}
# shellcheck source=../scripts/lib/docker-image-utils.sh
. "$script_dir/../scripts/lib/docker-image-utils.sh"

# shellcheck source=/dev/null
. "$repo_root/third-party/tt-metal-version"
: "${TT_METAL_TAG:?third-party/tt-metal-version: TT_METAL_TAG not set}"
tt_metal_short_sha="$(git -C "$repo_root/third-party/tt-metal" rev-parse --short=10 HEAD)"

if [ -z "$PYTHON_TAGS" ]; then
    echo "At least one Python tag is required" >&2
    exit 2
fi

case "$BUILD_PARALLEL_LEVEL" in
    "" | *[!0-9]* | 0)
        if [ -n "$BUILD_PARALLEL_LEVEL" ]; then
            echo "Build parallel level must be a positive integer: $BUILD_PARALLEL_LEVEL" >&2
            exit 2
        fi
        ;;
esac

if [ -n "$LLVM_CACHE_REF" ] || [ -n "$TTMETAL_CACHE_REF" ]; then
    if [ -z "$LLVM_CACHE_REF" ] || [ -z "$TTMETAL_CACHE_REF" ]; then
        echo "--llvm-cache-ref and --ttmetal-cache-ref must be provided together" >&2
        exit 2
    fi
fi

for python_tag in $(printf '%s\n' "$PYTHON_TAGS" | tr ',' ' '); do
    image_name="$(ttlang_wheel_builder_image_name "$python_tag")"
    local_image="${image_name}:${docker_tag}"
    registry_image="$(ttlang_wheel_builder_registry_image \
        "$python_tag" "$docker_tag" "ghcr.io/${repo}")"

    if [ "$NO_PUSH" != true ] && ${DOCKER:-docker} manifest inspect "$registry_image" >/dev/null 2>&1; then
        echo "Image already exists, skipping build: $registry_image"
        continue
    fi

    set -- \
        --progress=plain \
        --target wheel-builder \
        --build-arg "WORKFLOW_SOURCE=$WORKFLOW_SOURCE" \
        --build-arg "PYTHON_TAG=$python_tag" \
        --build-arg "TT_METAL_TAG=$TT_METAL_TAG" \
        --build-arg "TT_METAL_SHORT_SHA=$tt_metal_short_sha"
    if [ -n "$BUILD_PARALLEL_LEVEL" ]; then
        set -- "$@" --build-arg "TTLANG_BUILD_PARALLEL_LEVEL=$BUILD_PARALLEL_LEVEL"
    fi

    if [ -n "$LLVM_CACHE_REF" ]; then
        set -- "$@" \
            --cache-from "type=registry,ref=$LLVM_CACHE_REF" \
            --cache-from "type=registry,ref=$TTMETAL_CACHE_REF"
        if [ "$NO_PUSH" = true ]; then
            echo "Building local image from component caches: $local_image"
            ${DOCKER:-docker} buildx build "$@" --load -t "$local_image" -f "$dockerfile" "$repo_root"
        else
            echo "Building registry image from component caches: $registry_image"
            set -- "$@" --push -t "$registry_image"
            ${DOCKER:-docker} buildx build "$@" -f "$dockerfile" "$repo_root"
        fi
        continue
    fi

    if [ "$NO_PUSH" = true ]; then
        echo "Building local image: $local_image"
        ${DOCKER:-docker} build "$@" -t "$local_image" -f "$dockerfile" "$repo_root"
    else
        echo "Building registry image: $registry_image"
        ${DOCKER:-docker} build "$@" -t "$registry_image" -t "$local_image" -f "$dockerfile" "$repo_root"
    fi

    if [ "$NO_PUSH" != true ]; then
        ${DOCKER:-docker} push "$registry_image"
    fi
done
