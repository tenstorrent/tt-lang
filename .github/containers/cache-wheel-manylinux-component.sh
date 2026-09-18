#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Build one manylinux wheel-toolchain component and publish it to GHCR with
# inline cache metadata. The LLVM and tt-metal references are intentionally
# distinct.

set -eu

component=""
python_tag=""
cache_ref=""
build_parallel_level=""
workflow_source="."

usage() {
    cat >&2 <<'EOF'
Usage: cache-wheel-manylinux-component.sh --component llvm|ttmetal --cache-ref <registry-ref> [options]

Options:
  --python-tag cp310|cp312       Required for LLVM; tt-metal always uses cp312.
  --build-parallel-level <jobs>  CMake build parallelism.
  --workflow-source <dir>        Workflow implementation within the build context.
EOF
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --component)
            [ "$#" -ge 2 ] || usage
            component="$2"
            shift 2
            ;;
        --python-tag)
            [ "$#" -ge 2 ] || usage
            python_tag="$2"
            shift 2
            ;;
        --cache-ref)
            [ "$#" -ge 2 ] || usage
            cache_ref="$2"
            shift 2
            ;;
        --build-parallel-level)
            [ "$#" -ge 2 ] || usage
            build_parallel_level="$2"
            shift 2
            ;;
        --workflow-source)
            [ "$#" -ge 2 ] || usage
            workflow_source="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

case "$component" in
    llvm)
        case "$python_tag" in
            cp310 | cp312) ;;
            *) echo "LLVM requires --python-tag cp310 or cp312" >&2; exit 2 ;;
        esac
        target=llvm-toolchain
        ;;
    ttmetal)
        if [ -n "$python_tag" ]; then
            echo "--python-tag is not valid for ttmetal; it uses cp312" >&2
            exit 2
        fi
        target=ttmetal-toolchain
        ;;
    *)
        usage
        ;;
esac

case "$build_parallel_level" in
    "" | *[!0-9]* | 0)
        if [ -n "$build_parallel_level" ]; then
            echo "Build parallel level must be a positive integer: $build_parallel_level" >&2
            exit 2
        fi
        ;;
esac

[ -n "$cache_ref" ] || usage

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(git rev-parse --show-toplevel)"
dockerfile="$script_dir/Dockerfile.wheel-manylinux-2-34"
case "$workflow_source" in
    /* | *../* | */.. | ..)
        echo "--workflow-source must stay within the build context" >&2
        exit 2
        ;;
esac
[ -f "$repo_root/$workflow_source/.github/containers/CMakeLists.wheel-toolchain" ] || {
    echo "Workflow source not found: $workflow_source" >&2
    exit 2
}

set -- \
    buildx build \
    --progress=plain \
    --target "$target" \
    --build-arg "WORKFLOW_SOURCE=$workflow_source" \
    --cache-from "type=registry,ref=$cache_ref" \
    --cache-to type=inline \
    --push \
    -t "$cache_ref"

if [ -n "$python_tag" ]; then
    set -- "$@" --build-arg "PYTHON_TAG=$python_tag"
else
    # shellcheck source=/dev/null
    . "$repo_root/third-party/tt-metal-version"
    : "${TT_METAL_TAG:?third-party/tt-metal-version: TT_METAL_TAG not set}"
    tt_metal_short_sha="$(git -C "$repo_root/third-party/tt-metal" rev-parse --short=10 HEAD)"
    set -- "$@" \
        --build-arg "TT_METAL_TAG=$TT_METAL_TAG" \
        --build-arg "TT_METAL_SHORT_SHA=$tt_metal_short_sha"
fi
if [ -n "$build_parallel_level" ]; then
    set -- "$@" --build-arg "TTLANG_BUILD_PARALLEL_LEVEL=$build_parallel_level"
fi

echo "Publishing $component component image: $cache_ref"
${DOCKER:-docker} "$@" -f "$dockerfile" "$repo_root"
