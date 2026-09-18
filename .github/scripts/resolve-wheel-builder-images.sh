#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Resolve the content-derived manylinux builder tag and report whether all
# required ABI images already exist.

set -eu

repository="${GITHUB_REPOSITORY:-tenstorrent/tt-lang}"
docker_tag=""
workflow_source=""

usage() {
    echo "Usage: $0 [--repository <owner/repo>] [--docker-tag <tag>] [--workflow-source <git-checkout>]" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repository)
            [ "$#" -ge 2 ] || usage
            repository="$2"
            shift 2
            ;;
        --docker-tag)
            [ "$#" -ge 2 ] || usage
            docker_tag="$2"
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

case "$repository" in
    */*) ;;
    *) usage ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/docker-image-utils.sh
. "$script_dir/lib/docker-image-utils.sh"
if [ -z "$docker_tag" ]; then
    docker_tag="$("$script_dir/../containers/get-wheel-builder-version-tag.sh")"
    if [ -n "$workflow_source" ]; then
        workflow_sha="$(git -C "$workflow_source" rev-parse HEAD)"
        target_sha="$(git rev-parse HEAD)"
        if [ "$workflow_sha" != "$target_sha" ]; then
            docker_tag="${docker_tag}-wf$(printf '%s' "$workflow_sha" | cut -c1-8)"
        fi
    fi
fi
update_latest=false
if [ "${GITHUB_REF:-}" = refs/heads/main ]; then
    update_latest=true
    if [ -n "$workflow_source" ] &&
        [ "$(git -C "$workflow_source" rev-parse HEAD)" != "$(git rev-parse HEAD)" ]; then
        update_latest=false
    fi
fi
all_images_exist=true

for python_tag in cp310 cp312; do
    image="$(ttlang_wheel_builder_registry_image \
        "$python_tag" "$docker_tag" "ghcr.io/${repository}")"
    if ${DOCKER:-docker} manifest inspect "$image" >/dev/null 2>&1; then
        echo "Image exists: $image"
    else
        echo "Image missing: $image"
        all_images_exist=false
    fi
done

if [ -n "${GITHUB_OUTPUT:-}" ]; then
    {
        echo "docker-tag=$docker_tag"
        echo "all-images-exist=$all_images_exist"
        echo "update-latest=$update_latest"
    } >> "$GITHUB_OUTPUT"
else
    echo "docker-tag=$docker_tag"
    echo "all-images-exist=$all_images_exist"
    echo "update-latest=$update_latest"
fi

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo "### manylinux wheel-builder images"
        echo
        echo "- Docker tag: \`$docker_tag\`"
        echo "- All images exist: \`$all_images_exist\`"
    } >> "$GITHUB_STEP_SUMMARY"
fi
