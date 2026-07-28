#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Publish :latest manifests for a complete pair of content-tagged manylinux
# wheel-builder images.

set -eu

repository="${GITHUB_REPOSITORY:-tenstorrent/tt-lang}"
docker_tag=""

usage() {
    echo "Usage: $0 [--repository <owner/repo>] --docker-tag <tag>" >&2
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
        *)
            usage
            ;;
    esac
done

case "$repository" in
    */*) ;;
    *) usage ;;
esac
[ -n "$docker_tag" ] || usage

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/docker-image-utils.sh
. "$script_dir/lib/docker-image-utils.sh"

for python_tag in cp310 cp312; do
    image="$(ttlang_wheel_builder_registry_image \
        "$python_tag" "$docker_tag" "ghcr.io/${repository}")"
    if ! ${DOCKER:-docker} manifest inspect "$image" >/dev/null 2>&1; then
        echo "Required image does not exist: $image" >&2
        exit 1
    fi
done

for python_tag in cp310 cp312; do
    image="$(ttlang_wheel_builder_registry_image \
        "$python_tag" "$docker_tag" "ghcr.io/${repository}")"
    latest="${image%:*}:latest"
    ${DOCKER:-docker} buildx imagetools create -t "$latest" "$image"
    echo "Published: $latest -> $image"
done
