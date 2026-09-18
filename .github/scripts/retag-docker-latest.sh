#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Point the dist and ird :latest tags at an already-published image.
#
# build-docker pushes :latest but runs only when the image is missing, so an
# image published before the main push leaves :latest stale. Copies the manifest
# rather than rebuilding.
#
# Usage: retag-docker-latest.sh <tag>

set -euo pipefail

TAG="${1:?usage: retag-docker-latest.sh <tag>}"
REPO="${GITHUB_REPOSITORY:-tenstorrent/tt-lang}"
REGISTRY="${TTLANG_REGISTRY:-ghcr.io}"

for name in tt-lang-dist-ubuntu-24-04 tt-lang-ird-ubuntu-24-04; do
    image="${REGISTRY}/${REPO}/${name}"
    if ! docker manifest inspect "${image}:${TAG}" >/dev/null 2>&1; then
        echo "::error::${image}:${TAG} is missing from the registry; refusing to move :latest" >&2
        exit 1
    fi
    docker buildx imagetools create --tag "${image}:latest" "${image}:${TAG}"
    echo "Pointed ${image}:latest at ${TAG}"
done
