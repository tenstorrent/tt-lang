#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Probe GHCR for the ird and dist container images at the given docker tag.
#
# Writes needs_rebuild=true|false to $GITHUB_OUTPUT.
# Rebuild is skipped only when BOTH the ird and dist images exist; if either
# is absent the build must run so tutorials (which use the dist image) do not
# fail against a missing image.
#
# Refuses to rebuild a bare release tag (vX.Y.Z without a deterministic hash suffix)
# when an image is missing: pushing PR/main content under the release
# tag would silently corrupt the release image in GHCR. publish-pypi.yml
# publishes the release tag on a tag push; recover a missing one by running
# call-build-docker.yml with push=true at that exact tag.
#
# Usage: probe-docker-image.sh <tag>

set -euo pipefail

TAG="${1:?usage: probe-docker-image.sh <tag>}"
IRD_IMAGE="ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-24-04:${TAG}"
DIST_IMAGE="ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-24-04:${TAG}"

IRD_EXISTS=false
DIST_EXISTS=false

if docker manifest inspect "$IRD_IMAGE" >/dev/null 2>&1; then
    IRD_EXISTS=true
    echo "Image exists: $IRD_IMAGE"
else
    echo "Image missing: $IRD_IMAGE"
fi

if docker manifest inspect "$DIST_IMAGE" >/dev/null 2>&1; then
    DIST_EXISTS=true
    echo "Image exists: $DIST_IMAGE"
else
    echo "Image missing: $DIST_IMAGE"
fi

if $IRD_EXISTS && $DIST_EXISTS; then
    echo "needs_rebuild=false" >> "$GITHUB_OUTPUT"
    exit 0
fi

if [[ ! "$TAG" =~ -(uplift-)?[0-9a-f]{8}$ ]]; then
    echo "::error::Release image(s) missing from GHCR at tag $TAG. Refusing to rebuild from a non-release context. Run call-build-docker.yml with push=true at ref $TAG."
    exit 1
fi

echo "needs_rebuild=true" >> "$GITHUB_OUTPUT"
echo "Rebuild required for tag $TAG"
