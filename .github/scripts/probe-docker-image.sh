#!/bin/bash
# Probe GHCR for the ird container image at the given docker tag.
#
# Writes needs_rebuild=true|false to $GITHUB_OUTPUT.
#
# Refuses to rebuild a bare release tag (vX.Y.Z without -uplift- suffix)
# when the image is missing: pushing PR/main content under the release
# tag would silently corrupt the release image in GHCR. Only
# publish-pypi.yml on a tag push may rebuild the release tag.
#
# Usage: probe-docker-image.sh <tag>

set -e

TAG="${1:?usage: probe-docker-image.sh <tag>}"
IMAGE="ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04:${TAG}"

if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
    echo "needs_rebuild=false" >> "$GITHUB_OUTPUT"
    echo "Image exists: $IMAGE"
    exit 0
fi

if [[ "$TAG" != *-uplift-* ]]; then
    echo "::error::Release image $IMAGE is missing from GHCR. Refusing to rebuild from a non-release context — re-publish via publish-pypi.yml (workflow_dispatch on tag $TAG)."
    exit 1
fi

echo "needs_rebuild=true" >> "$GITHUB_OUTPUT"
echo "Image missing: $IMAGE (will rebuild)"
