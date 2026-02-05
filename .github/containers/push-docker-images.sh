#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Push locally built Docker images to ghcr.io with :latest tag
#
# Usage: ./push-docker-images.sh
#
# Note: You must be logged in to ghcr.io first:
#   echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

set -e

REPO=tenstorrent/tt-lang

echo "=== Pushing tt-lang Docker Images to ghcr.io ==="
echo ""

# Tag and push base image
echo "--- Pushing tt-lang-base ---"
docker tag tt-lang-base:local ghcr.io/$REPO/tt-lang-base-ubuntu-22-04:latest
docker push ghcr.io/$REPO/tt-lang-base-ubuntu-22-04:latest
echo "✓ Base image pushed"
echo ""

# Tag and push CI image
echo "--- Pushing tt-lang-ci ---"
docker tag tt-lang-ci-ubuntu-22-04:latest ghcr.io/$REPO/tt-lang-ci-ubuntu-22-04:latest
docker push ghcr.io/$REPO/tt-lang-ci-ubuntu-22-04:latest
echo "✓ CI image pushed"
echo ""

# Tag and push dist alias (same as CI)
echo "--- Pushing tt-lang-dist (alias for ci) ---"
docker tag tt-lang-ci-ubuntu-22-04:latest ghcr.io/$REPO/tt-lang-dist-ubuntu-22-04:latest
docker push ghcr.io/$REPO/tt-lang-dist-ubuntu-22-04:latest
echo "✓ Dist image pushed"
echo ""

# Tag and push IRD image if it exists
if docker images tt-lang-ird-ubuntu-22-04:latest --format "{{.ID}}" | grep -q .; then
    echo "--- Pushing tt-lang-ird ---"
    docker tag tt-lang-ird-ubuntu-22-04:latest ghcr.io/$REPO/tt-lang-ird-ubuntu-22-04:latest
    docker push ghcr.io/$REPO/tt-lang-ird-ubuntu-22-04:latest
    echo "✓ IRD image pushed"
    echo ""
else
    echo "⚠ IRD image not found locally, skipping"
    echo ""
fi

echo "=== Push Complete ==="
echo ""
echo "Images pushed to ghcr.io/$REPO:"
echo "  - tt-lang-base-ubuntu-22-04:latest"
echo "  - tt-lang-ci-ubuntu-22-04:latest"
echo "  - tt-lang-dist-ubuntu-22-04:latest (alias for ci)"
if docker images tt-lang-ird-ubuntu-22-04:latest --format "{{.ID}}" | grep -q .; then
    echo "  - tt-lang-ird-ubuntu-22-04:latest"
fi
