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

# Tag and push dev image
echo "--- Pushing tt-lang-dev ---"
docker tag tt-lang-dev:local ghcr.io/$REPO/tt-lang-dev-ubuntu-22-04:latest
docker push ghcr.io/$REPO/tt-lang-dev-ubuntu-22-04:latest
echo "✓ Dev image pushed"
echo ""

# Tag and push user image
echo "--- Pushing tt-lang-user ---"
docker tag tt-lang-user:local ghcr.io/$REPO/tt-lang-user-ubuntu-22-04:latest
docker push ghcr.io/$REPO/tt-lang-user-ubuntu-22-04:latest
echo "✓ User image pushed"
echo ""

echo "=== Push Complete ==="
echo ""
echo "Images pushed to ghcr.io/$REPO:"
echo "  - tt-lang-dev-ubuntu-22-04:latest (tt-mlir toolchain + dev tools)"
echo "  - tt-lang-user-ubuntu-22-04:latest (dev + pre-built tt-lang)"
