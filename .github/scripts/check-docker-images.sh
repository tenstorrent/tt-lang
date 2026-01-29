#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Check if Docker images exist and set GitHub Actions outputs.
#
# Usage:
#   ./check-docker-images.sh <mlir-sha>
#
# Outputs (via GITHUB_OUTPUT):
#   docker-image: Full image name if exists, empty if not
#
# Exit codes:
#   0: Always (uses outputs to signal state, not exit codes)
#   2: Missing required argument

set -e

MLIR_SHA="${1:-}"

if [ -z "$MLIR_SHA" ]; then
    echo "Error: MLIR_SHA argument required"
    exit 2
fi

# Run the check-only script and capture its exit code
set +e
.github/containers/build-docker-images.sh "$MLIR_SHA" --check-only | tee /tmp/docker-check.log
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [ $EXIT_CODE -eq 0 ]; then
    # Images exist - extract image name from last line
    DOCKER_IMAGE=$(tail -n 1 /tmp/docker-check.log)
    echo "docker-image=$DOCKER_IMAGE" >> "$GITHUB_OUTPUT"
    echo "✓ Docker images already exist"
    exit 0
else
    # Images don't exist - set empty outputs so build job runs
    echo "docker-image=" >> "$GITHUB_OUTPUT"
    echo "ℹ Docker images need to be built (not cached)"
    # Exit 0 - this is not an error condition, just indicates build is needed
    exit 0
fi
