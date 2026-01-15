#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Generate deterministic Docker tags from file content hashes
# Usage: ./get-docker-tag.sh <MLIR_DOCKER_TAG>
#
# Must be run from the repository root directory

set -e

MLIR_DOCKER_TAG=$1
if [ -z "$MLIR_DOCKER_TAG" ]; then
    echo "Error: MLIR_DOCKER_TAG required as first argument" >&2
    echo "Usage: $0 <MLIR_DOCKER_TAG>" >&2
    exit 1
fi

# Files that affect Docker image content
# Changes to any of these files will result in a new Docker tag
FILES="\
.github/containers/Dockerfile.base \
.github/containers/Dockerfile.dist \
requirements.txt \
dev-requirements.txt \
third-party/tt-mlir.commit"

# Compute hash of all tracked files
HASH=$(sha256sum $FILES 2>/dev/null | sha256sum | cut -d ' ' -f 1)

# Combine with MLIR tag to create unique tag
# This ensures tt-lang images are rebuilt when tt-mlir version changes
COMBINED=$(echo "${HASH}${MLIR_DOCKER_TAG}" | sha256sum | cut -d ' ' -f 1)

# Output short tag (12 chars is enough for uniqueness)
echo "dt-${COMBINED:0:12}"
