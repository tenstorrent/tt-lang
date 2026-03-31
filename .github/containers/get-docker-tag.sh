#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Generate deterministic Docker tags from file content and submodule hashes
# Usage: ./get-docker-tag.sh
#
# Must be run from the repository root directory

set -e

# Submodule SHAs that affect the Docker image
LLVM_SHA=$(git -C third-party/llvm-project rev-parse HEAD 2>/dev/null || echo "unknown")
TTMLIR_SHA=$(git -C third-party/tt-mlir rev-parse HEAD 2>/dev/null || echo "unknown")
TTMETAL_SHA=$(git -C third-party/tt-metal rev-parse HEAD 2>/dev/null || echo "unknown")

# Files that affect Docker image content
# Changes to any of these files will result in a new Docker tag
FILES="\
.github/containers/Dockerfile.base \
.github/containers/Dockerfile \
requirements.txt \
dev-requirements.txt"

# Compute hash of all tracked files
HASH=$(sha256sum $FILES 2>/dev/null | sha256sum | cut -d ' ' -f 1)

# Combine file hash with submodule SHAs to create unique tag
COMBINED=$(echo "${HASH}${LLVM_SHA}${TTMLIR_SHA}${TTMETAL_SHA}" | sha256sum | cut -d ' ' -f 1)

# Output short tag (12 chars is enough for uniqueness)
echo "dt-${COMBINED:0:12}"
