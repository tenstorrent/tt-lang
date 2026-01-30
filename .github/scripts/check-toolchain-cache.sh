#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Check if toolchain cache was restored and provide helpful error if not.
#
# Usage: check-toolchain-cache.sh <cache-hit> <ttmlir-commit> <repository>
#
# Arguments:
#   cache-hit    - "true" if cache was restored, anything else means miss
#   ttmlir-commit - The tt-mlir commit SHA
#   repository   - GitHub repository (e.g., "tenstorrent/tt-lang")

set -e

CACHE_HIT="$1"
TTMLIR_COMMIT="$2"
REPOSITORY="${3:-tenstorrent/tt-lang}"

if [ "$CACHE_HIT" = "true" ]; then
    echo "Toolchain cache restored successfully"
    exit 0
fi

# Cache miss - print helpful error
echo "::error::Toolchain cache not found for tt-mlir commit ${TTMLIR_COMMIT}"
echo ""
echo "============================================================"
echo "  TOOLCHAIN CACHE NOT FOUND"
echo "============================================================"
echo ""
echo "The LLVM + tt-mlir toolchain cache does not exist for this commit."
echo ""
echo "To fix this, run the toolchain build workflow:"
echo ""
echo "  1. Go to: https://github.com/${REPOSITORY}/actions/workflows/call-build-ttmlir-toolchain.yml"
echo "  2. Click 'Run workflow' button"
echo "  3. Click the green 'Run workflow' button in the dropdown"
echo "  4. Wait for completion (~3-4 hours for full LLVM build)"
echo "  5. Re-run this workflow"
echo ""
echo "Cache details:"
echo "  Key: Linux-ttlang-toolchain-v1-${TTMLIR_COMMIT}"
echo "  tt-mlir commit: ${TTMLIR_COMMIT}"
echo ""
echo "See .github/CI_WORKFLOWS.md for more information."
echo "============================================================"
exit 1
