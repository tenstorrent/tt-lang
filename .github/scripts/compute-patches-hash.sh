#!/bin/bash
# Compute hash of tt-mlir patches
# Outputs: hash=<SHA256> to GITHUB_OUTPUT

set -e

PATCHES_DIR="${1:-third-party/patches}"

if [ -d "$PATCHES_DIR" ]; then
    # Find all tt-mlir*.patch files, compute their sha256, sort, and hash the result
    PATCHES_HASH=$(find "$PATCHES_DIR" -name 'tt-mlir*.patch' -type f -exec sha256sum {} \; 2>/dev/null | sort | sha256sum | cut -d' ' -f1)
    if [ -z "$PATCHES_HASH" ]; then
        # No patches found
        PATCHES_HASH="none"
        echo "No tt-mlir patches found"
    else
        echo "Computed patches hash: $PATCHES_HASH"
    fi
else
    # Directory doesn't exist
    PATCHES_HASH="none"
    echo "Patches directory not found, using 'none'"
fi

if [ -n "$GITHUB_OUTPUT" ]; then
    echo "hash=$PATCHES_HASH" >> "$GITHUB_OUTPUT"
else
    # Fallback for local testing
    echo "hash=$PATCHES_HASH"
fi
