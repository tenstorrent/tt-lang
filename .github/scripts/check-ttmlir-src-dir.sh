#!/bin/bash
# Check if TTMLIR_SRC_DIR matches the expected commit SHA
# Usage: check-ttmlir-src-dir.sh <TTMLIR_SRC_DIR> <EXPECTED_SHA>
# Exit code: 0 if SHA matches, 1 otherwise

set -e

TTMLIR_SRC_DIR="${1:-}"
EXPECTED_SHA="${2:-}"

if [ -z "$TTMLIR_SRC_DIR" ] || [ -z "$EXPECTED_SHA" ]; then
    echo "ERROR: Usage: check-ttmlir-src-dir.sh <TTMLIR_SRC_DIR> <EXPECTED_SHA>"
    exit 1
fi

if [ ! -d "$TTMLIR_SRC_DIR" ]; then
    echo "ERROR: TTMLIR_SRC_DIR does not exist: $TTMLIR_SRC_DIR"
    exit 1
fi

if [ ! -d "$TTMLIR_SRC_DIR/.git" ]; then
    echo "ERROR: TTMLIR_SRC_DIR is not a git repository: $TTMLIR_SRC_DIR"
    exit 1
fi

# Get the current commit SHA
CURRENT_SHA=$(cd "$TTMLIR_SRC_DIR" && git rev-parse HEAD 2>/dev/null || echo "")

if [ -z "$CURRENT_SHA" ]; then
    echo "ERROR: Failed to get commit SHA from $TTMLIR_SRC_DIR"
    exit 1
fi

# Compare SHAs
if [ "$CURRENT_SHA" = "$EXPECTED_SHA" ]; then
    # Initialize submodules
    if ! (cd "$TTMLIR_SRC_DIR" && git submodule update --init --recursive >/dev/null 2>&1); then
        echo "WARNING: Failed to initialize submodules, continuing anyway"
    fi
    exit 0
else
    echo "SHA mismatch: expected $EXPECTED_SHA, got $CURRENT_SHA"
    exit 1
fi
