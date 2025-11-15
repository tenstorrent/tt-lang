#!/bin/bash
# Determine the tt-mlir commit to use
# Outputs: commit=<SHA> to GITHUB_OUTPUT

set -e

COMMIT_INPUT="${1:-}"
COMMIT_FILE="${2:-third-party/tt-mlir.commit}"

if [ -n "$COMMIT_INPUT" ]; then
    COMMIT="$COMMIT_INPUT"
    echo "Using tt-mlir commit from input: $COMMIT"
else
    if [ ! -f "$COMMIT_FILE" ]; then
        echo "ERROR: $COMMIT_FILE file not found"
        exit 1
    fi
    COMMIT=$(cat "$COMMIT_FILE" | tr -d '[:space:]')
    if [ -z "$COMMIT" ]; then
        echo "ERROR: $COMMIT_FILE is empty"
        exit 1
    fi
    echo "Using tt-mlir commit from file: $COMMIT"
fi

if [ -n "$GITHUB_OUTPUT" ]; then
    echo "commit=$COMMIT" >> "$GITHUB_OUTPUT"
else
    # Fallback for local testing
    echo "commit=$COMMIT"
fi
