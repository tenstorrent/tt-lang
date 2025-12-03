#!/bin/bash
# Generate standardized artifact archive name with OS and git SHA
# Usage: get-artifact-archive-name.sh <os> <git-sha>
# Example: get-artifact-archive-name.sh Linux abc123def456...
# Output: ttlang-build-artifacts-Linux-abc123de.tar.zst

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <os> <git-sha>" >&2
    echo "Example: $0 Linux abc123def456" >&2
    exit 1
fi

OS="$1"
GIT_SHA="$2"

# Extract first 8 characters of SHA
SHORT_SHA="${GIT_SHA:0:8}"

# Generate artifact name
ARTIFACT_NAME="ttlang-build-artifacts-${OS}-${SHORT_SHA}.tar.zst"

echo "${ARTIFACT_NAME}"
