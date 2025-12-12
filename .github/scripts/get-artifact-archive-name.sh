#!/bin/bash
# Generate standardized artifact archive name with git SHA
# Usage: get-artifact-archive-name.sh <git-sha>
# Example: get-artifact-archive-name.sh abc123def456...
# Output: ttlang-build-artifacts-abc123de.tar.zst

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <git-sha>" >&2
    echo "Example: $0 abc123def456" >&2
    exit 1
fi

GIT_SHA="$1"

# Extract first 8 characters of SHA
SHORT_SHA="${GIT_SHA:0:8}"

# Generate artifact name
ARTIFACT_NAME="ttlang-build-artifacts-${SHORT_SHA}.tar.zst"

echo "${ARTIFACT_NAME}"
