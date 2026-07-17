#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Refuse to run unless the supplied ref, or GITHUB_REF by default, identifies a
# public PyPI release tag, then print the version for downstream steps.
#
# Usage: .github/scripts/require-release-tag.sh [release-ref]
# Reads:  $GITHUB_REF when release-ref is omitted
# Writes: tag_version=<MAJOR.MINOR.PATCH...> to $GITHUB_OUTPUT (if set)
# Stdout: <MAJOR.MINOR.PATCH...>

set -euo pipefail

ref="${1:-${GITHUB_REF:-}}"
if [[ ! "$ref" =~ ^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z][0-9A-Za-z.-]*)?$ ]]; then
    echo "Release ref must match refs/tags/vX.Y.Z or refs/tags/vX.Y.Z-<prerelease> (got '$ref')." >&2
    echo "Create and push a tag like 'vX.Y.Z', or pass 'refs/tags/vX.Y.Z' as release-ref." >&2
    exit 1
fi

tag_version_raw="${ref#refs/tags/v}"

# Normalize to the PEP 440 canonical form so it matches what setuptools writes
# into the wheel filename (tag 'vX.Y.Z-devYYYYMMDD' -> wheel version
# 'X.Y.Z.devYYYYMMDD'; tag 'vX.Y.Z-rcN' -> 'X.Y.ZrcN').
# verify-wheel-version.sh compares the wheel filename's version field to this
# output, so both must use the canonical form.
set +e
tag_version=$(python3 - "$tag_version_raw" <<'PY'
import sys
from packaging.version import Version, InvalidVersion
try:
    print(str(Version(sys.argv[1])))
except InvalidVersion:
    sys.exit(2)
PY
)
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
    echo "Tag '${ref#refs/tags/}' is not a valid PEP 440 version." >&2
    echo "  Use forms like vX.Y.Z, vX.Y.Z-rcN, or vX.Y.Z-devYYYYMMDD." >&2
    exit 1
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "tag_version=${tag_version}" >> "$GITHUB_OUTPUT"
fi
echo "$tag_version"
