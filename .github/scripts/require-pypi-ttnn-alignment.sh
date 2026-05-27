#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Refuse public PyPI publishing when the ttnn wheel dependency was built from a
# different tt-metal tag than the one used to build this tt-lang release.

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
VERSION_FILE="$ROOT/third-party/tt-metal-version"

[[ -f "$VERSION_FILE" ]] || { echo "missing $VERSION_FILE" >&2; exit 1; }

# shellcheck source=../../third-party/tt-metal-version
. "$VERSION_FILE"
: "${TTNN_PYPI:?$VERSION_FILE: TTNN_PYPI not set}"
: "${TTNN_PYPI_TT_METAL_TAG:?$VERSION_FILE: TTNN_PYPI_TT_METAL_TAG not set}"
: "${TT_METAL_TAG:?$VERSION_FILE: TT_METAL_TAG not set}"

if [[ "$TTNN_PYPI_TT_METAL_TAG" != "$TT_METAL_TAG" ]]; then
  cat >&2 <<EOF
Public PyPI publish requires ttnn provenance to match TT_METAL_TAG.
TTNN_PYPI=$TTNN_PYPI was built from TTNN_PYPI_TT_METAL_TAG=$TTNN_PYPI_TT_METAL_TAG,
but this release builds against TT_METAL_TAG=$TT_METAL_TAG.
Use the S3 bundled wheel workflow for this tt-metal selection, or publish after
ttnn is available for TT_METAL_TAG=$TT_METAL_TAG.
EOF
  exit 1
fi

echo "ok: ttnn==$TTNN_PYPI and tt-lang both use tt-metal $TT_METAL_TAG"
