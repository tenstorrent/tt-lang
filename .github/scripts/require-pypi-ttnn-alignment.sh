#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Refuse public PyPI publishing when the ttnn wheel dependency was built from a
# different tt-metal release component than the one used to build this tt-lang
# release.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/tt-metal-version-utils.sh
. "$script_dir/lib/tt-metal-version-utils.sh"

load_tt_metal_version

if ! ttnn_pypi_aligned; then
  cat >&2 <<EOF
Public PyPI publish requires ttnn provenance to match the TT_METAL_TAG vX.Y.Z component.
TTNN_PYPI=$TTNN_PYPI was built from TTNN_PYPI_TT_METAL_TAG=$TTNN_PYPI_TT_METAL_TAG,
but this release builds against TT_METAL_TAG=$TT_METAL_TAG.
Use the S3 bundled wheel workflow for this tt-metal selection, or publish after
ttnn is available for TT_METAL_TAG=$TT_METAL_TAG.
EOF
  exit 1
fi

tt_release_component="$(tt_metal_release_component "$TT_METAL_TAG")"
echo "ok: ttnn==$TTNN_PYPI and tt-lang both use tt-metal release $tt_release_component"
