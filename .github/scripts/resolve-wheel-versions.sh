#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Validate wheel-build inputs and emit the resolved version strings as step
# outputs for the calling workflow.
#
# Required env:
#   TTNN_DEP_MODE     One of: pypi, external, bundled.
#   VERSION_OVERRIDE  PEP 440 version (may be empty for non-external modes).
#   GITHUB_OUTPUT     Path that receives `core_version=` and `light_version=`.
#                     When unset, the outputs are written to stdout.
#
# Outputs:
#   core_version   Version setup.py should embed in the tt-lang wheel. In
#                  `external` mode this is `${VERSION_OVERRIDE}+light` so the
#                  tt-lang-light metapackage's pin resolves; otherwise it is
#                  `${VERSION_OVERRIDE}` verbatim (possibly empty).
#   light_version  Version the tt-lang-light metapackage should pin. Only
#                  populated in `external` mode; empty otherwise.

set -euo pipefail

: "${TTNN_DEP_MODE:?TTNN_DEP_MODE is required}"
VERSION_OVERRIDE="${VERSION_OVERRIDE:-}"

case "$TTNN_DEP_MODE" in
    pypi|external|bundled) ;;
    *)
        echo "unknown ttnn_dep_mode: $TTNN_DEP_MODE (must be pypi|external|bundled)" >&2
        exit 1
        ;;
esac

if [[ "$TTNN_DEP_MODE" == "external" && -z "$VERSION_OVERRIDE" ]]; then
    echo "ttnn_dep_mode=external requires version_override" >&2
    exit 1
fi

if [[ "$TTNN_DEP_MODE" == "external" ]]; then
    core_version="${VERSION_OVERRIDE}+light"
    light_version="${VERSION_OVERRIDE}+light"
else
    core_version="$VERSION_OVERRIDE"
    light_version=""
fi

output_file="${GITHUB_OUTPUT:-/dev/stdout}"
{
    echo "core_version=$core_version"
    echo "light_version=$light_version"
} >> "$output_file"
