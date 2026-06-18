# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Sourceable helpers for reading third-party/tt-metal-version, the single
# source of truth for the tt-metal release tag (TT_METAL_TAG), the public ttnn
# wheel version (TTNN_PYPI), and the tt-metal tag that ttnn wheel was built from
# (TTNN_PYPI_TT_METAL_TAG).
#
# This is library code for the workflow scripts; it is not the version file
# itself. Source it; do not execute it. To change tt-metal pinning, edit
# third-party/tt-metal-version instead.

# Resolve the version-file path. Honors $TTLANG_TT_METAL_VERSION_FILE so tests
# can point at a synthetic file; otherwise uses <repo-root>/third-party/tt-metal-version.
tt_metal_version_file() {
    if [[ -n "${TTLANG_TT_METAL_VERSION_FILE:-}" ]]; then
        printf '%s\n' "$TTLANG_TT_METAL_VERSION_FILE"
        return 0
    fi
    printf '%s/third-party/tt-metal-version\n' "$(git rev-parse --show-toplevel)"
}

# Source the version file and require its three fields, defining TT_METAL_TAG,
# TTNN_PYPI, and TTNN_PYPI_TT_METAL_TAG in the caller's scope. Pass an explicit
# file to bypass tt_metal_version_file resolution.
load_tt_metal_version() {
    local version_file="${1:-}"
    [[ -n "$version_file" ]] || version_file="$(tt_metal_version_file)"
    [[ -f "$version_file" ]] || { echo "missing $version_file" >&2; return 1; }
    # shellcheck source=/dev/null
    . "$version_file"
    : "${TT_METAL_TAG:?$version_file: TT_METAL_TAG not set}"
    : "${TTNN_PYPI:?$version_file: TTNN_PYPI not set}"
    : "${TTNN_PYPI_TT_METAL_TAG:?$version_file: TTNN_PYPI_TT_METAL_TAG not set}"
}

tt_metal_release_component() {
    local tag="$1"
    if [[ "$tag" =~ ^(v[0-9]+\.[0-9]+\.[0-9]+)($|[-+]) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

# True when the public ttnn wheel and this release use the same tt-metal
# release component. This treats a release candidate tag such as vX.Y.Z-rcN as
# compatible with the final vX.Y.Z tag. Requires load_tt_metal_version to have
# run.
ttnn_pypi_aligned() {
    local pypi_release_component tt_release_component
    pypi_release_component="$(tt_metal_release_component "$TTNN_PYPI_TT_METAL_TAG")" || return 1
    tt_release_component="$(tt_metal_release_component "$TT_METAL_TAG")" || return 1
    [[ "$pypi_release_component" == "$tt_release_component" ]]
}
