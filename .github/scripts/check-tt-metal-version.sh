#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify (or update) that everything tied to tt-metal points at the same
# release tag. The single source of truth is third-party/tt-metal-version,
# a sourceable shell snippet defining TTNN_PYPI, TTNN_PYPI_TT_METAL_TAG, and
# TT_METAL_TAG. See that file's header for variable semantics.
#
# Checks:
#   - TT_METAL_TAG points at a real tt-metal release tag
#   - TTNN_PYPI_TT_METAL_TAG is recorded for public PyPI publish alignment
#   - third-party/tt-metal submodule HEAD == commit pointed to by the tag
#   - Dockerfile.base does not hard-code a tt-metal SHA
#
# Usage:
#   .github/scripts/check-tt-metal-version.sh           # verify only (CI mode)
#   .github/scripts/check-tt-metal-version.sh --update  # check out submodule
#                                                       # at the tag's commit

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/tt-metal-version-utils.sh
. "$script_dir/lib/tt-metal-version-utils.sh"

ROOT=$(git rev-parse --show-toplevel)
VERSION_FILE="$ROOT/third-party/tt-metal-version"
DOCKERFILE="$ROOT/.github/containers/Dockerfile.base"
SUBMODULE="$ROOT/third-party/tt-metal"
TT_METAL_REMOTE="https://github.com/tenstorrent/tt-metal"

UPDATE=0
[[ "${1:-}" == "--update" ]] && UPDATE=1

load_tt_metal_version "$VERSION_FILE"
TAG="$TT_METAL_TAG"
PYPI="$TTNN_PYPI"
PYPI_TAG="$TTNN_PYPI_TT_METAL_TAG"
require_semver_tag() {
  local name=$1 value=$2
  [[ "$value" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]] \
    || { echo "$VERSION_FILE: $name '$value' does not look like vX.Y.Z" >&2; exit 1; }
}
require_semver_tag TT_METAL_TAG "$TAG"
require_semver_tag TTNN_PYPI_TT_METAL_TAG "$PYPI_TAG"

# Resolve tag -> commit via ls-remote. Annotated tags get a `^{}` deref line.
RESOLVED=$(git ls-remote --tags "$TT_METAL_REMOTE" \
  "refs/tags/$TAG" "refs/tags/$TAG^{}" \
  | awk '$2 ~ /\^\{\}$/ {deref=$1} $2 !~ /\^\{\}$/ {direct=$1} END {print (deref ? deref : direct)}')
[[ -n "$RESOLVED" ]] \
  || { echo "tt-metal has no release tag $TAG" >&2; exit 1; }

# --- Dockerfile.base: must not hard-code a SHA -----------------------------
if grep -qE 'TT_METAL_DEPENDENCIES_COMMIT=[0-9a-f]{40}' "$DOCKERFILE"; then
  echo "drift: $DOCKERFILE still hard-codes TT_METAL_DEPENDENCIES_COMMIT=<sha>; replace with TT_METAL_TAG and let build-docker-images.sh pass it from $VERSION_FILE" >&2
  exit 1
fi

# --- third-party/tt-metal submodule ---------------------------------------
# Read the gitlink SHA recorded in the parent tree. This works without
# the submodule being checked out (CI checks out submodules: false).
GITLINK_SHA=$(git -C "$ROOT" ls-tree HEAD third-party/tt-metal | awk '{print $3}')
[[ -n "$GITLINK_SHA" ]] \
  || { echo "no gitlink for third-party/tt-metal in HEAD" >&2; exit 1; }

if [[ "$GITLINK_SHA" != "$RESOLVED" ]]; then
  if (( UPDATE )); then
    # Nuke and re-init. The simpler in-place sequence (fetch + checkout +
    # recursive submodule update) leaves stale state behind when bumping
    # tt-metal across versions: shallow clones drop files, nested
    # submodules stay at the previous tt-metal's SHAs, and untracked
    # artifacts mirrored into the source tree (e.g. _ttnn.so from a prior
    # build) survive. Removing the directory and re-cloning is slower
    # (~30s + CPM cache re-population) but guarantees a clean state
    # matching the new tag exactly. Anything saved under
    # third-party/tt-metal that you want to keep should live elsewhere.
    echo "Removing third-party/tt-metal for a clean re-clone..."
    git -C "$ROOT" submodule deinit -f "$SUBMODULE" 2>/dev/null || true
    rm -rf "$SUBMODULE"
    git -C "$ROOT" submodule update --init "$SUBMODULE"
    git -C "$SUBMODULE" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
    git -C "$SUBMODULE" checkout --detach "$RESOLVED"
    git -C "$SUBMODULE" submodule update --init --recursive --depth 1
    echo "updated: third-party/tt-metal gitlink ${GITLINK_SHA:0:12} -> ${RESOLVED:0:12} ($TAG)"
  else
    echo "drift: third-party/tt-metal gitlink is ${GITLINK_SHA:0:12}, expected ${RESOLVED:0:12} ($TAG); run: $0 --update" >&2
    exit 1
  fi
fi

if (( UPDATE )); then
  echo "ok: submodule re-cloned at $TAG ($(echo "$RESOLVED" | cut -c1-12)) with nested submodules"
else
  echo "ok: tt-metal $TAG ($(echo "$RESOLVED" | cut -c1-12)) matches submodule; setup.py requires ttnn==$PYPI from $PYPI_TAG"
fi
