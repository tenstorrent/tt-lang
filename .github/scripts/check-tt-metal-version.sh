#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify (or update) that everything tied to tt-metal points at the same
# release tag. The single source of truth is third-party/tt-metal-version.
#
# Checks:
#   - third-party/tt-metal-version is well-formed and points at a real
#     tt-metal release tag
#   - third-party/tt-metal submodule HEAD == commit pointed to by the tag
#   - Dockerfile.base does not hard-code a tt-metal SHA
#
# The ttnn version in `pyproject.toml`'s `[project.optional-dependencies]
# device` is derived dynamically from this same file by
# setup.py:_ttnn_device_extras(); no separate verification is needed.
#
# Usage:
#   .github/scripts/check-tt-metal-version.sh           # verify only (CI mode)
#   .github/scripts/check-tt-metal-version.sh --update  # check out submodule
#                                                       # at the tag's commit

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
VERSION_FILE="$ROOT/third-party/tt-metal-version"
DOCKERFILE="$ROOT/.github/containers/Dockerfile.base"
SUBMODULE="$ROOT/third-party/tt-metal"
TT_METAL_REMOTE="https://github.com/tenstorrent/tt-metal"

UPDATE=0
[[ "${1:-}" == "--update" ]] && UPDATE=1

[[ -f "$VERSION_FILE" ]] || { echo "missing $VERSION_FILE" >&2; exit 1; }
TAG=$(tr -d '[:space:]' < "$VERSION_FILE")
[[ -n "$TAG" ]] || { echo "$VERSION_FILE is empty" >&2; exit 1; }
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]] \
  || { echo "$VERSION_FILE: '$TAG' does not look like vX.Y.Z" >&2; exit 1; }
VERSION=${TAG#v}

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
    git -C "$ROOT" submodule update --init "$SUBMODULE"
    git -C "$SUBMODULE" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
    git -C "$SUBMODULE" checkout --detach "$RESOLVED"
    echo "updated: third-party/tt-metal gitlink ${GITLINK_SHA:0:12} -> ${RESOLVED:0:12} ($TAG)"
  else
    echo "drift: third-party/tt-metal gitlink is ${GITLINK_SHA:0:12}, expected ${RESOLVED:0:12} ($TAG); run: $0 --update" >&2
    exit 1
  fi
fi

if (( UPDATE )); then
  echo "ok: submodule checked out at $TAG ($(echo "$RESOLVED" | cut -c1-12))"
else
  echo "ok: tt-metal $TAG ($(echo "$RESOLVED" | cut -c1-12)) matches submodule (ttnn version derived dynamically by setup.py)"
fi
