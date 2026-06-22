#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build the pure-Python tt-lang-light metapackage wheel.

set -eu

VERSION=""
DIST_DIR=dist
ALLOW_FINAL_INTERNAL_VERSION="${TTLANG_ALLOW_FINAL_INTERNAL_VERSION:-false}"

usage() {
    cat >&2 <<'EOF'
Usage: build-s3-light-metapackage-wheel.sh --version <version> [options]

Options:
  --dist-dir <dir>                Final wheel directory. Default: dist.
  --allow-final-internal-version  Allow final release versions for internal light wheels.
EOF
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            if [ "$#" -lt 2 ]; then usage; fi
            VERSION="$2"
            shift 2
            ;;
        --dist-dir)
            if [ "$#" -lt 2 ]; then usage; fi
            DIST_DIR="$2"
            shift 2
            ;;
        --allow-final-internal-version)
            ALLOW_FINAL_INTERNAL_VERSION=true
            shift
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$VERSION" ]; then
    echo "--version is required" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
version_output="$(mktemp)"
trap 'rm -f "$version_output"' EXIT
TTNN_DEP_MODE=external \
VERSION_OVERRIDE="$VERSION" \
GITHUB_OUTPUT="$version_output" \
    "$repo_root/.github/scripts/resolve-wheel-versions.sh"
light_version="$(sed -n 's/^light_version=//p' "$version_output")"

. /opt/ttlang-toolchain/venv/bin/activate

mkdir -p "$DIST_DIR"
rm -f "$DIST_DIR"/tt_lang_light-*.whl

TTLANG_VERSION_OVERRIDE="$VERSION" \
TTLANG_LIGHT_TTLANG_VERSION="$light_version" \
TTLANG_ALLOW_FINAL_INTERNAL_VERSION="$ALLOW_FINAL_INTERNAL_VERSION" \
    python -m pip wheel packaging/light \
        --wheel-dir="$DIST_DIR" \
        --no-deps \
        --no-build-isolation

expected_wheel="$DIST_DIR/tt_lang_light-${VERSION}-py3-none-any.whl"
if [ ! -f "$expected_wheel" ]; then
    echo "Expected metapackage wheel was not produced: $expected_wheel" >&2
    ls -lh "$DIST_DIR" >&2 || true
    exit 1
fi

python "$repo_root/.github/scripts/check-light-metapackage.py" \
    --dist-dir "$DIST_DIR" \
    --expect-ttlang-version "$light_version"
