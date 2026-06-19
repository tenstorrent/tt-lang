#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build one S3 tt-lang-light core wheel inside a manylinux_2_34 builder image.

set -eu

PYTHON_TAG=""
VERSION=""
BUILD_DIR=""
RAW_DIR=""
DIST_DIR=dist
ALLOW_FINAL_INTERNAL_VERSION="${TTLANG_ALLOW_FINAL_INTERNAL_VERSION:-false}"

usage() {
    cat >&2 <<'EOF'
Usage: build-s3-light-core-wheel.sh --python-tag cp310|cp312 --version <version> [options]

Options:
  --build-dir <dir>               CMake build directory. Default: build-<python-tag>.
  --raw-dir <dir>                 Unrepaired wheel directory. Default: dist-raw-<python-tag>.
  --dist-dir <dir>                Final wheel directory. Default: dist.
  --allow-final-internal-version  Allow final release versions for internal light wheels.
EOF
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python-tag)
            if [ "$#" -lt 2 ]; then usage; fi
            PYTHON_TAG="$2"
            shift 2
            ;;
        --version)
            if [ "$#" -lt 2 ]; then usage; fi
            VERSION="$2"
            shift 2
            ;;
        --build-dir)
            if [ "$#" -lt 2 ]; then usage; fi
            BUILD_DIR="$2"
            shift 2
            ;;
        --raw-dir)
            if [ "$#" -lt 2 ]; then usage; fi
            RAW_DIR="$2"
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

case "$PYTHON_TAG" in
    cp310 | cp312) ;;
    *) echo "Unsupported Python tag: $PYTHON_TAG" >&2; exit 2 ;;
esac

if [ -z "$VERSION" ]; then
    echo "--version is required" >&2
    exit 2
fi

BUILD_DIR="${BUILD_DIR:-build-${PYTHON_TAG}}"
RAW_DIR="${RAW_DIR:-dist-raw-${PYTHON_TAG}}"

repo_root="$(git rev-parse --show-toplevel)"
version_output="$(mktemp)"
trap 'rm -f "$version_output"' EXIT
TTNN_DEP_MODE=external \
VERSION_OVERRIDE="$VERSION" \
GITHUB_OUTPUT="$version_output" \
    "$repo_root/.github/scripts/resolve-wheel-versions.sh"
core_version="$(sed -n 's/^core_version=//p' "$version_output")"

. /opt/ttlang-toolchain/venv/bin/activate

rm -rf "$BUILD_DIR" "$RAW_DIR"
mkdir -p "$DIST_DIR"
rm -f "$DIST_DIR"/tt_lang-*-"${PYTHON_TAG}"-"${PYTHON_TAG}"-manylinux_2_34_x86_64.whl

export CMAKE_BINARY_DIR="$BUILD_DIR"
export TTLANG_TTNN_DEP_MODE=external
export TTLANG_VERSION_OVERRIDE="$core_version"
export TTLANG_EXTERNAL_TT_METAL_DIR="${TTLANG_EXTERNAL_TT_METAL_DIR:-/opt/ttlang-toolchain/tt-metal}"
export TTLANG_PYTHON_VENV="${TTLANG_PYTHON_VENV:-/opt/ttlang-toolchain/venv}"
export TTLANG_ALLOW_FINAL_INTERNAL_VERSION="$ALLOW_FINAL_INTERNAL_VERSION"

"$repo_root/.github/scripts/configure-ttlang-build.sh" "$BUILD_DIR"
python -m pip wheel . --wheel-dir="$RAW_DIR" --no-deps --no-build-isolation
auditwheel repair \
    --plat manylinux_2_34_x86_64 \
    "$RAW_DIR"/tt_lang-*.whl \
    --wheel-dir="$DIST_DIR"

expected_wheel="$DIST_DIR/tt_lang-${core_version}-${PYTHON_TAG}-${PYTHON_TAG}-manylinux_2_34_x86_64.whl"
if [ ! -f "$expected_wheel" ]; then
    echo "Expected wheel was not produced: $expected_wheel" >&2
    ls -lh "$DIST_DIR" >&2 || true
    exit 1
fi

auditwheel show "$expected_wheel"
python "$repo_root/.github/scripts/check-wheel-ttnn-metadata.py" \
    --mode external \
    --dist-dir "$DIST_DIR"
