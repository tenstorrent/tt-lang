#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build one tt-lang core wheel inside a manylinux_2_34 builder image. PyPI mode
# records a dependency on published ttnn; external mode produces the core wheel
# consumed by the tt-lang-light metapackage.

set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

PYTHON_TAG=""
VERSION=""
TTNN_DEP_MODE=""
BUILD_DIR=""
RAW_DIR=""
DIST_DIR=dist
ALLOW_FINAL_INTERNAL_VERSION="${TTLANG_ALLOW_FINAL_INTERNAL_VERSION:-false}"

usage() {
    cat >&2 <<'EOF'
Usage: build-manylinux-core-wheel.sh --python-tag cp310|cp312 --version <version> --ttnn-dep-mode pypi|external [options]

Options:
  --build-dir <dir>               CMake build directory. Default: build-<python-tag>.
  --raw-dir <dir>                 Unrepaired wheel directory. Default: dist-raw-<python-tag>.
  --dist-dir <dir>                Final wheel directory. Default: dist.
  --allow-final-internal-version  Allow final release versions for S3 light wheels.
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
        --ttnn-dep-mode)
            if [ "$#" -lt 2 ]; then usage; fi
            TTNN_DEP_MODE="$2"
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
case "$TTNN_DEP_MODE" in
    pypi | external) ;;
    *) echo "--ttnn-dep-mode must be pypi or external" >&2; exit 2 ;;
esac

BUILD_DIR="${BUILD_DIR:-build-${PYTHON_TAG}}"
RAW_DIR="${RAW_DIR:-dist-raw-${PYTHON_TAG}}"

repo_root="$(git rev-parse --show-toplevel)"
TTLANG_GIT_COMMIT="${TTLANG_GIT_COMMIT:-$(git rev-parse HEAD)}"
TT_METAL_COMMIT="${TT_METAL_COMMIT:-$(git rev-parse HEAD:third-party/tt-metal)}"
export TTLANG_GIT_COMMIT TT_METAL_COMMIT
version_output="$(mktemp)"
trap 'rm -f "$version_output"' EXIT
TTNN_DEP_MODE="$TTNN_DEP_MODE" \
VERSION_OVERRIDE="$VERSION" \
GITHUB_OUTPUT="$version_output" \
    "$script_dir/resolve-wheel-versions.sh"
core_version="$(sed -n 's/^core_version=//p' "$version_output")"

. /opt/ttlang-toolchain/venv/bin/activate

rm -rf "$BUILD_DIR" "$RAW_DIR"
mkdir -p "$DIST_DIR"
rm -f "$DIST_DIR"/tt_lang-*-"${PYTHON_TAG}"-"${PYTHON_TAG}"-manylinux_2_34_x86_64.whl

export CMAKE_BINARY_DIR="$BUILD_DIR"
export TTLANG_TTNN_DEP_MODE="$TTNN_DEP_MODE"
export TTLANG_VERSION_OVERRIDE="$core_version"
export TTLANG_EXTERNAL_TT_METAL_DIR="${TTLANG_EXTERNAL_TT_METAL_DIR:-/opt/ttlang-toolchain/tt-metal}"
export TTLANG_PYTHON_VENV="${TTLANG_PYTHON_VENV:-/opt/ttlang-toolchain/venv}"
export TTLANG_ALLOW_FINAL_INTERNAL_VERSION="$ALLOW_FINAL_INTERNAL_VERSION"

"$script_dir/configure-ttlang-build.sh" "$BUILD_DIR"
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
set -- \
    --mode "$TTNN_DEP_MODE" \
    --dist-dir "$DIST_DIR" \
    --expect-tt-metal-commit "$TT_METAL_COMMIT"
if [ "$TTNN_DEP_MODE" = pypi ]; then
    # shellcheck source=/dev/null
    . "$repo_root/third-party/tt-metal-version"
    : "${TTNN_PYPI:?third-party/tt-metal-version: TTNN_PYPI not set}"
    set -- "$@" --expect-ttnn-version "$TTNN_PYPI"
fi
python "$script_dir/check-wheel-ttnn-metadata.py" "$@"
"$script_dir/assert-no-gnu-unique.sh" "$expected_wheel"
