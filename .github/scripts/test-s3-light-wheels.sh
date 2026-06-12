#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Install-test S3 tt-lang-light wheels under the manylinux_2_34 builder images.

set -eu

VERSION=""
PYTHON_TAGS=cp310,cp312
DOCKER_TAG=""
DIST_DIR=dist

usage() {
    cat >&2 <<'EOF'
Usage: test-s3-light-wheels.sh --version <version> [options]

Options:
  --python-tags cp310,cp312  Python ABI tags to test. Default: cp310,cp312.
  --docker-tag <tag>         Builder image tag. Default: current Docker tag.
  --dist-dir <dir>           Wheel directory. Default: dist.
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
        --python-tags)
            if [ "$#" -lt 2 ]; then usage; fi
            PYTHON_TAGS="$2"
            shift 2
            ;;
        --docker-tag)
            if [ "$#" -lt 2 ]; then usage; fi
            DOCKER_TAG="$2"
            shift 2
            ;;
        --dist-dir)
            if [ "$#" -lt 2 ]; then usage; fi
            DIST_DIR="$2"
            shift 2
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
# shellcheck source=lib/docker-image-utils.sh
. "$repo_root/.github/scripts/lib/docker-image-utils.sh"
docker_tag="${DOCKER_TAG:-$("$repo_root/.github/containers/get-version-tag.sh")}"
dist_dir="$(cd "$DIST_DIR" && pwd)"

if ! python_tags="$(ttlang_python_tags "$PYTHON_TAGS")"; then
    exit 2
fi

for python_tag in $python_tags; do
    image="$(ttlang_wheel_builder_image "$python_tag" "$docker_tag")"
    echo "Install-testing tt-lang-light on $python_tag with $image"
    ttlang_docker run --rm \
        -v "$repo_root:/workspace" \
        -v "$dist_dir:/dist" \
        -w /workspace \
        -e "PYTHON_TAG=$python_tag" \
        -e "TTLANG_VERSION=$VERSION" \
        "$image" \
        bash -euxo pipefail -c '
          python_dir="/opt/python/${PYTHON_TAG}-${PYTHON_TAG}"
          test -x "${python_dir}/bin/python"
          rm -rf /tmp/test-venv-s3-light
          "${python_dir}/bin/python" -m venv /tmp/test-venv-s3-light
          . /tmp/test-venv-s3-light/bin/activate
          python -m pip install --upgrade pip
          python -m pip install \
            --no-cache-dir \
            --find-links=/dist \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            "tt-lang-light==${TTLANG_VERSION}"
          python .github/scripts/check-installed-ttnn.py --mode external
          python .github/scripts/smoke-test-wheel.py
        '
done
