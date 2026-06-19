#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build and test the S3 tt-lang-light wheel set locally using the same
# manylinux_2_34 builder images as CI.

set -eu

VERSION=""
PYTHON_TAGS=cp310,cp312
IMAGE_TAG=""
OUTPUT_DIR=/tmp/ttlang-s3-light-wheels
BUILD_IMAGES=false
DOCKER_USER="$(id -u):$(id -g)"

usage() {
    cat >&2 <<'EOF'
Usage: build-s3-light-wheels-local.sh --version <pep440-version> [options]

Options:
  --python-tags cp310,cp312  Python ABI tags to build. Default: cp310,cp312.
  --image-tag <tag>          Builder image tag. Default: current Docker tag.
  --output-dir <dir>         Final wheel output directory. Default: /tmp/ttlang-s3-light-wheels.
  --build-images             Build local manylinux builder images first.
EOF
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            VERSION="$2"
            shift 2
            ;;
        --python-tags)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            PYTHON_TAGS="$2"
            shift 2
            ;;
        --image-tag)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            IMAGE_TAG="$2"
            shift 2
            ;;
        --output-dir)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --build-images)
            BUILD_IMAGES=true
            shift
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$VERSION" ]; then
    usage
fi

case "$OUTPUT_DIR" in
    "" | "." | "/")
        echo "Unsafe output directory: $OUTPUT_DIR" >&2
        exit 2
        ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
# shellcheck source=.github/scripts/lib/docker-image-utils.sh
. "$repo_root/.github/scripts/lib/docker-image-utils.sh"
image_tag="${IMAGE_TAG:-$("$repo_root/.github/containers/get-version-tag.sh")}"
python_tags_csv="$PYTHON_TAGS"
if ! python_tags="$(ttlang_python_tags "$PYTHON_TAGS")"; then
    exit 2
fi
metapackage_python_tag="${PYTHON_TAGS%%,*}"
for python_tag in $python_tags; do
    if [ "$python_tag" = "cp312" ]; then
        metapackage_python_tag=cp312
        break
    fi
done

mkdir -p "$OUTPUT_DIR"
output_dir="$(cd "$OUTPUT_DIR" && pwd)"
rm -rf "$output_dir/dist" "$output_dir"/dist-*
mkdir -p "$output_dir/dist"

if [ "$BUILD_IMAGES" = true ]; then
    "$repo_root/.github/containers/build-wheel-manylinux-images.sh" \
        --no-push \
        --image-tag "$image_tag" \
        --python-tags "$python_tags_csv"
fi

for python_tag in $python_tags; do
    image="$(ttlang_wheel_builder_image "$python_tag" "$image_tag")"
    echo "=== Building S3 light core wheel for $python_tag with $image ==="
    # Per-ABI dist dir: build-s3-light-core-wheel.sh runs
    # check-wheel-ttnn-metadata.py, which requires exactly one tt_lang wheel in
    # the dir. CI isolates this per matrix leg; mirror that here, then collect.
    mkdir -p "$output_dir/dist-$python_tag"
    ttlang_docker run --rm \
        --user "$DOCKER_USER" \
        -v "$repo_root:/workspace" \
        -v "$output_dir:/out" \
        -w /workspace \
        -e HOME=/tmp \
        -e TTLANG_EXTERNAL_TT_METAL_DIR=/opt/ttlang-toolchain/tt-metal \
        -e TTLANG_PYTHON_VENV=/opt/ttlang-toolchain/venv \
        "$image" \
        .github/scripts/build-s3-light-core-wheel.sh \
            --python-tag "$python_tag" \
            --version "$VERSION" \
            --dist-dir "/out/dist-$python_tag"
    mv "$output_dir/dist-$python_tag"/*.whl "$output_dir/dist/"
    rmdir "$output_dir/dist-$python_tag"
done

image="$(ttlang_wheel_builder_image "$metapackage_python_tag" "$image_tag")"
echo "=== Building tt-lang-light metapackage with $image ==="
ttlang_docker run --rm \
    --user "$DOCKER_USER" \
    -v "$repo_root:/workspace" \
    -v "$output_dir:/out" \
    -w /workspace \
    -e HOME=/tmp \
    "$image" \
    .github/scripts/build-s3-light-metapackage-wheel.sh \
        --version "$VERSION" \
        --dist-dir /out/dist

"$repo_root/.github/scripts/verify-s3-wheel-versions.sh" \
    --no-sim \
    --python-tags "$python_tags_csv" \
    light \
    "$VERSION" \
    "$output_dir/dist"

"$repo_root/.github/scripts/test-s3-light-wheels.sh" \
    --version "$VERSION" \
    --python-tags "$python_tags_csv" \
    --docker-tag "$image_tag" \
    --dist-dir "$output_dir/dist"

echo "=== S3 light wheels ready ==="
ls -lh "$output_dir/dist"
