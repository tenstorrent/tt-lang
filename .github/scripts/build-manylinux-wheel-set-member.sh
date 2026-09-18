#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Build one ABI member of a manylinux wheel set. The cp312 member also produces
# the ABI-independent simulator wheel and, in external mode, the light
# metapackage.

set -eu

python_tag=""
version=""
ttnn_dep_mode=""
build_sim=true
dist_dir=dist

usage() {
    echo "Usage: $0 --python-tag cp310|cp312 --version <version> --ttnn-dep-mode pypi|external [--build-sim true|false] [--dist-dir <dir>]" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python-tag)
            [ "$#" -ge 2 ] || usage
            python_tag="$2"
            shift 2
            ;;
        --version)
            [ "$#" -ge 2 ] || usage
            version="$2"
            shift 2
            ;;
        --ttnn-dep-mode)
            [ "$#" -ge 2 ] || usage
            ttnn_dep_mode="$2"
            shift 2
            ;;
        --build-sim)
            [ "$#" -ge 2 ] || usage
            build_sim="$2"
            shift 2
            ;;
        --dist-dir)
            [ "$#" -ge 2 ] || usage
            dist_dir="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

case "$python_tag" in
    cp310 | cp312) ;;
    *) usage ;;
esac
case "$ttnn_dep_mode" in
    pypi | external) ;;
    *) usage ;;
esac
case "$build_sim" in
    true | false) ;;
    *) usage ;;
esac
[ -n "$version" ] || usage

script_dir="$(cd "$(dirname "$0")" && pwd)"

"$script_dir/build-manylinux-core-wheel.sh" \
    --python-tag "$python_tag" \
    --version "$version" \
    --ttnn-dep-mode "$ttnn_dep_mode" \
    --dist-dir "$dist_dir"

if [ "$python_tag" != cp312 ]; then
    exit 0
fi

if [ "$build_sim" = true ]; then
    . /opt/ttlang-toolchain/venv/bin/activate
    TTLANG_VERSION_OVERRIDE="$version" \
        python -m pip wheel packaging/sim \
            --wheel-dir="$dist_dir" \
            --no-deps \
            --no-build-isolation

    test_venv="$(mktemp -d /tmp/ttlang-sim-wheel-test.XXXXXX)"
    trap 'rm -rf "$test_venv"' EXIT
    /opt/python/cp312-cp312/bin/python -m venv "$test_venv"
    # shellcheck disable=SC1090
    . "$test_venv/bin/activate"
    pip install \
        --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "$dist_dir"/tt_lang_sim-*.whl
    python "$script_dir/smoke-test-wheel.py"
fi

if [ "$ttnn_dep_mode" = external ]; then
    "$script_dir/build-s3-light-metapackage-wheel.sh" \
        --version "$version" \
        --dist-dir "$dist_dir"
fi
