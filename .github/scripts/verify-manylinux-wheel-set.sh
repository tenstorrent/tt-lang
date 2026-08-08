#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Verify the complete cp310/cp312 manylinux_2_34 wheel set produced by the
# shared PyPI/light build workflow.

set -eu

build_sim=true
ttnn_dep_mode=""

usage() {
    echo "Usage: $0 --ttnn-dep-mode pypi|external [--build-sim true|false] <version> <dist-dir>" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
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
        --*)
            usage
            ;;
        *)
            break
            ;;
    esac
done

[ "$#" -eq 2 ] || usage
case "$ttnn_dep_mode" in
    pypi | external) ;;
    *) usage ;;
esac
case "$build_sim" in
    true | false) ;;
    *) usage ;;
esac

version="$1"
dist_dir="$2"
script_dir="$(cd "$(dirname "$0")" && pwd)"

version_output="$(mktemp)"
trap 'rm -f "$version_output"' EXIT
TTNN_DEP_MODE="$ttnn_dep_mode" \
VERSION_OVERRIDE="$version" \
GITHUB_OUTPUT="$version_output" \
    "$script_dir/resolve-wheel-versions.sh"
core_version="$(sed -n 's/^core_version=//p' "$version_output")"

expected_files="
tt_lang-${core_version}-cp310-cp310-manylinux_2_34_x86_64.whl
tt_lang-${core_version}-cp312-cp312-manylinux_2_34_x86_64.whl
"
if [ "$ttnn_dep_mode" = external ]; then
    expected_files="${expected_files}tt_lang_light-${version}-py3-none-any.whl
"
fi
if [ "$build_sim" = true ]; then
    expected_files="${expected_files}tt_lang_sim-${version}-py3-none-any.whl
"
fi

failed=0
seen=0
for wheel in "$dist_dir"/*.whl; do
    if [ ! -e "$wheel" ]; then
        continue
    fi
    seen=$((seen + 1))
    wheel_name="$(basename "$wheel")"
    if ! printf '%s' "$expected_files" | grep -Fxq "$wheel_name"; then
        echo "Unexpected manylinux wheel: $wheel_name" >&2
        failed=1
    fi
done

if [ "$seen" -eq 0 ]; then
    echo "No wheels found in $dist_dir" >&2
    exit 1
fi

printf '%s' "$expected_files" | while IFS= read -r expected_file; do
    [ -n "$expected_file" ] || continue
    if [ ! -f "$dist_dir/$expected_file" ]; then
        echo "Expected manylinux wheel was not produced: $expected_file" >&2
        exit 1
    fi
done || failed=1

exit "$failed"
