#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify the wheel versions produced by the S3 PyPI publish workflow. The light
# variant publishes cp310/cp312 tt-lang wheels with a +light local version plus
# the tt-lang-light metapackage; bundled and pypi variants publish all wheels at
# the requested internal version.
#
# Usage: verify-s3-wheel-versions.sh [--no-sim] [--python-tags cp310,cp312] <wheel_variant> <version_override> <dist_dir>

set -eu

usage() {
    echo "Usage: $0 [--no-sim] [--python-tags cp310,cp312] <wheel_variant> <version_override> <dist_dir>" >&2
    exit 2
}

include_sim=1
python_tags=cp310,cp312

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-sim)
            include_sim=0
            shift
            ;;
        --python-tags)
            if [ "$#" -lt 2 ]; then
                usage
            fi
            python_tags="$2"
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

if [ "$#" -ne 3 ]; then
    usage
fi

variant="$1"
version="$2"
dist_dir="$3"
script_dir="$(cd "$(dirname "$0")" && pwd)"

verify_light_wheels() {
    verify_light_version="$1"
    verify_light_dist_dir="$2"
    verify_light_include_sim="$3"
    verify_light_python_tags="$4"
    verify_light_failed=0
    verify_light_seen_cp310=0
    verify_light_seen_cp312=0
    verify_light_metapackage_count=0
    verify_light_sim_count=0
    verify_light_expected_cp310=0
    verify_light_expected_cp312=0
    verify_light_seen_any=0

    if [ -z "$verify_light_python_tags" ]; then
        echo "At least one Python tag is required" >&2
        return 2
    fi

    for verify_light_python_tag in $(printf '%s\n' "$verify_light_python_tags" | tr ',' ' '); do
        case "$verify_light_python_tag" in
            cp310) verify_light_expected_cp310=1 ;;
            cp312) verify_light_expected_cp312=1 ;;
            *)
                echo "Unsupported Python tag: $verify_light_python_tag" >&2
                return 2
                ;;
        esac
    done

    for verify_light_wheel in "$verify_light_dist_dir"/*.whl; do
        if [ ! -e "$verify_light_wheel" ]; then
            continue
        fi
        verify_light_seen_any=1
        verify_light_wheel_name="$(basename "$verify_light_wheel")"
        case "$verify_light_wheel_name" in
            "tt_lang-${verify_light_version}+light-cp310-cp310-manylinux_2_34_x86_64.whl")
                verify_light_seen_cp310=$((verify_light_seen_cp310 + 1))
                ;;
            "tt_lang-${verify_light_version}+light-cp312-cp312-manylinux_2_34_x86_64.whl")
                verify_light_seen_cp312=$((verify_light_seen_cp312 + 1))
                ;;
            "tt_lang_light-${verify_light_version}-py3-none-any.whl")
                verify_light_metapackage_count=$((verify_light_metapackage_count + 1))
                ;;
            "tt_lang_sim-${verify_light_version}-py3-none-any.whl")
                if [ "$verify_light_include_sim" -eq 1 ]; then
                    verify_light_sim_count=$((verify_light_sim_count + 1))
                else
                    echo "Unexpected tt-lang-sim wheel in no-sim light artifact: $verify_light_wheel_name" >&2
                    verify_light_failed=1
                fi
                ;;
            tt_lang-*)
                echo "Unexpected tt-lang light wheel filename: $verify_light_wheel_name" >&2
                verify_light_failed=1
                ;;
            *)
                echo "Unexpected wheel in light artifact: $verify_light_wheel_name" >&2
                verify_light_failed=1
                ;;
        esac
    done

    if [ "$verify_light_seen_any" -eq 0 ]; then
        echo "No wheels found in $verify_light_dist_dir" >&2
        return 1
    fi

    if [ "$verify_light_expected_cp310" -eq 1 ] && [ "$verify_light_seen_cp310" -ne 1 ]; then
        echo "Expected exactly one cp310 manylinux_2_34 light core wheel, found $verify_light_seen_cp310" >&2
        verify_light_failed=1
    fi
    if [ "$verify_light_expected_cp310" -eq 0 ] && [ "$verify_light_seen_cp310" -ne 0 ]; then
        echo "Unexpected cp310 manylinux_2_34 light core wheel, found $verify_light_seen_cp310" >&2
        verify_light_failed=1
    fi
    if [ "$verify_light_expected_cp312" -eq 1 ] && [ "$verify_light_seen_cp312" -ne 1 ]; then
        echo "Expected exactly one cp312 manylinux_2_34 light core wheel, found $verify_light_seen_cp312" >&2
        verify_light_failed=1
    fi
    if [ "$verify_light_expected_cp312" -eq 0 ] && [ "$verify_light_seen_cp312" -ne 0 ]; then
        echo "Unexpected cp312 manylinux_2_34 light core wheel, found $verify_light_seen_cp312" >&2
        verify_light_failed=1
    fi
    if [ "$verify_light_metapackage_count" -ne 1 ]; then
        echo "Expected exactly one tt-lang-light metapackage wheel, found $verify_light_metapackage_count" >&2
        verify_light_failed=1
    fi
    if [ "$verify_light_include_sim" -eq 1 ] && [ "$verify_light_sim_count" -ne 1 ]; then
        echo "Expected exactly one tt-lang-sim wheel, found $verify_light_sim_count" >&2
        verify_light_failed=1
    fi

    return "$verify_light_failed"
}

case "$variant" in
    light)
        verify_light_wheels "$version" "$dist_dir" "$include_sim" "$python_tags"
        ;;
    bundled | pypi)
        "$script_dir/verify-wheel-version.sh" "$version" "$dist_dir"
        ;;
    *)
        echo "Unknown wheel variant: $variant" >&2
        exit 2
        ;;
esac
