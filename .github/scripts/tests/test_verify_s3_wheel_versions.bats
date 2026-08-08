#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/verify-s3-wheel-versions.sh.

load test_helper

VER="99.99.99.dev20260515"

setup() {
    SCRIPT="$SCRIPTS_DIR/verify-s3-wheel-versions.sh"
}

@test "no arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT"
}

@test "too few arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT" pypi "$VER"
}

@test "too many arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT" pypi "$VER" dist extra
}

@test "--no-sim without enough arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT" --no-sim light "$VER"
}

@test "unknown variant -> usage error (exit 2)" {
    dir=$(make_wheel_dir "$(whl "$VER")")
    run -2 "$SCRIPT" unknown "$VER" "$dir"
    assert_output --partial "Unknown wheel variant"
}

@test "pypi mode verifies every wheel against the requested version" {
    dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_sim "$VER")")
    run -0 "$SCRIPT" pypi "$VER" "$dir"
}

@test "bundled mode verifies every wheel against the requested version" {
    dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_sim "$VER")")
    run -0 "$SCRIPT" bundled "$VER" "$dir"
}

@test "light variant accepts cp310/cp312 +light core wheels plus metapackage and sim" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp310 "$VER")" \
        "$(whl_light_core_cp312 "$VER")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -0 "$SCRIPT" light "$VER" "$dir"
}

@test "light variant with --no-sim accepts cp310/cp312 core wheels without sim" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp310 "$VER")" \
        "$(whl_light_core_cp312 "$VER")" \
        "$(whl_light "$VER")")
    run -0 "$SCRIPT" --no-sim light "$VER" "$dir"
}

@test "light variant can verify a single requested Python tag" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp312 "$VER")" \
        "$(whl_light "$VER")")
    run -0 "$SCRIPT" --no-sim --python-tags cp312 light "$VER" "$dir"
}

@test "light variant rejects an unrequested Python tag" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp310 "$VER")" \
        "$(whl_light_core_cp312 "$VER")" \
        "$(whl_light "$VER")")
    run -1 "$SCRIPT" --no-sim --python-tags cp312 light "$VER" "$dir"
    assert_output --partial "Unexpected cp310 manylinux_2_34 light core wheel"
}

@test "light variant rejects unsupported requested Python tag" {
    dir=$(make_wheel_dir "$(whl_light "$VER")")
    run -2 "$SCRIPT" --no-sim --python-tags cp311 light "$VER" "$dir"
    assert_output --partial "Unsupported Python tag: cp311"
}

@test "light variant with --no-sim rejects a sim wheel" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp310 "$VER")" \
        "$(whl_light_core_cp312 "$VER")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" --no-sim light "$VER" "$dir"
    assert_output --partial "Unexpected tt-lang-sim wheel"
}

@test "light variant rejects tt-lang without +light" {
    dir=$(make_wheel_dir \
        "$(whl "$VER")" \
        "$(whl_light_core_cp310 "$VER")" \
        "$(whl_light_core_cp312 "$VER")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" light "$VER" "$dir"
    assert_output --partial "Unexpected tt-lang light wheel filename"
}

@test "light variant requires the tt-lang-light wheel" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp310 "$VER")" \
        "$(whl_light_core_cp312 "$VER")" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" light "$VER" "$dir"
    assert_output --partial "Expected exactly one tt-lang-light metapackage wheel"
}

@test "light variant requires cp310 core wheel" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp312 "$VER")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" light "$VER" "$dir"
    assert_output --partial "Expected exactly one cp310 manylinux_2_34 light core wheel"
}

@test "light variant requires cp312 core wheel" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp310 "$VER")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" light "$VER" "$dir"
    assert_output --partial "Expected exactly one cp312 manylinux_2_34 light core wheel"
}

@test "light variant rejects manylinux_2_35 core wheel" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp310 "$VER")" \
        "$(whl_light_core_tagged "$VER" "cp312-cp312-manylinux_2_35_x86_64")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" light "$VER" "$dir"
    assert_output --partial "Unexpected tt-lang light wheel filename"
}

@test "light variant rejects duplicate metapackage wheel" {
    dir=$(make_wheel_dir \
        "$(whl_light_core_cp310 "$VER")" \
        "$(whl_light_core_cp312 "$VER")" \
        "$(whl_light "$VER")" \
        "tt_lang_light-${VER}-1-py3-none-any.whl" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" light "$VER" "$dir"
    assert_output --partial "Unexpected wheel in light artifact"
}
