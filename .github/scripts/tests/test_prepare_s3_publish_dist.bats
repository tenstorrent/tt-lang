#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/prepare-s3-publish-dist.sh.

load test_helper

VER="99.99.99.dev20260515"

setup() {
    SCRIPT="$SCRIPTS_DIR/prepare-s3-publish-dist.sh"
    PUBLISH_DIR="$BATS_TEST_TMPDIR/publish-dist"
}

@test "no arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT"
}

@test "missing artifact spec -> usage error (exit 2)" {
    run -2 "$SCRIPT" "$VER" "$PUBLISH_DIR"
}

@test "malformed artifact spec -> usage error (exit 2)" {
    run -2 "$SCRIPT" "$VER" "$PUBLISH_DIR" bundled
}

@test "unsafe publish dir -> usage error (exit 2)" {
    dir=$(make_wheel_dir "$(whl "$VER")")
    run -2 "$SCRIPT" "$VER" "." "bundled=$dir"
    assert_output --partial "Unsafe publish directory: ."
}

@test "unknown variant -> usage error (exit 2)" {
    dir=$(make_wheel_dir "$(whl "$VER")")
    run -2 "$SCRIPT" "$VER" "$PUBLISH_DIR" "garbage=$dir"
    assert_output --partial "Unknown wheel variant: garbage"
}

@test ":no-sim on a non-light variant -> usage error (exit 2)" {
    dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_sim "$VER")")
    run -2 "$SCRIPT" "$VER" "$PUBLISH_DIR" "bundled:no-sim=$dir"
    assert_output --partial ":no-sim is only valid for the light variant"
}

@test "missing artifact dir -> error" {
    run -1 "$SCRIPT" "$VER" "$PUBLISH_DIR" bundled="$BATS_TEST_TMPDIR/missing"
    assert_output --partial "Wheel artifact directory not found"
}

@test "bundled artifact is verified and copied" {
    bundled_dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_sim "$VER")")

    run -0 "$SCRIPT" "$VER" "$PUBLISH_DIR" "bundled=$bundled_dir"

    run ls "$PUBLISH_DIR"
    assert_output --partial "$(whl "$VER")"
    assert_output --partial "$(whl_sim "$VER")"
}

@test "combined bundled and light artifacts copy unique wheel set" {
    bundled_dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_sim "$VER")")
    light_dir=$(make_wheel_dir "$(whl "$VER+light")" "$(whl_light "$VER")")

    run -0 "$SCRIPT" \
        "$VER" \
        "$PUBLISH_DIR" \
        "bundled=$bundled_dir" \
        "light:no-sim=$light_dir"

    run ls "$PUBLISH_DIR"
    assert_output --partial "$(whl "$VER")"
    assert_output --partial "$(whl_sim "$VER")"
    assert_output --partial "$(whl "$VER+light")"
    assert_output --partial "$(whl_light "$VER")"
}

@test "light no-sim spec rejects unexpected sim wheel" {
    light_dir=$(make_wheel_dir \
        "$(whl "$VER+light")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")

    run -1 "$SCRIPT" "$VER" "$PUBLISH_DIR" "light:no-sim=$light_dir"

    assert_output --partial "No expected version configured for distribution 'tt_lang_sim'"
}

@test "duplicate wheel filename across artifacts -> error" {
    first_dir=$(make_wheel_dir "$(whl "$VER")")
    second_dir=$(make_wheel_dir "$(whl "$VER")")

    run -1 "$SCRIPT" \
        "$VER" \
        "$PUBLISH_DIR" \
        "bundled=$first_dir" \
        "pypi=$second_dir"

    assert_output --partial "Duplicate wheel filename across S3 publish artifacts"
}
