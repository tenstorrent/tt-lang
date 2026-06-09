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

@test "light variant accepts +light tt-lang plus normal light and sim wheels" {
    dir=$(make_wheel_dir \
        "$(whl "$VER+light")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -0 "$SCRIPT" light "$VER" "$dir"
}

@test "light variant with --no-sim accepts light wheels without sim" {
    dir=$(make_wheel_dir \
        "$(whl "$VER+light")" \
        "$(whl_light "$VER")")
    run -0 "$SCRIPT" --no-sim light "$VER" "$dir"
}

@test "light variant with --no-sim rejects a sim wheel" {
    dir=$(make_wheel_dir \
        "$(whl "$VER+light")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" --no-sim light "$VER" "$dir"
    assert_output --partial "No expected version configured for distribution 'tt_lang_sim'"
}

@test "light variant rejects tt-lang without +light" {
    dir=$(make_wheel_dir \
        "$(whl "$VER")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" light "$VER" "$dir"
    assert_output --partial "does not match expected '$VER+light'"
}

@test "light variant requires the tt-lang-light wheel" {
    dir=$(make_wheel_dir "$(whl "$VER+light")" "$(whl_sim "$VER")")
    run -1 "$SCRIPT" light "$VER" "$dir"
    assert_output --partial "No wheel found for expected distribution 'tt_lang_light'"
}
