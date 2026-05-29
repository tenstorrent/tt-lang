#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/verify-s3-wheel-versions.sh.

load test_helper

VER="99.99.99.dev20260515"
PYTAG="cp312-cp312-linux_x86_64"

whl()       { printf 'tt_lang-%s-%s.whl' "$1" "$PYTAG"; }
whl_sim()   { printf 'tt_lang_sim-%s-py3-none-any.whl' "$1"; }
whl_light() { printf 'tt_lang_light-%s-py3-none-any.whl' "$1"; }

make_wheel_dir() {
    local dir
    dir=$(mktemp -d "$BATS_TEST_TMPDIR/wheels.XXXXXX")
    for name in "$@"; do
        : > "$dir/$name"
    done
    echo "$dir"
}

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

@test "unknown mode -> usage error (exit 2)" {
    dir=$(make_wheel_dir "$(whl "$VER")")
    run -2 "$SCRIPT" unknown "$VER" "$dir"
    assert_output --partial "Unknown ttnn dependency mode"
}

@test "pypi mode verifies every wheel against the requested version" {
    dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_sim "$VER")")
    run -0 "$SCRIPT" pypi "$VER" "$dir"
}

@test "bundled mode verifies every wheel against the requested version" {
    dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_sim "$VER")")
    run -0 "$SCRIPT" bundled "$VER" "$dir"
}

@test "external mode accepts +light tt-lang plus normal light and sim wheels" {
    dir=$(make_wheel_dir \
        "$(whl "$VER+light")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -0 "$SCRIPT" external "$VER" "$dir"
}

@test "external mode rejects tt-lang without +light" {
    dir=$(make_wheel_dir \
        "$(whl "$VER")" \
        "$(whl_light "$VER")" \
        "$(whl_sim "$VER")")
    run -1 "$SCRIPT" external "$VER" "$dir"
    assert_output --partial "does not match expected '$VER+light'"
}

@test "external mode requires the tt-lang-light wheel" {
    dir=$(make_wheel_dir "$(whl "$VER+light")" "$(whl_sim "$VER")")
    run -1 "$SCRIPT" external "$VER" "$dir"
    assert_output --partial "No wheel found for expected distribution 'tt_lang_light'"
}
