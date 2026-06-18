#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/verify-wheel-version.sh.

load test_helper

# Synthetic version under test, chosen well outside any real release range so
# the literals can never be confused with a production version.
VER="99.99.99"
WRONG_VER="99.99.98"

setup() {
    SCRIPT="$SCRIPTS_DIR/verify-wheel-version.sh"
}

@test "no args -> usage error (exit 2)" {
    run -2 "$SCRIPT"
}

@test "one arg -> usage error (exit 2)" {
    run -2 "$SCRIPT" "$VER"
}

@test "empty wheel directory -> error (exit 1)" {
    dir=$(make_wheel_dir)
    run -1 "$SCRIPT" "$VER" "$dir"
}

@test "single matching wheel passes" {
    dir=$(make_wheel_dir "$(whl "$VER")")
    run -0 "$SCRIPT" "$VER" "$dir"
}

@test "multiple matching wheels pass" {
    dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_sim "$VER")")
    run -0 "$SCRIPT" "$VER" "$dir"
}

@test "single mismatched wheel fails" {
    dir=$(make_wheel_dir "$(whl "$WRONG_VER")")
    run -1 "$SCRIPT" "$VER" "$dir"
}

@test "one match + one mismatch -> fails" {
    dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_sim "$WRONG_VER")")
    run -1 "$SCRIPT" "$VER" "$dir"
}

@test "PEP 440 form: dev YYYYMMDD matches" {
    v="${VER}.dev20260515"
    dir=$(make_wheel_dir "$(whl "$v")")
    run -0 "$SCRIPT" "$v" "$dir"
}

@test "PEP 440 form: rc1 matches" {
    v="${VER}rc1"
    dir=$(make_wheel_dir "$(whl "$v")")
    run -0 "$SCRIPT" "$v" "$dir"
}

@test "PEP 440 form: +uplift local label matches" {
    v="${VER}+uplift"
    dir=$(make_wheel_dir "$(whl "$v")")
    run -0 "$SCRIPT" "$v" "$dir"
}

@test "PEP 440 form: post1 matches" {
    v="${VER}.post1"
    dir=$(make_wheel_dir "$(whl "$v")")
    run -0 "$SCRIPT" "$v" "$dir"
}

# PEP 427 wheel filenames may carry an optional build-number field between
# the version and the python tag: {name}-{version}-{build}-{python}-{abi}-{plat}.whl.
# The verifier extracts field 2 (version); field 3 (build number) must not
# be mistaken for the version.
@test "PEP 427 build-number suffix: version still extracted from field 2" {
    dir=$(make_wheel_dir "$(whl_build "$VER" 1)")
    run -0 "$SCRIPT" "$VER" "$dir"
}

@test "PEP 427 build-number suffix: mismatch on version (not build-number) still fails" {
    dir=$(make_wheel_dir "$(whl_build "$WRONG_VER" 1)")
    run -1 "$SCRIPT" "$VER" "$dir"
}

@test "PEP 427 build-number suffix: expected version must not match build-number field" {
    # If the verifier mistakenly compared field 3, this wheel's '7' would match
    # expected '7'. Field 2 is the real version ($VER), so this must NOT match.
    dir=$(make_wheel_dir "$(whl_build "$VER" 7)")
    run -1 "$SCRIPT" "7" "$dir"
}

@test "--expect validates per-distribution versions" {
    light_core="${VER}+light"
    dir=$(make_wheel_dir "$(whl "$light_core")" "$(whl_light "$VER")" "$(whl_sim "$VER")")

    run -0 "$SCRIPT" \
        --expect "tt_lang=$light_core" \
        --expect "tt_lang_light=$VER" \
        --expect "tt_lang_sim=$VER" \
        "$dir"
}

@test "--expect fails on a mismatched distribution version" {
    light_core="${VER}+light"
    dir=$(make_wheel_dir "$(whl "$VER")" "$(whl_light "$VER")")

    run -1 "$SCRIPT" \
        --expect "tt_lang=$light_core" \
        --expect "tt_lang_light=$VER" \
        "$dir"
}

@test "--expect fails on unexpected wheel distribution" {
    dir=$(make_wheel_dir "$(whl "$VER")")

    run -1 "$SCRIPT" --expect "tt_lang_light=$VER" "$dir"
}

@test "--expect fails when an expected distribution is missing" {
    dir=$(make_wheel_dir "$(whl "$VER")")

    run -1 "$SCRIPT" \
        --expect "tt_lang=$VER" \
        --expect "tt_lang_light=$VER" \
        "$dir"
}

@test "--expect requires distribution=version syntax" {
    dir=$(make_wheel_dir "$(whl "$VER")")

    run -2 "$SCRIPT" --expect "$VER" "$dir"
}
