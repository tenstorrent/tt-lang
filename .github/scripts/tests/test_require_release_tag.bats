#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/require-release-tag.sh — refusal of non-tag refs
# and PEP 440 normalization of accepted forms.

load test_helper

BASE="99.99.99"
REF="refs/tags/v${BASE}"

setup() {
    SCRIPT="$SCRIPTS_DIR/require-release-tag.sh"
    GH_OUT="$BATS_TEST_TMPDIR/gh_out"
    : > "$GH_OUT"
    export GITHUB_OUTPUT="$GH_OUT"
}

# --- Rejection cases ---

@test "rejects branch ref" {
    GITHUB_REF="refs/heads/main" run "$SCRIPT"
    assert_failure
}

@test "rejects empty ref" {
    GITHUB_REF="" run "$SCRIPT"
    assert_failure
}

@test "rejects non-version tag" {
    GITHUB_REF="refs/tags/somefeature" run "$SCRIPT"
    assert_failure
}

@test "rejects tag without leading v" {
    GITHUB_REF="refs/tags/1.0.0" run "$SCRIPT"
    assert_failure
}

@test "rejects malformed PEP 440 segment with a clean message" {
    GITHUB_REF="${REF}-foobar" run --separate-stderr "$SCRIPT"
    assert_failure
    [[ "$stderr" == *"not a valid PEP 440 version"* ]] \
        || fail "stderr missing PEP 440 message: $stderr"
    # Confirm we did NOT emit a Python traceback.
    [[ "$stderr" != *"Traceback"* ]] \
        || fail "stderr contains a Python traceback: $stderr"
}

# --- PEP 440 normalization cases ---

@test "final release: vX.Y.Z -> X.Y.Z" {
    GITHUB_REF="$REF" run -0 "$SCRIPT"
    assert_output "$BASE"
}

@test "patch release" {
    GITHUB_REF="refs/tags/v99.99.3" run -0 "$SCRIPT"
    assert_output "99.99.3"
}

@test "dev pre-release (tt-metal style)" {
    GITHUB_REF="${REF}-dev20260515" run -0 "$SCRIPT"
    assert_output "${BASE}.dev20260515"
}

@test "rc pre-release" {
    GITHUB_REF="${REF}-rc1" run -0 "$SCRIPT"
    assert_output "${BASE}rc1"
}

@test "alpha pre-release" {
    GITHUB_REF="${REF}-alpha3" run -0 "$SCRIPT"
    assert_output "${BASE}a3"
}

@test "beta pre-release" {
    GITHUB_REF="${REF}-beta2" run -0 "$SCRIPT"
    assert_output "${BASE}b2"
}

@test "post release" {
    GITHUB_REF="${REF}-post1" run -0 "$SCRIPT"
    assert_output "${BASE}.post1"
}

@test "local version label (+uplift)" {
    GITHUB_REF="${REF}+uplift" run -0 "$SCRIPT"
    assert_output "${BASE}+uplift"
}

@test "dev + local combined" {
    GITHUB_REF="${REF}-dev20260515+ci123" run -0 "$SCRIPT"
    assert_output "${BASE}.dev20260515+ci123"
}

# PEP 440 specifies that pre-release segment markers are case-insensitive and
# normalize to lowercase. Pin this so a future change that swaps `packaging`
# for a hand-rolled regex doesn't silently regress case folding.

@test "case-folded RC normalizes to lowercase rc" {
    GITHUB_REF="${REF}-RC1" run -0 "$SCRIPT"
    assert_output "${BASE}rc1"
}

@test "case-folded DEV normalizes to lowercase dev" {
    GITHUB_REF="${REF}-DEV20260515" run -0 "$SCRIPT"
    assert_output "${BASE}.dev20260515"
}

@test "case-folded Alpha normalizes to a" {
    GITHUB_REF="${REF}-Alpha3" run -0 "$SCRIPT"
    assert_output "${BASE}a3"
}

@test "case-folded BETA normalizes to b" {
    GITHUB_REF="${REF}-BETA2" run -0 "$SCRIPT"
    assert_output "${BASE}b2"
}

@test "case-folded POST normalizes to post" {
    GITHUB_REF="${REF}-POST1" run -0 "$SCRIPT"
    assert_output "${BASE}.post1"
}

# --- GITHUB_OUTPUT writes ---

@test "writes tag_version to GITHUB_OUTPUT" {
    GITHUB_REF="${REF}-rc1" run -0 "$SCRIPT"
    run -0 cat "$GH_OUT"
    assert_output "tag_version=${BASE}rc1"
}

@test "stdout matches GITHUB_OUTPUT line" {
    GITHUB_REF="${REF}-dev20260515" run -0 "$SCRIPT"
    stdout="$output"
    out_line=$(grep '^tag_version=' "$GH_OUT" | sed 's/^tag_version=//')
    assert_equal "$stdout" "$out_line"
}
