#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/publish-s3-summary.sh.

load test_helper

VER="99.99.99.dev20260515"

setup() {
    SCRIPT="$SCRIPTS_DIR/publish-s3-summary.sh"
    SUMMARY_FILE="$BATS_TEST_TMPDIR/summary.md"
    : > "$SUMMARY_FILE"
    unset GITHUB_STEP_SUMMARY
}

@test "no arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT"
}

@test "single argument -> usage error (exit 2)" {
    run -2 "$SCRIPT" pypi
}

@test "--dry-run without enough arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT" --dry-run pypi
}

@test "pypi mode emits single install block to stdout" {
    run -0 "$SCRIPT" pypi "$VER"
    assert_output --partial "### Published wheels"
    assert_output --partial "tt-lang==$VER"
    refute_output --partial "tt-lang-light"
    refute_output --partial "$VER+light"
}

@test "bundled mode emits single install block to stdout" {
    run -0 "$SCRIPT" bundled "$VER"
    assert_output --partial "tt-lang==$VER"
    refute_output --partial "tt-lang-light"
}

@test "external mode emits both light and underlying install blocks" {
    run -0 "$SCRIPT" external "$VER"
    assert_output --partial "tt-lang-light==$VER"
    assert_output --partial "tt-lang==$VER+light"
    assert_output --partial "Light install:"
    assert_output --partial "Underlying no-ttnn tt-lang wheel:"
}

@test "--dry-run marks summary as not uploaded" {
    run -0 "$SCRIPT" --dry-run bundled "$VER"
    assert_output --partial "### Wheel publish dry run"
    assert_output --partial "No wheels were uploaded."
    assert_output --partial "tt-lang==$VER"
    refute_output --partial "### Published wheels"
}

@test "appends to GITHUB_STEP_SUMMARY when set" {
    GITHUB_STEP_SUMMARY="$SUMMARY_FILE" run -0 "$SCRIPT" pypi "$VER"
    # Output to stdout should be empty when redirected.
    assert_output ""
    run cat "$SUMMARY_FILE"
    assert_output --partial "tt-lang==$VER"
}

@test "GITHUB_STEP_SUMMARY appends rather than overwrites" {
    echo "prior content" > "$SUMMARY_FILE"
    GITHUB_STEP_SUMMARY="$SUMMARY_FILE" run -0 "$SCRIPT" pypi "$VER"
    run cat "$SUMMARY_FILE"
    assert_line --index 0 "prior content"
    assert_output --partial "tt-lang==$VER"
}
