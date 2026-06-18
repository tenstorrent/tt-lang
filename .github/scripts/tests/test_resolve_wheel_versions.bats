#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/resolve-wheel-versions.sh.

load test_helper

VER="99.99.99.dev20260527"

setup() {
    SCRIPT="$SCRIPTS_DIR/resolve-wheel-versions.sh"
    GITHUB_OUTPUT_FILE="$BATS_TEST_TMPDIR/github_output"
    : > "$GITHUB_OUTPUT_FILE"
    export GITHUB_OUTPUT="$GITHUB_OUTPUT_FILE"
    unset TTNN_DEP_MODE
    unset VERSION_OVERRIDE
}

# Read one `key=value` line from the captured GITHUB_OUTPUT file. Echoes the
# value with no decoration; fails the assertion if the key is missing.
output_value() {
    local key="$1"
    grep "^${key}=" "$GITHUB_OUTPUT_FILE" | sed "s/^${key}=//"
}

@test "missing TTNN_DEP_MODE -> error" {
    run -1 "$SCRIPT"
    assert_output --partial "TTNN_DEP_MODE is required"
}

@test "unknown TTNN_DEP_MODE -> error (exit 1)" {
    TTNN_DEP_MODE=garbage VERSION_OVERRIDE="$VER" run -1 "$SCRIPT"
    assert_output --partial "unknown ttnn_dep_mode: garbage"
}

@test "external without VERSION_OVERRIDE -> error (exit 1)" {
    TTNN_DEP_MODE=external VERSION_OVERRIDE="" run -1 "$SCRIPT"
    assert_output --partial "ttnn_dep_mode=external requires version_override"
}

@test "external with VERSION_OVERRIDE -> both versions get +light" {
    TTNN_DEP_MODE=external VERSION_OVERRIDE="$VER" run -0 "$SCRIPT"
    assert_equal "$(output_value core_version)" "${VER}+light"
    assert_equal "$(output_value light_version)" "${VER}+light"
}

@test "bundled with VERSION_OVERRIDE -> core only, no light" {
    TTNN_DEP_MODE=bundled VERSION_OVERRIDE="$VER" run -0 "$SCRIPT"
    assert_equal "$(output_value core_version)" "$VER"
    assert_equal "$(output_value light_version)" ""
}

@test "pypi with VERSION_OVERRIDE -> core only, no light" {
    TTNN_DEP_MODE=pypi VERSION_OVERRIDE="$VER" run -0 "$SCRIPT"
    assert_equal "$(output_value core_version)" "$VER"
    assert_equal "$(output_value light_version)" ""
}

@test "pypi with empty VERSION_OVERRIDE -> empty core_version" {
    TTNN_DEP_MODE=pypi VERSION_OVERRIDE="" run -0 "$SCRIPT"
    assert_equal "$(output_value core_version)" ""
    assert_equal "$(output_value light_version)" ""
}

@test "bundled with empty VERSION_OVERRIDE -> empty core_version" {
    TTNN_DEP_MODE=bundled VERSION_OVERRIDE="" run -0 "$SCRIPT"
    assert_equal "$(output_value core_version)" ""
    assert_equal "$(output_value light_version)" ""
}

@test "GITHUB_OUTPUT unset -> writes to stdout" {
    unset GITHUB_OUTPUT
    TTNN_DEP_MODE=external VERSION_OVERRIDE="$VER" run -0 "$SCRIPT"
    assert_output --partial "core_version=${VER}+light"
    assert_output --partial "light_version=${VER}+light"
}

@test "appends rather than overwrites GITHUB_OUTPUT" {
    echo "prior=line" > "$GITHUB_OUTPUT_FILE"
    TTNN_DEP_MODE=pypi VERSION_OVERRIDE="$VER" run -0 "$SCRIPT"
    run cat "$GITHUB_OUTPUT_FILE"
    assert_line --index 0 "prior=line"
    assert_output --partial "core_version=$VER"
}
