#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/require-pypi-ttnn-alignment.sh.

load test_helper

setup() {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
}

run_alignment_check() {
    (
        cd "$REPO"
        .github/scripts/require-pypi-ttnn-alignment.sh 2>&1
    )
}

@test "accepts matching ttnn provenance and tt-metal tag" {
    write_tt_metal_version_file "$REPO/third-party/tt-metal-version" \
        "$TEST_TTNN_PYPI_VERSION" \
        "$TEST_TT_METAL_RC1_TAG" \
        "$TEST_TT_METAL_RC1_TAG"
    commit_all "$REPO" "aligned"

    run run_alignment_check

    assert_success
    assert_output --partial "ok: ttnn==$TEST_TTNN_PYPI_VERSION and tt-lang both use tt-metal release $TEST_TT_METAL_TAG"
}

@test "accepts matching release component with prerelease provenance tag" {
    write_tt_metal_version_file "$REPO/third-party/tt-metal-version" \
        "$TEST_TTNN_PYPI_VERSION" \
        "$TEST_TT_METAL_RC1_TAG" \
        "$TEST_TT_METAL_TAG"
    commit_all "$REPO" "aligned release component"

    run run_alignment_check

    assert_success
    assert_output --partial "ok: ttnn==$TEST_TTNN_PYPI_VERSION and tt-lang both use tt-metal release $TEST_TT_METAL_TAG"
}

@test "rejects mismatched ttnn provenance and tt-metal release component" {
    write_tt_metal_version_file "$REPO/third-party/tt-metal-version" \
        "$TEST_TTNN_PYPI_VERSION" \
        "$TEST_TT_METAL_RC1_TAG" \
        "$TEST_TT_METAL_NEXT_TAG"
    commit_all "$REPO" "mismatched"

    run run_alignment_check

    assert_failure
    assert_output --partial "Public PyPI publish requires ttnn provenance to match the TT_METAL_TAG vX.Y.Z component."
    assert_output --partial "TTNN_PYPI=$TEST_TTNN_PYPI_VERSION was built from TTNN_PYPI_TT_METAL_TAG=$TEST_TT_METAL_RC1_TAG"
    assert_output --partial "TT_METAL_TAG=$TEST_TT_METAL_NEXT_TAG"
}

@test "rejects missing ttnn provenance tag" {
    cat > "$REPO/third-party/tt-metal-version" <<EOF
TTNN_PYPI="$TEST_TTNN_PYPI_VERSION"
TT_METAL_TAG="$TEST_TT_METAL_RC1_TAG"
EOF
    commit_all "$REPO" "missing provenance"

    run run_alignment_check

    assert_failure
    assert_output --partial "TTNN_PYPI_TT_METAL_TAG not set"
}
