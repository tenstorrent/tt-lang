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
    cat > "$REPO/third-party/tt-metal-version" <<'EOF'
TTNN_PYPI="0.70.1"
TTNN_PYPI_TT_METAL_TAG="v0.70.1-rc1"
TT_METAL_TAG="v0.70.1-rc1"
EOF
    commit_all "$REPO" "aligned"

    run run_alignment_check

    assert_success
    assert_output --partial "ok: ttnn==0.70.1 and tt-lang both use tt-metal v0.70.1-rc1"
}

@test "rejects mismatched ttnn provenance and tt-metal tag" {
    cat > "$REPO/third-party/tt-metal-version" <<'EOF'
TTNN_PYPI="0.70.1"
TTNN_PYPI_TT_METAL_TAG="v0.70.1-rc1"
TT_METAL_TAG="v0.71.0-rc2"
EOF
    commit_all "$REPO" "mismatched"

    run run_alignment_check

    assert_failure
    assert_output --partial "Public PyPI publish requires ttnn provenance to match TT_METAL_TAG."
    assert_output --partial "TTNN_PYPI=0.70.1 was built from TTNN_PYPI_TT_METAL_TAG=v0.70.1-rc1"
    assert_output --partial "TT_METAL_TAG=v0.71.0-rc2"
}

@test "rejects missing ttnn provenance tag" {
    cat > "$REPO/third-party/tt-metal-version" <<'EOF'
TTNN_PYPI="0.70.1"
TT_METAL_TAG="v0.70.1-rc1"
EOF
    commit_all "$REPO" "missing provenance"

    run run_alignment_check

    assert_failure
    assert_output --partial "TTNN_PYPI_TT_METAL_TAG not set"
}
