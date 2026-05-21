#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/check-tt-metal-version.sh.
#
# Focus: format parsing and validation. The script also resolves a tag to
# a SHA via `git ls-remote` and (in --update mode) re-clones the submodule;
# those paths require network access and are not covered here.

load test_helper

setup() {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
}

# Run the script in verify-only mode, replacing the git ls-remote call with
# a stub by injecting a wrapper for `git`. Returns the script's stdout+stderr.
run_check_no_network() {
    # The script always invokes `git ls-remote` to resolve the tag; without
    # network we cannot exercise the gitlink comparison path. Tests here
    # only need to confirm that the FILE PARSING / VALIDATION step runs
    # before any network call. We force ls-remote to fail and assert on
    # the error message that surfaces.
    (
        cd "$REPO"
        # Use a wrapper PATH directory that shadows git only when called
        # with the ls-remote subcommand from this script.
        PATH="$BATS_TEST_TMPDIR/bin:$PATH"
        mkdir -p "$BATS_TEST_TMPDIR/bin"
        cat > "$BATS_TEST_TMPDIR/bin/git" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "ls-remote" ]]; then
    # Pretend the tag does not exist on the remote.
    exit 0
fi
exec /usr/bin/git "$@"
EOF
        chmod +x "$BATS_TEST_TMPDIR/bin/git"
        .github/scripts/check-tt-metal-version.sh 2>&1
    )
}

@test "rejects missing TT_METAL_TAG" {
    cat > "$REPO/third-party/tt-metal-version" <<'EOF'
TTNN_PYPI="0.69.0"
EOF
    commit_all "$REPO" "missing tag"
    run run_check_no_network
    assert_failure
    assert_output --partial "TT_METAL_TAG not set"
}

@test "rejects missing TTNN_PYPI" {
    cat > "$REPO/third-party/tt-metal-version" <<'EOF'
TT_METAL_TAG="v0.69.0"
EOF
    commit_all "$REPO" "missing pypi"
    run run_check_no_network
    assert_failure
    assert_output --partial "TTNN_PYPI not set"
}

@test "rejects malformed TT_METAL_TAG (no leading v)" {
    cat > "$REPO/third-party/tt-metal-version" <<'EOF'
TTNN_PYPI="0.69.0"
TT_METAL_TAG="0.69.0"
EOF
    commit_all "$REPO" "bad tag"
    run run_check_no_network
    assert_failure
    assert_output --partial "does not look like vX.Y.Z"
}

@test "accepts valid two-variable format" {
    cat > "$REPO/third-party/tt-metal-version" <<'EOF'
TTNN_PYPI="0.70.1"
TT_METAL_TAG="v0.70.1-rc1"
EOF
    commit_all "$REPO" "valid"
    run run_check_no_network
    # ls-remote stub returns empty -> "tt-metal has no release tag" surfaces
    # AFTER parsing succeeded. That confirms parsing passed.
    assert_failure
    assert_output --partial "tt-metal has no release tag"
}

@test "ignores '#' comment lines and blank lines" {
    cat > "$REPO/third-party/tt-metal-version" <<'EOF'
# leading comment about the file
# describing fields

TTNN_PYPI="0.70.1"
# inter-variable comment
TT_METAL_TAG="v0.70.1-rc1"

# trailing comment
EOF
    commit_all "$REPO" "comments"
    run run_check_no_network
    # Parsing succeeded -> we get past validation and into ls-remote.
    assert_output --partial "tt-metal has no release tag"
}

@test "accepts rc-suffixed tag (vX.Y.Z-rcN)" {
    cat > "$REPO/third-party/tt-metal-version" <<'EOF'
TTNN_PYPI="0.70.1"
TT_METAL_TAG="v0.70.1-rc2"
EOF
    commit_all "$REPO" "rc tag"
    run run_check_no_network
    # Regex ^vX.Y.Z accepts the prefix; -rcN passes through to ls-remote.
    assert_output --partial "tt-metal has no release tag"
}
