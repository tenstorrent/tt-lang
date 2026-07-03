#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/resolve-xla-build-inputs.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/resolve-xla-build-inputs.sh"
    GH_OUT="$BATS_TEST_TMPDIR/gh_out"
    : > "$GH_OUT"

    # A stand-in target checkout with stub versions of the two scripts the
    # resolver invokes, so the orchestration is tested without a real git repo.
    TARGET="$BATS_TEST_TMPDIR/target"
    mkdir -p "$TARGET/.github/containers" "$TARGET/.github/scripts"
    cat > "$TARGET/.github/containers/get-version-tag.sh" <<'EOF'
#!/usr/bin/env bash
echo computed-tag
EOF
    cat > "$TARGET/.github/scripts/compute-nightly-version.py" <<'EOF'
print("computed-ver")
EOF
    chmod +x "$TARGET/.github/containers/get-version-tag.sh"
}

@test "computes tag and version from the target checkout" {
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" --target-dir "$TARGET"
    run cat "$GH_OUT"
    assert_line "tag=computed-tag"
    assert_line "version=computed-ver"
}

@test "explicit overrides win and skip the target scripts" {
    # Make the target scripts fail; overrides must mean they are never run.
    echo 'exit 3' >> "$TARGET/.github/containers/get-version-tag.sh"
    echo 'import sys; sys.exit(3)' >> "$TARGET/.github/scripts/compute-nightly-version.py"
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" \
        --target-dir "$TARGET" --docker-tag my-tag --version 9.9.9
    run cat "$GH_OUT"
    assert_line "tag=my-tag"
    assert_line "version=9.9.9"
}

@test "trims surrounding whitespace from overrides" {
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" \
        --target-dir "$TARGET" --docker-tag "  spaced-tag  " --version "  1.2.3  "
    run cat "$GH_OUT"
    assert_line "tag=spaced-tag"
    assert_line "version=1.2.3"
}

@test "resolves missing bare release to unique existing IRD image tag" {
    cat > "$BATS_TEST_TMPDIR/tags" <<'EOF'
v1.1.2-uplift-b4f623c2
EOF
    cat > "$TARGET/.github/containers/get-version-tag.sh" <<'EOF'
#!/usr/bin/env bash
echo v1.1.2
EOF
    chmod +x "$TARGET/.github/containers/get-version-tag.sh"

    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" \
        --target-dir "$TARGET" \
        --resolve-existing-docker-tag \
        --docker-tags-file "$BATS_TEST_TMPDIR/tags"
    run cat "$GH_OUT"
    assert_line "tag=v1.1.2-uplift-b4f623c2"
    assert_line "version=computed-ver"
}

@test "explicit docker tag validates exactly when resolving existing images" {
    cat > "$BATS_TEST_TMPDIR/tags" <<'EOF'
my-tag
v1.1.2-uplift-b4f623c2
EOF

    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" \
        --target-dir "$TARGET" \
        --docker-tag my-tag \
        --resolve-existing-docker-tag \
        --docker-tags-file "$BATS_TEST_TMPDIR/tags"
    run cat "$GH_OUT"
    assert_line "tag=my-tag"
    assert_line "version=computed-ver"
}

@test "missing --target-dir -> exit 2" {
    run "$SCRIPT" --docker-tag t --version v
    assert_equal "$status" 2
}

@test "nonexistent --target-dir -> exit 2" {
    run "$SCRIPT" --target-dir "$BATS_TEST_TMPDIR/nope"
    assert_equal "$status" 2
}

@test "unknown argument -> exit 2" {
    run "$SCRIPT" --target-dir "$TARGET" --bogus
    assert_equal "$status" 2
}
