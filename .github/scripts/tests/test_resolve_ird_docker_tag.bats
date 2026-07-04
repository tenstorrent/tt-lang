#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/resolve-ird-docker-tag.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/resolve-ird-docker-tag.sh"
    TAGS="$BATS_TEST_TMPDIR/tags"
    cat > "$TAGS" <<'EOF'
v1.1.2-uplift-b4f623c2
v1.1.3
v1.1.4-6746fe2e
EOF
}

@test "returns exact existing tag" {
    run -0 "$SCRIPT" --candidate v1.1.3 --tags-file "$TAGS"
    assert_output "v1.1.3"
}

@test "resolves missing bare release to unique version-prefixed tag" {
    run -0 "$SCRIPT" \
        --candidate v1.1.2 \
        --tags-file "$TAGS" \
        --allow-version-prefix-fallback
    assert_line "Exact IRD image tag v1.1.2 is missing; using v1.1.2-uplift-b4f623c2"
    assert_line "v1.1.2-uplift-b4f623c2"
}

@test "does not fallback when exact explicit tag is missing" {
    run "$SCRIPT" --candidate v1.1.2 --tags-file "$TAGS"
    assert_equal "$status" 1
    assert_output --partial "IRD builder image tag does not exist: v1.1.2"
}

@test "does not fallback for missing non-release tag" {
    run "$SCRIPT" \
        --candidate v1.1.4-deadbeef \
        --tags-file "$TAGS" \
        --allow-version-prefix-fallback
    assert_equal "$status" 1
    assert_output --partial "IRD builder image tag does not exist: v1.1.4-deadbeef"
}

@test "ambiguous version-prefixed matches fail" {
    cat >> "$TAGS" <<'EOF'
v1.1.2-uplift-cafebabe
EOF
    run "$SCRIPT" \
        --candidate v1.1.2 \
        --tags-file "$TAGS" \
        --allow-version-prefix-fallback
    assert_equal "$status" 1
    assert_output --partial "multiple IRD image tags match v1.1.2-*"
    assert_output --partial "v1.1.2-uplift-b4f623c2"
    assert_output --partial "v1.1.2-uplift-cafebabe"
}

@test "tolerates trailing whitespace and CRLF in the tags file" {
    printf 'v1.1.3  \r\n  v1.1.4-6746fe2e\r\n' > "$TAGS"
    run -0 "$SCRIPT" --candidate v1.1.3 --tags-file "$TAGS"
    assert_output "v1.1.3"
}

@test "bare release does not match a longer patch sibling" {
    cat > "$TAGS" <<'EOF'
v1.1.20-uplift-aaaa
v1.1.3
EOF
    run "$SCRIPT" --candidate v1.1.2 --tags-file "$TAGS" --allow-version-prefix-fallback
    assert_equal "$status" 1
    assert_output --partial "no existing IRD image tag matches v1.1.2 or v1.1.2-*"
}

@test "missing candidate -> exit 2" {
    run "$SCRIPT" --tags-file "$TAGS"
    assert_equal "$status" 2
}

@test "unknown argument -> exit 2" {
    run "$SCRIPT" --candidate v1.1.3 --tags-file "$TAGS" --bogus
    assert_equal "$status" 2
}
