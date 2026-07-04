#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/record-ttmetal-miss.sh.

load test_helper

FULL_SHA="13adda80fef07a3c5d6f2f8b9a0c1d2e3f405162"
SHORT_SHA="13adda8"
HEAD_SHA="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

# Install a fake `aws` that records its argv to $FAKE_AWS_ARGS and its stdin to
# $FAKE_AWS_STDIN. Echoes the bindir.
make_aws_mock() {
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
cat >> "$FAKE_AWS_STDIN"
exit 0
EOF
    chmod +x "$bindir/aws"
    echo "$bindir"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/record-ttmetal-miss.sh"
    FAKE_AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    FAKE_AWS_STDIN="$BATS_TEST_TMPDIR/aws_stdin"
    : > "$FAKE_AWS_ARGS"; : > "$FAKE_AWS_STDIN"
    export FAKE_AWS_ARGS FAKE_AWS_STDIN
    BINDIR=$(make_aws_mock)
    export PATH="$BINDIR:$PATH"
}

@test "missing required args -> usage error (exit 2)" {
    run -2 "$SCRIPT" --ttmetal-sha "$FULL_SHA"
}

@test "writes attempt.json under tt-lang/<ttmetal7>" {
    run -0 "$SCRIPT" --ttmetal-sha "$FULL_SHA" --ttlang-head "$HEAD_SHA" \
        --max-age-days 14 --date "2026-06-30T00:00:00Z"

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 cp - s3://tenstorrent-pypi/tt-lang/$SHORT_SHA/attempt.json"
    assert_output --partial "--content-type application/json"
}

@test "marker records the tt-metal sha, tt-lang head, and result" {
    run -0 "$SCRIPT" --ttmetal-sha "$FULL_SHA" --ttlang-head "$HEAD_SHA" \
        --max-age-days 14 --date "2026-06-30T00:00:00Z"

    run cat "$FAKE_AWS_STDIN"
    assert_output --partial "\"ttmetal_sha\":\"$FULL_SHA\""
    assert_output --partial "\"ttlang_head\":\"$HEAD_SHA\""
    assert_output --partial "\"result\":\"no_compatible\""
}

@test "bucket is overridable via TTLANG_S3_BUCKET" {
    TTLANG_S3_BUCKET=other-bucket run -0 "$SCRIPT" \
        --ttmetal-sha "$FULL_SHA" --ttlang-head "$HEAD_SHA"
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3://other-bucket/tt-lang/$SHORT_SHA/attempt.json"
}
