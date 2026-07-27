#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

load test_helper

make_aws_mock() {
    local mock="$BATS_TEST_TMPDIR/aws"
    cat > "$mock" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_AWS_CALLS"
if [ "$1 $2" = "s3api head-object" ]; then
    case "${FAKE_AWS_MODE:-existing}" in
        missing)
            echo "An error occurred (404) when calling the HeadObject operation: Not Found" >&2
            exit 1
            ;;
        misleading-404)
            echo "An error occurred (AccessDenied) when calling the HeadObject operation: request ID 404-example" >&2
            exit 1
            ;;
    esac
    exit 0
fi
if [ "$1 $2 $3" = "s3 cp s3://test-bucket/state.json" ]; then
    case "${FAKE_AWS_MODE:-existing}" in
        existing)
            printf '{"ttlang_sha":"%s","version":"1.2.3.dev20260725"}\n' "$FAKE_MARKER_SHA"
            ;;
        invalid)
            echo "not-json"
            ;;
        error)
            echo "fatal error: AccessDenied" >&2
            exit 1
            ;;
    esac
    exit 0
fi
if [ "$1 $2 $3" = "s3 cp -" ]; then
    cat > "$FAKE_AWS_PAYLOAD"
    exit 0
fi
exit 2
EOF
    chmod +x "$mock"
    echo "$mock"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/s3-nightly-state.py"
    AWS_MOCK="$(make_aws_mock)"
    OUTPUT_FILE="$BATS_TEST_TMPDIR/output"
    CURRENT_SHA=1111111111111111111111111111111111111111
    PREVIOUS_SHA=2222222222222222222222222222222222222222
    export FAKE_AWS_CALLS="$BATS_TEST_TMPDIR/aws.calls"
    export FAKE_AWS_PAYLOAD="$BATS_TEST_TMPDIR/aws.payload"
    : > "$OUTPUT_FILE"
    : > "$FAKE_AWS_CALLS"
    export AWS="$AWS_MOCK"
    export GITHUB_OUTPUT="$OUTPUT_FILE"
}

output_value() {
    local key="$1"
    sed -n "s/^${key}=//p" "$OUTPUT_FILE"
}

@test "scheduled check skips an already-published SHA" {
    FAKE_MARKER_SHA="$CURRENT_SHA" run "$SCRIPT" check \
        --event schedule \
        --sha "$CURRENT_SHA" \
        --bucket test-bucket \
        --key state.json

    assert_success
    assert_equal "$(output_value publish-needed)" "false"
    assert_equal "$(output_value previous-sha)" "$CURRENT_SHA"
}

@test "scheduled check publishes a changed SHA" {
    FAKE_MARKER_SHA="$PREVIOUS_SHA" run "$SCRIPT" check \
        --event schedule \
        --sha "$CURRENT_SHA" \
        --bucket test-bucket \
        --key state.json

    assert_success
    assert_equal "$(output_value publish-needed)" "true"
    assert_equal "$(output_value previous-sha)" "$PREVIOUS_SHA"
}

@test "missing marker bootstraps the scheduled publish" {
    FAKE_AWS_MODE=missing run "$SCRIPT" check \
        --event schedule \
        --sha "$CURRENT_SHA" \
        --bucket test-bucket \
        --key state.json

    assert_success
    assert_equal "$(output_value publish-needed)" "true"
    assert_equal "$(output_value previous-sha)" ""
}

@test "manual check never reads S3 and always publishes" {
    FAKE_AWS_MODE=error run "$SCRIPT" check \
        --event workflow_dispatch \
        --sha "$CURRENT_SHA" \
        --bucket test-bucket \
        --key state.json

    assert_success
    assert_equal "$(output_value publish-needed)" "true"
    [[ ! -s "$FAKE_AWS_CALLS" ]]
}

@test "unexpected S3 read error fails" {
    FAKE_AWS_MODE=error run "$SCRIPT" check \
        --event schedule \
        --sha "$CURRENT_SHA" \
        --bucket test-bucket \
        --key state.json

    assert_failure
    assert_output --partial "AccessDenied"
}

@test "unrelated 404 text is not treated as a missing marker" {
    FAKE_AWS_MODE=misleading-404 run "$SCRIPT" check \
        --event schedule \
        --sha "$CURRENT_SHA" \
        --bucket test-bucket \
        --key state.json

    assert_failure
    assert_output --partial "AccessDenied"
    assert_output --partial "404-example"
}

@test "invalid marker JSON fails" {
    FAKE_AWS_MODE=invalid run "$SCRIPT" check \
        --event schedule \
        --sha "$CURRENT_SHA" \
        --bucket test-bucket \
        --key state.json

    assert_failure
    assert_output --partial "invalid nightly marker JSON"
}

@test "record writes source SHA and version as JSON" {
    run "$SCRIPT" record \
        --event schedule \
        --sha "$CURRENT_SHA" \
        --version 1.2.3.dev20260726 \
        --run-id 123 \
        --bucket test-bucket \
        --key state.json

    assert_success
    run python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["ttlang_sha"], data["version"], data["run_id"])' "$FAKE_AWS_PAYLOAD"
    assert_success
    assert_output "$CURRENT_SHA 1.2.3.dev20260726 123"
    grep -q '^s3 cp - s3://test-bucket/state.json --content-type application/json' "$FAKE_AWS_CALLS"
}

@test "record rejects non-schedule events" {
    run "$SCRIPT" record \
        --event workflow_dispatch \
        --sha "$CURRENT_SHA" \
        --version 1.2.3.dev20260726 \
        --bucket test-bucket \
        --key state.json

    assert_failure
    assert_output --partial "recorded only for schedule events"
}

@test "scheduled check rejects an invalid marker SHA" {
    FAKE_MARKER_SHA='invalid\npublish-needed=false' run "$SCRIPT" check \
        --event schedule \
        --sha "$CURRENT_SHA" \
        --bucket test-bucket \
        --key state.json

    assert_failure
    assert_output --partial "ttlang_sha is not a full SHA"
}
