#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/pull-image-retry.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/pull-image-retry.sh"
    MOCK_DOCKER="$BATS_TEST_TMPDIR/docker"
    CALLS="$BATS_TEST_TMPDIR/n"
    export PULL_RETRY_BASE_DELAY=0 # no real sleeps under test
}

# Mock docker whose `pull` fails its first $1 invocations, then succeeds.
# Records the running invocation count in $CALLS.
write_flaky_docker() {
    local fail_count="$1"
    cat > "$MOCK_DOCKER" <<EOF
#!/usr/bin/env bash
n=\$(cat "$CALLS" 2>/dev/null || echo 0)
n=\$((n + 1))
echo "\$n" > "$CALLS"
[ "\$n" -gt "$fail_count" ]
EOF
    chmod +x "$MOCK_DOCKER"
    export DOCKER="$MOCK_DOCKER"
}

@test "succeeds on the first try without retrying" {
    write_flaky_docker 0

    run "$SCRIPT" ghcr.io/example/image:tag

    assert_success
    run cat "$CALLS"
    assert_output "1"
}

@test "retries transient failures then succeeds" {
    write_flaky_docker 2 # fail twice, succeed on the third

    run "$SCRIPT" ghcr.io/example/image:tag

    assert_success
    assert_output --partial "retrying in"
    run cat "$CALLS"
    assert_output "3"
}

@test "fails after exhausting attempts" {
    write_flaky_docker 99 # always fail

    run "$SCRIPT" ghcr.io/example/image:tag

    assert_failure
    assert_output --partial "after 3 attempts"
    run cat "$CALLS"
    assert_output "3"
}

@test "honors PULL_RETRY_ATTEMPTS" {
    write_flaky_docker 99
    export PULL_RETRY_ATTEMPTS=2

    run "$SCRIPT" ghcr.io/example/image:tag

    assert_failure
    assert_output --partial "after 2 attempts"
    run cat "$CALLS"
    assert_output "2"
}

@test "requires an image argument" {
    run "$SCRIPT"

    assert_failure
    assert_output --partial "usage"
}
