#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

load test_helper

setup() {
    REPO="$(mkrepo)"
    install_scripts_in_repo "$REPO"
    (
        cd "$REPO"
        git tag v1.2.3
    )
    SHA="$(git -C "$REPO" rev-parse HEAD)"
    SCRIPT="$SCRIPTS_DIR/resolve-pypi-publish-inputs.sh"
    OUTPUT_FILE="$BATS_TEST_TMPDIR/output"
    : > "$OUTPUT_FILE"

    export DRY_RUN=false
    export EVENT_NAME=workflow_dispatch
    export GITHUB_OUTPUT="$OUTPUT_FILE"
    export GITHUB_REF=refs/heads/main
    export GITHUB_SHA="$SHA"
    export RELEASE_SOURCE="$REPO"
    export TTLANG_SHA="$SHA"
}

output_value() {
    local key="$1"
    sed -n "s/^${key}=//p" "$OUTPUT_FILE"
}

@test "non-dry dispatch resolves the unique release tag" {
    run "$SCRIPT"

    assert_success
    assert_equal "$(output_value tag_version)" "1.2.3"
    assert_equal "$(output_value wheel_version)" "1.2.3"
}

@test "tag push resolves the triggering tag" {
    EVENT_NAME=push \
    GITHUB_REF=refs/tags/v1.2.3 \
    TTLANG_SHA="" \
        run "$SCRIPT"

    assert_success
    assert_equal "$(output_value tag_version)" "1.2.3"
    assert_equal "$(output_value wheel_version)" "1.2.3"
}

@test "dry dispatch uses a nightly version without requiring a release tag" {
    git -C "$REPO" tag -d v1.2.3 >/dev/null
    git -C "$REPO" tag v1.1.0

    DRY_RUN=true run "$SCRIPT"

    assert_success
    assert_equal "$(output_value tag_version)" ""
    [[ "$(output_value wheel_version)" =~ ^1\.1\.0\.dev[0-9]{8}$ ]]
}

@test "dispatch requires a full SHA" {
    TTLANG_SHA=main run "$SCRIPT"

    assert_failure
    assert_output --partial "ttlang_sha must be a full 40-character commit SHA"
}

@test "dispatch rejects an invalid Docker tag" {
    DOCKER_TAG='bad tag' run "$SCRIPT"

    assert_failure 2
    assert_output --partial "DOCKER_TAG is not a valid Docker tag"
}

@test "dispatch rejects a checkout that differs from the requested SHA" {
    TTLANG_SHA=0000000000000000000000000000000000000000 run "$SCRIPT"

    assert_failure
    assert_output --partial "Checked-out commit does not match ttlang_sha"
}

@test "non-dry dispatch requires main" {
    GITHUB_REF=refs/heads/feature run "$SCRIPT"

    assert_failure
    assert_output --partial "restricted to workflow dispatches from refs/heads/main"
}

@test "non-dry dispatch rejects multiple release tags" {
    git -C "$REPO" tag v1.2.4

    run "$SCRIPT"

    assert_failure
    assert_output --partial "must have exactly one v* release tag; found 2"
}

@test "unsupported event is rejected" {
    EVENT_NAME=pull_request run "$SCRIPT"

    assert_failure
    assert_output --partial "Unsupported public PyPI event"
}
