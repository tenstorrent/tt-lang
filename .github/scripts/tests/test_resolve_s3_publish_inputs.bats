#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/resolve-s3-publish-inputs.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/resolve-s3-publish-inputs.sh"
    GITHUB_OUTPUT_FILE="$BATS_TEST_TMPDIR/github_output"
    : > "$GITHUB_OUTPUT_FILE"
    export GITHUB_OUTPUT="$GITHUB_OUTPUT_FILE"

    # Defaults for required env; tests override individual vars per case.
    export DISPATCH_DOCKER_TAG=""
    export DISPATCH_DRY_RUN=false
    export DISPATCH_OVERWRITE_RELEASES=false
    export DISPATCH_VERSION_OVERRIDE="42.42.42.dev20260527"
    export DISPATCH_TTNN_DEP_MODE=bundled
    export EVENT_NAME=workflow_dispatch
}

# Read one `key=value` line from the captured GITHUB_OUTPUT file.
output_value() {
    local key="$1"
    grep "^${key}=" "$GITHUB_OUTPUT_FILE" | sed "s/^${key}=//"
}

@test "missing DISPATCH_DRY_RUN -> error" {
    unset DISPATCH_DRY_RUN
    run -1 "$SCRIPT"
    assert_output --partial "DISPATCH_DRY_RUN is required"
}

@test "missing EVENT_NAME -> error" {
    unset EVENT_NAME
    run -1 "$SCRIPT"
    assert_output --partial "EVENT_NAME is required"
}

@test "workflow_dispatch with explicit inputs -> pass-through" {
    DISPATCH_DOCKER_TAG=mytag \
    DISPATCH_DRY_RUN=true \
    DISPATCH_OVERWRITE_RELEASES=false \
    DISPATCH_VERSION_OVERRIDE=1.2.3.dev20260101 \
    DISPATCH_TTNN_DEP_MODE=external \
    EVENT_NAME=workflow_dispatch \
        run -0 "$SCRIPT"

    assert_equal "$(output_value docker_tag)" "mytag"
    assert_equal "$(output_value dry_run)" "true"
    assert_equal "$(output_value overwrite_releases)" "false"
    assert_equal "$(output_value version_override)" "1.2.3.dev20260101"
    assert_equal "$(output_value ttnn_dep_mode)" "external"
    assert_output --partial "Using existing docker_tag=mytag"
}

@test "empty docker_tag -> hint about build-docker" {
    DISPATCH_DOCKER_TAG="" run -0 "$SCRIPT"
    assert_output --partial "No docker_tag provided; build-docker will create one"
}

@test "schedule event forces overwrite_releases=true even if dispatch said false" {
    DISPATCH_OVERWRITE_RELEASES=false EVENT_NAME=schedule run -0 "$SCRIPT"
    assert_equal "$(output_value overwrite_releases)" "true"
}

@test "schedule event keeps overwrite_releases=true if already set" {
    DISPATCH_OVERWRITE_RELEASES=true EVENT_NAME=schedule run -0 "$SCRIPT"
    assert_equal "$(output_value overwrite_releases)" "true"
}

@test "non-schedule event does not force overwrite_releases" {
    DISPATCH_OVERWRITE_RELEASES=false EVENT_NAME=workflow_dispatch run -0 "$SCRIPT"
    assert_equal "$(output_value overwrite_releases)" "false"
}

@test "empty version_override invokes compute-nightly-version.py" {
    # Mock compute-nightly-version.py on PATH so we don't need git history.
    mock_bin="$BATS_TEST_TMPDIR/mock-bin"
    mkdir -p "$mock_bin"
    # The script invokes the compute-nightly script by absolute path
    # ($script_dir/compute-nightly-version.py), so shadow that file specifically.
    shadow_dir="$BATS_TEST_TMPDIR/shadow-scripts"
    mkdir -p "$shadow_dir/tests"
    # Copy real script (so its sibling import resolves) and override compute-nightly.
    cp "$SCRIPT" "$shadow_dir/"
    cat > "$shadow_dir/compute-nightly-version.py" <<'EOF'
#!/usr/bin/env python3
print("9.9.9.dev20991231")
EOF
    chmod +x "$shadow_dir/compute-nightly-version.py"

    DISPATCH_VERSION_OVERRIDE="" run -0 "$shadow_dir/resolve-s3-publish-inputs.sh"
    assert_equal "$(output_value version_override)" "9.9.9.dev20991231"
}

@test "GITHUB_OUTPUT unset -> writes to stdout" {
    unset GITHUB_OUTPUT
    run -0 "$SCRIPT"
    assert_output --partial "version_override=42.42.42.dev20260527"
    assert_output --partial "ttnn_dep_mode=bundled"
}

@test "appends rather than overwrites GITHUB_OUTPUT" {
    echo "prior=line" > "$GITHUB_OUTPUT_FILE"
    run -0 "$SCRIPT"
    run cat "$GITHUB_OUTPUT_FILE"
    assert_line --index 0 "prior=line"
    assert_output --partial "version_override=42.42.42.dev20260527"
}
