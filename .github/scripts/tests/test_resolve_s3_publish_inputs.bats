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
    export DISPATCH_WHEEL_VARIANT=bundled
    export EVENT_NAME=workflow_dispatch
    export GITHUB_REF=refs/heads/main
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
    DISPATCH_WHEEL_VARIANT=light \
    EVENT_NAME=workflow_dispatch \
        run -0 "$SCRIPT"

    assert_equal "$(output_value docker_tag)" "mytag"
    assert_equal "$(output_value dry_run)" "true"
    assert_equal "$(output_value overwrite_releases)" "false"
    assert_equal "$(output_value version_override)" "1.2.3.dev20260101"
    assert_equal "$(output_value wheel_variant)" "light"
    assert_equal "$(output_value wheel_variants)" '["light"]'
    assert_equal "$(output_value wheel_matrix)" '{"include":[{"wheel_variant":"light","ttnn_dep_mode":"external"}]}'
    assert_equal "$(output_value standard_wheel_matrix)" '{"include":[]}'
    assert_output --partial "Using existing docker_tag=mytag"
}

@test "bundled-and-light expands to both build modes" {
    DISPATCH_WHEEL_VARIANT=bundled-and-light run -0 "$SCRIPT"

    assert_equal "$(output_value wheel_variant)" "bundled-and-light"
    assert_equal "$(output_value wheel_variants)" '["bundled","light"]'
    assert_equal "$(output_value wheel_matrix)" '{"include":[{"wheel_variant":"bundled","ttnn_dep_mode":"bundled"},{"wheel_variant":"light","ttnn_dep_mode":"external"}]}'
    assert_equal "$(output_value standard_wheel_matrix)" '{"include":[{"wheel_variant":"bundled","ttnn_dep_mode":"bundled"}]}'
    assert_output --partial 'Resolved wheel_variants=["bundled","light"]'
}

@test "unknown wheel variant -> error" {
    DISPATCH_WHEEL_VARIANT=garbage run -2 "$SCRIPT"
    assert_output --partial "Unknown S3 wheel variant: garbage"
}

@test "empty docker_tag -> hint about build-docker" {
    DISPATCH_DOCKER_TAG="" run -0 "$SCRIPT"
    assert_output --partial "No docker_tag provided; build-docker will create one"
}

@test "schedule event forces overwrite_releases=true even if dispatch said false" {
    DISPATCH_OVERWRITE_RELEASES=false EVENT_NAME=schedule run -0 "$SCRIPT"
    assert_equal "$(output_value overwrite_releases)" "true"
}

@test "schedule event defaults to bundled and light" {
    DISPATCH_WHEEL_VARIANT="" EVENT_NAME=schedule run -0 "$SCRIPT"
    assert_equal "$(output_value wheel_variant)" "bundled-and-light"
    assert_equal "$(output_value wheel_variants)" '["bundled","light"]'
    assert_equal "$(output_value standard_wheel_matrix)" '{"include":[{"wheel_variant":"bundled","ttnn_dep_mode":"bundled"}]}'
}

@test "schedule event keeps overwrite_releases=true if already set" {
    DISPATCH_OVERWRITE_RELEASES=true EVENT_NAME=schedule run -0 "$SCRIPT"
    assert_equal "$(output_value overwrite_releases)" "true"
}

@test "non-schedule event does not force overwrite_releases" {
    DISPATCH_OVERWRITE_RELEASES=false EVENT_NAME=workflow_dispatch run -0 "$SCRIPT"
    assert_equal "$(output_value overwrite_releases)" "false"
}

@test "stable tag push publishes bundled and light when public PyPI is blocked" {
    version_file=$(make_tt_metal_version_file \
        "$TEST_TT_METAL_RC1_TAG" \
        "$TEST_TT_METAL_NEXT_TAG")

    DISPATCH_VERSION_OVERRIDE="" \
    DISPATCH_WHEEL_VARIANT="" \
    EVENT_NAME=push \
    GITHUB_REF=refs/tags/v1.2.3 \
    TTLANG_TT_METAL_VERSION_FILE="$version_file" \
        run -0 "$SCRIPT"

    assert_equal "$(output_value version_override)" "1.2.3"
    assert_equal "$(output_value wheel_variant)" "bundled-and-light"
    assert_equal "$(output_value wheel_variants)" '["bundled","light"]'
    assert_equal "$(output_value overwrite_releases)" "false"
    assert_equal "$(output_value allow_final_internal_version)" "true"
}

@test "stable tag push publishes only light when public PyPI is aligned" {
    version_file=$(make_tt_metal_version_file \
        "$TEST_TT_METAL_RC2_TAG" \
        "$TEST_TT_METAL_TAG")

    DISPATCH_VERSION_OVERRIDE="" \
    DISPATCH_WHEEL_VARIANT="" \
    EVENT_NAME=push \
    GITHUB_REF=refs/tags/v1.2.3 \
    TTLANG_TT_METAL_VERSION_FILE="$version_file" \
        run -0 "$SCRIPT"

    assert_equal "$(output_value version_override)" "1.2.3"
    assert_equal "$(output_value wheel_variant)" "light"
    assert_equal "$(output_value wheel_variants)" '["light"]'
    assert_equal "$(output_value allow_final_internal_version)" "true"
}

@test "stable manual bundled publish is rejected when public PyPI is aligned" {
    version_file=$(make_tt_metal_version_file \
        "$TEST_TT_METAL_RC2_TAG" \
        "$TEST_TT_METAL_TAG")

    DISPATCH_VERSION_OVERRIDE="1.2.3" \
    DISPATCH_WHEEL_VARIANT=bundled \
    EVENT_NAME=workflow_dispatch \
    TTLANG_TT_METAL_VERSION_FILE="$version_file" \
        run -1 "$SCRIPT"

    assert_output --partial "Refusing to publish bundled tt-lang==1.2.3 to S3"
}

@test "push event rejects non-stable tag when version is unset" {
    DISPATCH_VERSION_OVERRIDE="" \
    DISPATCH_WHEEL_VARIANT="" \
    EVENT_NAME=push \
    GITHUB_REF=refs/tags/v1.2.3-rc1 \
        run -1 "$SCRIPT"

    assert_output --partial "S3 release-tag publish requires a stable tag"
}

@test "empty version_override invokes compute-nightly-version.py" {
    # Mock compute-nightly-version.py on PATH so we don't need git history.
    mock_bin="$BATS_TEST_TMPDIR/mock-bin"
    mkdir -p "$mock_bin"
    # The script invokes the compute-nightly script by absolute path
    # ($script_dir/compute-nightly-version.py), so shadow that file specifically.
    shadow_dir="$BATS_TEST_TMPDIR/shadow-scripts"
    mkdir -p "$shadow_dir/tests"
    # Copy real script and its sibling lib (so the sourced helper resolves),
    # then override compute-nightly.
    cp "$SCRIPT" "$shadow_dir/"
    cp -r "$SCRIPTS_DIR/lib" "$shadow_dir/"
    cat > "$shadow_dir/compute-nightly-version.py" <<'EOF'
#!/usr/bin/env python3
print("9.9.9.dev20991231")
EOF
    chmod +x "$shadow_dir/compute-nightly-version.py"

    DISPATCH_VERSION_OVERRIDE="" run -0 "$shadow_dir/resolve-s3-publish-inputs.sh"
    assert_equal "$(output_value version_override)" "9.9.9.dev20991231"
    assert_equal "$(output_value allow_final_internal_version)" "false"
}

@test "GITHUB_OUTPUT unset -> writes to stdout" {
    unset GITHUB_OUTPUT
    run -0 "$SCRIPT"
    assert_output --partial "version_override=42.42.42.dev20260527"
    assert_output --partial "wheel_variant=bundled"
    assert_output --partial 'wheel_variants=["bundled"]'
    assert_output --partial 'standard_wheel_matrix={"include":[{"wheel_variant":"bundled","ttnn_dep_mode":"bundled"}]}'
    assert_output --partial "allow_final_internal_version=false"
}

@test "appends rather than overwrites GITHUB_OUTPUT" {
    echo "prior=line" > "$GITHUB_OUTPUT_FILE"
    run -0 "$SCRIPT"
    run cat "$GITHUB_OUTPUT_FILE"
    assert_line --index 0 "prior=line"
    assert_output --partial "version_override=42.42.42.dev20260527"
}
