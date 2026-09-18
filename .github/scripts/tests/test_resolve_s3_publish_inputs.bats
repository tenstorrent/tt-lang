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

@test "boolean inputs must be canonical" {
    DISPATCH_DRY_RUN=yes run -2 "$SCRIPT"
    assert_output --partial "DISPATCH_DRY_RUN must be true or false"

    DISPATCH_OVERWRITE_RELEASES=yes run -2 "$SCRIPT"
    assert_output --partial "DISPATCH_OVERWRITE_RELEASES must be true or false"
}

@test "docker tag rejects GitHub output injection" {
    DISPATCH_DOCKER_TAG=$'valid\npublish_needed=false' run -2 "$SCRIPT"
    assert_output --partial "DISPATCH_DOCKER_TAG is not a valid Docker tag"
    [[ ! -s "$GITHUB_OUTPUT_FILE" ]]
}

@test "version input is validated and normalized" {
    malicious_version='<script>alert(1)</script>'
    DISPATCH_VERSION_OVERRIDE="$malicious_version" run -1 "$SCRIPT"
    assert_output --partial "Invalid PEP 440 version"
    refute_output --partial "$malicious_version"

    DISPATCH_VERSION_OVERRIDE=1.2.3-rc1 run -0 "$SCRIPT"
    assert_equal "$(output_value version_override)" "1.2.3rc1"
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
    assert_equal "$(output_value bundled_selected)" "false"
    assert_equal "$(output_value manylinux_selected)" "true"
    assert_equal "$(output_value manylinux_wheel_matrix)" '{"include":[{"wheel_variant":"light","ttnn_dep_mode":"external","build_sim_wheel":false}]}'
    assert_output --partial "Using existing docker_tag=mytag"
}

@test "bundled-and-light expands to both build modes" {
    DISPATCH_WHEEL_VARIANT=bundled-and-light run -0 "$SCRIPT"

    assert_equal "$(output_value wheel_variant)" "bundled-and-light"
    assert_equal "$(output_value wheel_variants)" '["bundled","light"]'
    assert_equal "$(output_value bundled_selected)" "true"
    assert_equal "$(output_value manylinux_selected)" "true"
    assert_equal "$(output_value manylinux_wheel_matrix)" '{"include":[{"wheel_variant":"light","ttnn_dep_mode":"external","build_sim_wheel":false}]}'
    assert_output --partial 'Resolved wheel_variants=["bundled","light"]'
}

@test "unknown wheel variant -> error" {
    DISPATCH_WHEEL_VARIANT=garbage run -2 "$SCRIPT"
    assert_output --partial "Unknown S3 wheel variant: garbage"
}

@test "empty docker_tag -> hint about builder workflows" {
    DISPATCH_DOCKER_TAG="" run -0 "$SCRIPT"
    assert_output --partial "No docker_tag provided; required builder workflows will create it"
}

@test "schedule event forces overwrite_releases=true even if dispatch said false" {
    DISPATCH_OVERWRITE_RELEASES=false EVENT_NAME=schedule run -0 "$SCRIPT"
    assert_equal "$(output_value overwrite_releases)" "true"
}

@test "schedule event defaults to bundled and light" {
    DISPATCH_WHEEL_VARIANT="" EVENT_NAME=schedule run -0 "$SCRIPT"
    assert_equal "$(output_value wheel_variant)" "bundled-and-light"
    assert_equal "$(output_value wheel_variants)" '["bundled","light"]'
    assert_equal "$(output_value bundled_selected)" "true"
    assert_equal "$(output_value manylinux_selected)" "true"
    assert_equal "$(output_value manylinux_wheel_matrix)" '{"include":[{"wheel_variant":"light","ttnn_dep_mode":"external","build_sim_wheel":false}]}'
}

@test "schedule event keeps overwrite_releases=true if already set" {
    DISPATCH_OVERWRITE_RELEASES=true EVENT_NAME=schedule run -0 "$SCRIPT"
    assert_equal "$(output_value overwrite_releases)" "true"
}

@test "non-schedule event does not force overwrite_releases" {
    DISPATCH_OVERWRITE_RELEASES=false EVENT_NAME=workflow_dispatch run -0 "$SCRIPT"
    assert_equal "$(output_value overwrite_releases)" "false"
}

@test "pypi uses the shared manylinux build and includes sim" {
    DISPATCH_WHEEL_VARIANT=pypi run -0 "$SCRIPT"
    assert_equal "$(output_value bundled_selected)" "false"
    assert_equal "$(output_value manylinux_selected)" "true"
    assert_equal "$(output_value manylinux_wheel_matrix)" '{"include":[{"wheel_variant":"pypi","ttnn_dep_mode":"pypi","build_sim_wheel":true}]}'
}

@test "non-main dry run requires an existing docker tag" {
    GITHUB_REF=refs/heads/feature \
    DISPATCH_DRY_RUN=true \
    DISPATCH_DOCKER_TAG="" \
        run -1 "$SCRIPT"
    assert_output --partial "Non-main dry runs must provide docker_tag"
}

@test "non-main dry run with an existing docker tag is allowed" {
    GITHUB_REF=refs/heads/feature \
    DISPATCH_DRY_RUN=true \
    DISPATCH_DOCKER_TAG=existing \
        run -0 "$SCRIPT"
}

@test "push event is rejected" {
    EVENT_NAME=push run -1 "$SCRIPT"
    assert_output --partial "S3 PyPI publishing does not run for push events"
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

@test "unsupported event is rejected" {
    EVENT_NAME=pull_request run -1 "$SCRIPT"
    assert_output --partial "Unsupported S3 PyPI publish event: pull_request"
}

@test "empty version_override invokes compute-nightly-version.py" {
    # Shadow the helper next to a copied resolver so git history is not needed.
    shadow_dir="$BATS_TEST_TMPDIR/shadow-scripts"
    mkdir -p "$shadow_dir/tests"
    # Copy real script and its sibling lib (so the sourced helper resolves),
    # then override compute-nightly.
    cp "$SCRIPT" "$shadow_dir/"
    cp "$SCRIPTS_DIR/normalize-pep440-version.py" "$shadow_dir/"
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
    assert_output --partial "bundled_selected=true"
    assert_output --partial "manylinux_selected=false"
    assert_output --partial 'manylinux_wheel_matrix={"include":[]}'
    assert_output --partial "allow_final_internal_version=false"
}

@test "appends rather than overwrites GITHUB_OUTPUT" {
    echo "prior=line" > "$GITHUB_OUTPUT_FILE"
    run -0 "$SCRIPT"
    run cat "$GITHUB_OUTPUT_FILE"
    assert_line --index 0 "prior=line"
    assert_output --partial "version_override=42.42.42.dev20260527"
}
