#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the scripts that implement the shared manylinux wheel workflows.

load test_helper

make_docker_mock() {
    local mock="$BATS_TEST_TMPDIR/fake-docker"
    cat > "$mock" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_DOCKER_CALLS"
if [ "$1 $2" = "manifest inspect" ]; then
    case "$*" in
        *"${FAKE_MISSING_IMAGE:-__none__}"*) exit 1 ;;
        *) exit 0 ;;
    esac
fi
exit 0
EOF
    chmod +x "$mock"
    echo "$mock"
}

make_python_mock() {
    local mock="$BATS_TEST_TMPDIR/fake-python"
    cat > "$mock" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_PYTHON_CALLS"
if [ "$1 $2" = "-m venv" ]; then
    venv_dir="$3"
    mkdir -p "$venv_dir/bin"
    cp "$0" "$venv_dir/bin/python"
    : > "$venv_dir/bin/tt-triage"
    chmod +x "$venv_dir/bin/python" "$venv_dir/bin/tt-triage"
fi
exit 0
EOF
    chmod +x "$mock"
    echo "$mock"
}

setup() {
    export FAKE_DOCKER_CALLS="$BATS_TEST_TMPDIR/docker.calls"
    : > "$FAKE_DOCKER_CALLS"
    DOCKER_MOCK="$(make_docker_mock)"
}

@test "builder resolver reports a missing ABI image" {
    output_file="$BATS_TEST_TMPDIR/output"
    summary_file="$BATS_TEST_TMPDIR/summary"

    run env \
        DOCKER="$DOCKER_MOCK" \
        FAKE_MISSING_IMAGE=cp312 \
        GITHUB_OUTPUT="$output_file" \
        GITHUB_STEP_SUMMARY="$summary_file" \
        GITHUB_REF=refs/heads/feature \
        "$SCRIPTS_DIR/resolve-wheel-builder-images.sh" \
            --repository example/project \
            --docker-tag test-tag

    assert_success
    assert_output --partial "Image exists: ghcr.io/example/project/tt-lang-wheel-manylinux-2-34-cp310:test-tag"
    assert_output --partial "Image missing: ghcr.io/example/project/tt-lang-wheel-manylinux-2-34-cp312:test-tag"
    grep -qx 'docker-tag=test-tag' "$output_file"
    grep -qx 'all-images-exist=false' "$output_file"
    grep -q 'All images exist: `false`' "$summary_file"
}

@test "builder resolver updates both latest tags on main" {
    output_file="$BATS_TEST_TMPDIR/output"

    run env \
        DOCKER="$DOCKER_MOCK" \
        GITHUB_OUTPUT="$output_file" \
        GITHUB_REF=refs/heads/main \
        "$SCRIPTS_DIR/resolve-wheel-builder-images.sh" \
            --repository example/project \
            --docker-tag test-tag

    assert_success
    grep -qx 'all-images-exist=true' "$output_file"
    run grep -c '^buildx imagetools create -t .*:latest .*:test-tag$' "$FAKE_DOCKER_CALLS"
    assert_success
    assert_output "2"
}

@test "builder resolver distinguishes an older target from workflow source" {
    target_repo="$(mkrepo)"
    install_scripts_in_repo "$target_repo"
    (cd "$target_repo" && git tag v99.99.99)
    workflow_repo="$(mkrepo)"
    echo "workflow change" >> "$workflow_repo/CMakeLists.txt"
    commit_all "$workflow_repo" "workflow change"
    workflow_sha="$(git -C "$workflow_repo" rev-parse --short=8 HEAD)"
    output_file="$BATS_TEST_TMPDIR/output"

    cd "$target_repo"
    run env \
        DOCKER="$DOCKER_MOCK" \
        GITHUB_OUTPUT="$output_file" \
        GITHUB_REF=refs/heads/main \
        "$SCRIPTS_DIR/resolve-wheel-builder-images.sh" \
            --repository example/project \
            --workflow-source "$workflow_repo"

    assert_success
    grep -Eq "^docker-tag=v99\\.99\\.99-wf${workflow_sha}$" "$output_file"
    grep -qx 'update-latest=false' "$output_file"
    ! grep -q '^buildx imagetools create' "$FAKE_DOCKER_CALLS"
}

@test "manylinux input validation accepts supported dependency modes" {
    run "$SCRIPTS_DIR/validate-manylinux-wheel-inputs.sh" pypi 1.2.3
    assert_success

    run "$SCRIPTS_DIR/validate-manylinux-wheel-inputs.sh" \
        external \
        1.2.3.dev20260726
    assert_success
}

@test "manylinux input validation rejects unknown dependency modes" {
    run "$SCRIPTS_DIR/validate-manylinux-wheel-inputs.sh" bundled 1.2.3
    assert_failure 2
    assert_output --partial "ttnn_dep_mode must be pypi or external"
}

@test "S3 light compatibility entry point selects external-ttnn mode" {
    shadow_dir="$BATS_TEST_TMPDIR/shadow"
    mkdir -p "$shadow_dir"
    cp "$SCRIPTS_DIR/build-s3-light-core-wheel.sh" "$shadow_dir/"
    cat > "$shadow_dir/build-manylinux-core-wheel.sh" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" > "$BUILD_MEMBER_CALLS"
EOF
    chmod +x "$shadow_dir"/*.sh
    export BUILD_MEMBER_CALLS="$BATS_TEST_TMPDIR/member.calls"

    run "$shadow_dir/build-s3-light-core-wheel.sh" \
        --python-tag cp312 \
        --version 1.2.3 \
        --dist-dir dist

    assert_success
    run cat "$BUILD_MEMBER_CALLS"
    assert_output "--ttnn-dep-mode external --python-tag cp312 --version 1.2.3 --dist-dir dist"
}

@test "wheel-set member builds only the core wheel for cp310" {
    shadow_dir="$BATS_TEST_TMPDIR/shadow"
    mkdir -p "$shadow_dir"
    cp "$SCRIPTS_DIR/build-manylinux-wheel-set-member.sh" "$shadow_dir/"
    cat > "$shadow_dir/build-manylinux-core-wheel.sh" <<'EOF'
#!/bin/sh
printf 'core %s\n' "$*" >> "$BUILD_MEMBER_CALLS"
EOF
    cat > "$shadow_dir/build-s3-light-metapackage-wheel.sh" <<'EOF'
#!/bin/sh
printf 'meta %s\n' "$*" >> "$BUILD_MEMBER_CALLS"
EOF
    chmod +x "$shadow_dir"/*.sh
    export BUILD_MEMBER_CALLS="$BATS_TEST_TMPDIR/member.calls"

    run "$shadow_dir/build-manylinux-wheel-set-member.sh" \
        --python-tag cp310 \
        --version 1.2.3 \
        --ttnn-dep-mode pypi \
        --build-sim true \
        --dist-dir dist

    assert_success
    run cat "$BUILD_MEMBER_CALLS"
    assert_line "core --python-tag cp310 --version 1.2.3 --ttnn-dep-mode pypi --dist-dir dist"
    refute_output --partial "meta"
}

@test "external cp312 wheel-set member adds only the light metapackage when sim is disabled" {
    shadow_dir="$BATS_TEST_TMPDIR/shadow"
    mkdir -p "$shadow_dir"
    cp "$SCRIPTS_DIR/build-manylinux-wheel-set-member.sh" "$shadow_dir/"
    cat > "$shadow_dir/build-manylinux-core-wheel.sh" <<'EOF'
#!/bin/sh
printf 'core %s\n' "$*" >> "$BUILD_MEMBER_CALLS"
EOF
    cat > "$shadow_dir/build-s3-light-metapackage-wheel.sh" <<'EOF'
#!/bin/sh
printf 'meta %s\n' "$*" >> "$BUILD_MEMBER_CALLS"
EOF
    chmod +x "$shadow_dir"/*.sh
    export BUILD_MEMBER_CALLS="$BATS_TEST_TMPDIR/member.calls"

    run "$shadow_dir/build-manylinux-wheel-set-member.sh" \
        --python-tag cp312 \
        --version 1.2.3.dev20260726 \
        --ttnn-dep-mode external \
        --build-sim false \
        --dist-dir dist

    assert_success
    run cat "$BUILD_MEMBER_CALLS"
    assert_line "core --python-tag cp312 --version 1.2.3.dev20260726 --ttnn-dep-mode external --dist-dir dist"
    assert_line "meta --version 1.2.3.dev20260726 --dist-dir dist"
}

@test "wheel-set verification accepts a complete public PyPI set" {
    wheel_dir="$(make_wheel_dir \
        tt_lang-1.2.3-cp310-cp310-manylinux_2_34_x86_64.whl \
        tt_lang-1.2.3-cp312-cp312-manylinux_2_34_x86_64.whl \
        tt_lang_sim-1.2.3-py3-none-any.whl)"

    run "$SCRIPTS_DIR/verify-manylinux-wheel-set.sh" \
        --ttnn-dep-mode pypi \
        --build-sim true \
        1.2.3 \
        "$wheel_dir"

    assert_success
}

@test "wheel-set verification accepts a complete external-ttnn set without sim" {
    wheel_dir="$(make_wheel_dir \
        tt_lang-1.2.3.dev20260726+light-cp310-cp310-manylinux_2_34_x86_64.whl \
        tt_lang-1.2.3.dev20260726+light-cp312-cp312-manylinux_2_34_x86_64.whl \
        tt_lang_light-1.2.3.dev20260726-py3-none-any.whl)"

    run "$SCRIPTS_DIR/verify-manylinux-wheel-set.sh" \
        --ttnn-dep-mode external \
        --build-sim false \
        1.2.3.dev20260726 \
        "$wheel_dir"

    assert_success
}

@test "wheel-set verification rejects unexpected files" {
    wheel_dir="$(make_wheel_dir \
        tt_lang-1.2.3-cp310-cp310-manylinux_2_34_x86_64.whl \
        tt_lang-1.2.3-cp312-cp312-manylinux_2_34_x86_64.whl \
        tt_lang_sim-1.2.3-py3-none-any.whl \
        unrelated-1.0-py3-none-any.whl)"

    run "$SCRIPTS_DIR/verify-manylinux-wheel-set.sh" \
        --ttnn-dep-mode pypi \
        --build-sim true \
        1.2.3 \
        "$wheel_dir"

    assert_failure
    assert_output --partial "Unexpected manylinux wheel: unrelated-1.0-py3-none-any.whl"
}

@test "component cache exporter keeps LLVM and tt-metal cache references separate" {
    repo="$(mkrepo)"
    cd "$repo"

    run env DOCKER="$DOCKER_MOCK" \
        "$CONTAINERS_DIR/cache-wheel-manylinux-component.sh" \
            --component llvm \
            --python-tag cp310 \
            --cache-ref ghcr.io/example/cache:llvm-cp310 \
            --build-parallel-level 2
    assert_success

    run env DOCKER="$DOCKER_MOCK" \
        "$CONTAINERS_DIR/cache-wheel-manylinux-component.sh" \
            --component ttmetal \
            --python-tag "" \
            --cache-ref ghcr.io/example/cache:ttmetal-cp312 \
            --build-parallel-level 2
    assert_success

    run cat "$FAKE_DOCKER_CALLS"
    assert_line --partial "--target llvm-toolchain"
    assert_line --partial "--cache-to type=registry,ref=ghcr.io/example/cache:llvm-cp310,mode=max"
    assert_line --partial "--build-arg PYTHON_TAG=cp310"
    assert_line --partial "--build-arg WORKFLOW_SOURCE=."
    refute_line --regexp 'llvm-toolchain.*TT_METAL_TAG'
    assert_line --partial "--target ttmetal-toolchain"
    assert_line --partial "--cache-to type=registry,ref=ghcr.io/example/cache:ttmetal-cp312,mode=max"
}

@test "builder image consumes both component caches" {
    repo="$(mkrepo)"
    cd "$repo"

    run env \
        DOCKER="$DOCKER_MOCK" \
        FAKE_MISSING_IMAGE=cp312 \
        GITHUB_REF=refs/heads/feature \
        "$CONTAINERS_DIR/build-wheel-manylinux-images.sh" \
            --python-tags cp312 \
            --image-tag test-tag \
            --llvm-cache-ref ghcr.io/example/cache:llvm-cp312 \
            --ttmetal-cache-ref ghcr.io/example/cache:ttmetal-cp312

    assert_success
    run cat "$FAKE_DOCKER_CALLS"
    assert_line --partial "buildx build"
    assert_line --partial "--cache-from type=registry,ref=ghcr.io/example/cache:llvm-cp312"
    assert_line --partial "--cache-from type=registry,ref=ghcr.io/example/cache:ttmetal-cp312"
    assert_line --partial "--build-arg WORKFLOW_SOURCE=."
    assert_line --partial "--push -t ghcr.io/tenstorrent/tt-lang/tt-lang-wheel-manylinux-2-34-cp312:test-tag"
}

@test "hardware wheel test script orchestrates install and tutorials" {
    wheel_dir="$(make_wheel_dir \
        tt_lang-1.2.3-cp312-cp312-manylinux_2_34_x86_64.whl)"
    python_mock="$(make_python_mock)"
    tutorial_mock="$BATS_TEST_TMPDIR/tutorial"
    cat > "$tutorial_mock" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_TUTORIAL_CALLS"
EOF
    chmod +x "$tutorial_mock"
    export FAKE_PYTHON_CALLS="$BATS_TEST_TMPDIR/python.calls"
    export FAKE_TUTORIAL_CALLS="$BATS_TEST_TMPDIR/tutorial.calls"
    : > "$FAKE_PYTHON_CALLS"
    : > "$FAKE_TUTORIAL_CALLS"

    run env \
        FAKE_PYTHON_CALLS="$FAKE_PYTHON_CALLS" \
        FAKE_TUTORIAL_CALLS="$FAKE_TUTORIAL_CALLS" \
        "$SCRIPTS_DIR/test-manylinux-wheel.sh" \
            --dist-dir "$wheel_dir" \
            --ttnn-dep-mode pypi \
            --python "$python_mock" \
            --repo-root "$TTLANG_REPO_ROOT" \
            --tutorial-script "$tutorial_mock"

    assert_success
    grep -q -- "-m pip install .*tt_lang-1.2.3-cp312" "$FAKE_PYTHON_CALLS"
    grep -q -- "check-installed-ttnn.py --mode pypi" "$FAKE_PYTHON_CALLS"
    grep -q -- "smoke-test-wheel.py" "$FAKE_PYTHON_CALLS"
    grep -qx "$TTLANG_REPO_ROOT" "$FAKE_TUTORIAL_CALLS"
}

@test "hardware wheel test script rejects a missing cp312 wheel" {
    wheel_dir="$(make_wheel_dir \
        tt_lang-1.2.3-cp310-cp310-manylinux_2_34_x86_64.whl)"

    run "$SCRIPTS_DIR/test-manylinux-wheel.sh" \
        --dist-dir "$wheel_dir" \
        --ttnn-dep-mode pypi \
        --python /does/not/matter

    assert_failure
    assert_output --partial "cp312 manylinux_2_34 tt-lang wheel not found"
}
