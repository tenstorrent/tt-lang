#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

load test_helper

RUNNER="$TTLANG_REPO_ROOT/scripts/tt-lang-emule-container.sh"
ENTRYPOINT="$TTLANG_REPO_ROOT/scripts/tt-lang-emule-entrypoint.sh"

make_mock_docker() {
    local target="$1"
    cat > "$target" <<'EOF'
#!/usr/bin/env bash
for argument in "$@"; do
    printf '%s\n' "$argument" >> "$MOCK_DOCKER_LOG"
done
printf '%s\n' END >> "$MOCK_DOCKER_LOG"

case "${1:-}" in
    info)
        exit "${MOCK_DOCKER_INFO_STATUS:-0}"
        ;;
    image)
        exit "${MOCK_DOCKER_IMAGE_STATUS:-0}"
        ;;
    build|run)
        exit 0
        ;;
    *)
        exit 99
        ;;
esac
EOF
    chmod +x "$target"
}

setup() {
    MOCK_DOCKER="$BATS_TEST_TMPDIR/docker"
    MOCK_DOCKER_LOG="$BATS_TEST_TMPDIR/docker.log"
    export MOCK_DOCKER_LOG
    make_mock_docker "$MOCK_DOCKER"
}

assert_log_line() {
    run -0 grep -F -x -- "$1" "$MOCK_DOCKER_LOG"
}

assert_log_contains() {
    run -0 grep -F -- "$1" "$MOCK_DOCKER_LOG"
}

refute_log_line() {
    run -1 grep -F -x -- "$1" "$MOCK_DOCKER_LOG"
}

make_mock_entrypoint_commands() {
    local target_dir="$1"
    mkdir -p "$target_dir"
    cat > "$target_dir/cmake" <<'EOF'
#!/usr/bin/env bash
for argument in "$@"; do
    printf 'cmake=%s\n' "$argument" >> "$MOCK_ENTRYPOINT_LOG"
done
exit 0
EOF
    cat > "$target_dir/nproc" <<'EOF'
#!/usr/bin/env bash
printf '4\n'
EOF
    cat > "$target_dir/python" <<'EOF'
#!/usr/bin/env bash
printf 'emule=%s\n' "${TT_METAL_EMULE_MODE:-}"
printf 'slow_dispatch=%s\n' "${TT_METAL_SLOW_DISPATCH_MODE:-}"
printf 'cluster=%s\n' "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}"
printf 'mesh=%s\n' "${MESH_DEVICE:-}"
printf 'compile_only=%s\n' "${TTLANG_COMPILE_ONLY:-}"
printf 'sim_only=%s\n' "${TTLANG_SIM_ONLY:-}"
for argument in "$@"; do
    printf 'python=%s\n' "$argument"
done
EOF
    chmod +x "$target_dir/cmake" "$target_dir/nproc" "$target_dir/python"
}

@test "emule image context excludes the conventional local virtualenv" {
    run -0 grep -F -x -- ".venv/" "$TTLANG_REPO_ROOT/.dockerignore"
}

@test "existing image runs a repository script through the mounted checkout" {
    cd "$TTLANG_REPO_ROOT"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$RUNNER" \
        examples/eltwise_add.py "argument with spaces"

    assert_log_line \
        "type=bind,src=${TTLANG_REPO_ROOT},dst=/workspace"
    assert_log_line "/workspace/examples/eltwise_add.py"
    assert_log_line "argument with spaces"
    assert_log_line \
        "type=volume,src=tt-lang-emule-build-07f1bd83-d48d09de-r1,dst=/ttlang-build"
    assert_log_line \
        "type=volume,src=tt-lang-emule-cache-07f1bd83-d48d09de-r1,dst=/tt-metal-cache"
    refute_log_line "build"
}

@test "missing image triggers a pinned image build before the run" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_IMAGE_STATUS=1 TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -0 "$RUNNER" examples/eltwise_add.py

    assert_log_line "build"
    assert_log_contains "Dockerfile.emule"
    assert_log_line \
        "TT_EMULE_COMMIT=07f1bd8301544403c8bc1faa4038f6cbf69909f1"
    assert_log_line \
        "TT_METAL_COMMIT=d48d09dee19de51f694a52fdf75d569950d38ceb"
    assert_log_line "run"
}

@test "an unavailable daemon fails before inspecting or building the image" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_INFO_STATUS=1 TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -1 "$RUNNER" examples/eltwise_add.py

    assert_output --partial "cannot connect to the Docker daemon"
    refute_log_line "image"
    refute_log_line "build"
    refute_log_line "run"
}

@test "a missing script fails before image lookup" {
    cd "$TTLANG_REPO_ROOT"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -2 "$RUNNER" missing.py

    assert_output --partial "script not found: missing.py"
    refute_log_line "image"
}

@test "a script outside the working directory gets a dedicated mount" {
    local script_dir="$BATS_TEST_TMPDIR/external"
    local script_dir_physical
    mkdir -p "$script_dir"
    touch "$script_dir/program.py"
    script_dir_physical="$(cd "$script_dir" && pwd -P)"
    cd "$TTLANG_REPO_ROOT"

    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$RUNNER" \
        "$script_dir/program.py"

    assert_log_line \
        "type=bind,src=${script_dir_physical},dst=/ttlang-script"
    assert_log_line "/ttlang-script/program.py"
}

@test "entrypoint configures, builds, and runs with emule runtime state" {
    local mock_bin="$BATS_TEST_TMPDIR/entrypoint-bin"
    local build_dir="$BATS_TEST_TMPDIR/build"
    local cluster="$BATS_TEST_TMPDIR/wormhole_N150.yaml"
    MOCK_ENTRYPOINT_LOG="$BATS_TEST_TMPDIR/entrypoint.log"
    export MOCK_ENTRYPOINT_LOG
    make_mock_entrypoint_commands "$mock_bin"
    mkdir -p "$build_dir/env"
    touch "$build_dir/env/activate" "$cluster"

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_COMPILE_ONLY=1 \
        TTLANG_SIM_ONLY=1 \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        TTLANG_EMULE_SOURCE_DIR="$TTLANG_REPO_ROOT" \
        run -0 "$ENTRYPOINT" /workspace/program.py "argument with spaces"

    assert_line "emule=1"
    assert_line "slow_dispatch=1"
    assert_line "cluster=$cluster"
    assert_line "mesh=N150"
    assert_line "compile_only="
    assert_line "sim_only="
    assert_line "python=/workspace/program.py"
    assert_line "python=argument with spaces"
    run -0 grep -F -x -- "cmake=--parallel" "$MOCK_ENTRYPOINT_LOG"
    run -0 grep -F -x -- "cmake=4" "$MOCK_ENTRYPOINT_LOG"
    run -0 grep -F -x -- \
        "cmake=-DTTLANG_EXTERNAL_TT_METAL_DIR=/opt/tt-emule-runtime/tt-metal" \
        "$MOCK_ENTRYPOINT_LOG"
}

@test "entrypoint rejects a missing script argument before configuring" {
    run -2 "$ENTRYPOINT"
    assert_output --partial "no Python script was provided"
}

@test "entrypoint rejects a missing cluster descriptor before configuring" {
    local missing_cluster="$BATS_TEST_TMPDIR/missing.yaml"
    TT_METAL_MOCK_CLUSTER_DESC_PATH="$missing_cluster" \
        run -1 "$ENTRYPOINT" /workspace/program.py
    assert_output --partial "cluster descriptor not found: $missing_cluster"
}
