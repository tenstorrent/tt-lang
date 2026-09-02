#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

load test_helper

RUNNER="$TTLANG_REPO_ROOT/scripts/tt-lang-emule-container.sh"
ENTRYPOINT="$TTLANG_REPO_ROOT/scripts/tt-lang-emule-entrypoint.sh"
DOCKERFILE="$TTLANG_REPO_ROOT/.github/containers/Dockerfile.emule"

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

make_mock_pinned_git() {
    local target="$1"
    cat > "$target" <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *"rev-parse HEAD"* ]]; then
    printf '%s\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    exit 0
fi
exit 99
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
printf 'allocator_hybrid=%s\n' "${TT_METAL_ALLOCATOR_MODE_HYBRID:-}"
printf 'fabric8=%s\n' "${EMULE_FABRIC8:-}"
printf 'emule_cache=%s\n' "${TT_EMULE_JIT_CACHE_DIR:-}"
printf 'mesh=%s\n' "${MESH_DEVICE:-}"
printf 'compile_only=%s\n' "${TTLANG_COMPILE_ONLY:-}"
printf 'sim_only=%s\n' "${TTLANG_SIM_ONLY:-}"
for argument in "$@"; do
    printf 'python=%s\n' "$argument"
done
EOF
    chmod +x "$target_dir/cmake" "$target_dir/nproc" "$target_dir/python"
}

@test "emule runtime fetches only the pinned source revisions" {
    run -0 grep -F -- \
        'fetch --depth 1 origin "$_TT_EMULE_COMMIT"' "$RUNNER"
    run -0 grep -F -- \
        'fetch --depth 1 origin "$TT_METAL_COMMIT"' "$DOCKERFILE"
    run -1 grep -F -- "git clone" "$RUNNER" "$DOCKERFILE"
}

@test "emule image receives its source as a credential-free build context" {
    run -0 grep -F -- \
        "COPY --from=tt-emule-source . /opt/tt-emule" "$DOCKERFILE"
    run -0 grep -F -- \
        'grep -F -x -- "$TT_METAL_COMMIT" /opt/tt-emule/tt-metal-pin.txt' \
        "$DOCKERFILE"
    run -1 grep -F -- "github_token" "$DOCKERFILE"
}

@test "emule image verifies that the built ttnn binding is importable" {
    run -0 grep -F -- \
        "/opt/ttlang-toolchain/venv/bin/python -c 'import ttnn'" "$DOCKERFILE"
}

@test "CMake device detection recognizes emule mode" {
    local probe="$BATS_TEST_TMPDIR/emule-device-probe.cmake"
    cat > "$probe" <<EOF
include("$TTLANG_REPO_ROOT/cmake/modules/TTLangUtils.cmake")
ttlang_check_device_available(has_device)
if(NOT has_device)
  message(FATAL_ERROR "emule mode was not detected")
endif()
EOF

    run env -u TT_METAL_SIMULATOR TT_METAL_EMULE_MODE=1 cmake -P "$probe"
    assert_success
    assert_output --partial "Tenstorrent device: emule mode"
}

@test "existing image runs a repository script through the mounted checkout" {
    local source_id
    local runtime_id
    source_id="$(printf '%s' "$TTLANG_REPO_ROOT" | cksum | awk '{print $1}')"
    cd "$TTLANG_REPO_ROOT"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$RUNNER" \
        examples/eltwise_add.py "argument with spaces"
    runtime_id="$(awk '/^tt-lang-emule:/{sub(/^tt-lang-emule:/, ""); print; exit}' "$MOCK_DOCKER_LOG")"

    assert_log_line \
        "type=bind,src=${TTLANG_REPO_ROOT},dst=/workspace"
    assert_log_line "/workspace/examples/eltwise_add.py"
    assert_log_line "argument with spaces"
    [[ "$runtime_id" == 07f1bd83-d48d09de-r* ]]
    assert_log_line \
        "type=volume,src=tt-lang-emule-build-${runtime_id}-${source_id},dst=/ttlang-build"
    assert_log_line \
        "type=volume,src=tt-lang-emule-cache-${runtime_id},dst=/tt-metal-cache"
    refute_log_line "build"
}

@test "runtime image identity changes when an image input changes" {
    local synthetic_root="$BATS_TEST_TMPDIR/synthetic-repo"
    local synthetic_runner="$synthetic_root/scripts/tt-lang-emule-container.sh"
    local first_image
    local second_image
    mkdir -p "$synthetic_root/.github/containers" \
        "$synthetic_root/examples" "$synthetic_root/scripts"
    cp "$DOCKERFILE" "$synthetic_root/.github/containers/Dockerfile.emule"
    cp "$ENTRYPOINT" "$synthetic_root/scripts/tt-lang-emule-entrypoint.sh"
    cp "$RUNNER" "$synthetic_runner"
    touch "$synthetic_root/examples/program.py"

    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$synthetic_runner" \
        "$synthetic_root/examples/program.py"
    first_image="$(awk '/^tt-lang-emule:/{print; exit}' "$MOCK_DOCKER_LOG")"

    : > "$MOCK_DOCKER_LOG"
    printf '\n# changed image input\n' >> \
        "$synthetic_root/scripts/tt-lang-emule-entrypoint.sh"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$synthetic_runner" \
        "$synthetic_root/examples/program.py"
    second_image="$(awk '/^tt-lang-emule:/{print; exit}' "$MOCK_DOCKER_LOG")"

    [ "$first_image" != "$second_image" ]
}

@test "non-tty launch keeps stdin open without allocating a tty" {
    cd "$TTLANG_REPO_ROOT"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$RUNNER" \
        examples/eltwise_add.py

    assert_log_line "-i"
    refute_log_line "-t"
}

@test "missing image triggers a pinned image build before the run" {
    local emule_source="$BATS_TEST_TMPDIR/external-emule"
    local mock_bin="$BATS_TEST_TMPDIR/pinned-git-bin"
    mkdir -p "$emule_source"
    mkdir -p "$mock_bin"
    emule_source="$(cd "$emule_source" && pwd -P)"
    make_mock_pinned_git "$mock_bin/git"
    cd "$TTLANG_REPO_ROOT"
    PATH="$mock_bin:$PATH" \
        MOCK_DOCKER_IMAGE_STATUS=1 TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        TTLANG_EMULE_RUNTIME_SOURCE_DIR="$emule_source" \
        TTLANG_EMULE_RUNTIME_COMMIT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
        TTLANG_EMULE_RUNTIME_METAL_COMMIT=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
        run -0 "$RUNNER" examples/eltwise_add.py

    assert_log_line "build"
    assert_log_contains "Dockerfile.emule"
    assert_log_line \
        "TT_EMULE_COMMIT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert_log_line \
        "TT_METAL_COMMIT=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert_log_line "tt-emule-source=$emule_source"
    assert_log_line "${TTLANG_REPO_ROOT}/scripts"
    assert_log_line "run"
}

@test "an unpinned emulator checkout fails before the image build" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_IMAGE_STATUS=1 \
        TTLANG_EMULE_RUNTIME_SOURCE_DIR="$TTLANG_REPO_ROOT" \
        TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -1 "$RUNNER" examples/eltwise_add.py

    assert_output --partial "emulator source must be at"
    refute_log_line "build"
    refute_log_line "run"
}

@test "a missing emulator source directory fails before the image build" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_IMAGE_STATUS=1 \
        TTLANG_EMULE_RUNTIME_SOURCE_DIR="$BATS_TEST_TMPDIR/missing" \
        TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -1 "$RUNNER" examples/eltwise_add.py

    assert_output --partial "source directory not found"
    refute_log_line "build"
    refute_log_line "run"
}

@test "a symbolic emulator revision is rejected before Docker" {
    cd "$TTLANG_REPO_ROOT"
    TTLANG_EMULE_RUNTIME_COMMIT=main TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -2 "$RUNNER" examples/eltwise_add.py

    assert_output --partial "must be full lowercase commit SHAs"
    [ ! -e "$MOCK_DOCKER_LOG" ]
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

@test "a missing script fails before any Docker call" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_INFO_STATUS=1 TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -2 "$RUNNER" missing.py

    assert_output --partial "script not found: missing.py"
    run -1 test -e "$MOCK_DOCKER_LOG"
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

@test "a script symlink outside the working directory mounts its target" {
    local link_dir="$BATS_TEST_TMPDIR/links"
    local target_dir="$BATS_TEST_TMPDIR/external"
    local target_dir_physical
    mkdir -p "$link_dir" "$target_dir"
    touch "$target_dir/program.py"
    ln -s "../external/program.py" "$link_dir/program.py"
    target_dir_physical="$(cd "$target_dir" && pwd -P)"
    cd "$link_dir"

    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$RUNNER" program.py

    assert_log_line \
        "type=bind,src=${target_dir_physical},dst=/ttlang-script"
    assert_log_line "/ttlang-script/program.py"
}

@test "entrypoint configures, builds, and runs with emule runtime state" {
    local mock_bin="$BATS_TEST_TMPDIR/entrypoint-bin"
    local build_dir="$BATS_TEST_TMPDIR/build"
    local cluster="$BATS_TEST_TMPDIR/blackhole_P150_unharvested.yaml"
    local program="$BATS_TEST_TMPDIR/program.py"
    MOCK_ENTRYPOINT_LOG="$BATS_TEST_TMPDIR/entrypoint.log"
    export MOCK_ENTRYPOINT_LOG
    make_mock_entrypoint_commands "$mock_bin"
    mkdir -p "$build_dir/env"
    touch "$build_dir/env/activate" "$cluster" "$program"

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_COMPILE_ONLY=1 \
        TTLANG_SIM_ONLY=1 \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        TTLANG_EMULE_SOURCE_DIR="$TTLANG_REPO_ROOT" \
        run -0 "$ENTRYPOINT" "$program" "argument with spaces"

    assert_line "emule=1"
    assert_line "slow_dispatch=1"
    assert_line "cluster=$cluster"
    assert_line "allocator_hybrid=1"
    assert_line "fabric8=1"
    assert_line "emule_cache=/tt-metal-cache/emule-jit"
    assert_line "mesh=P150"
    assert_line "compile_only="
    assert_line "sim_only="
    assert_line "python=$program"
    assert_line "python=argument with spaces"
    run -0 grep -F -x -- "cmake=--parallel" "$MOCK_ENTRYPOINT_LOG"
    run -0 grep -F -x -- "cmake=4" "$MOCK_ENTRYPOINT_LOG"
    run -0 grep -F -x -- \
        "cmake=-DTTLANG_EXTERNAL_TT_METAL_DIR=/opt/tt-emule-runtime/tt-metal" \
        "$MOCK_ENTRYPOINT_LOG"
}

@test "entrypoint defaults to the runtime's full P150 descriptor" {
    run -0 grep -F -x -- \
        'readonly CLUSTER_DESCRIPTORS="${TT_EMULE_SOURCE_DIR}/cluster_descriptors"' \
        "$ENTRYPOINT"
    run -0 grep -F -- \
        'blackhole_P150_unharvested.yaml' "$ENTRYPOINT"
}

@test "entrypoint rejects a missing script argument before configuring" {
    run -2 "$ENTRYPOINT"
    assert_output --partial "no Python script was provided"
}

@test "entrypoint rejects a missing script before configuring" {
    run -2 "$ENTRYPOINT" "$BATS_TEST_TMPDIR/missing.py"
    assert_output --partial "script not found: $BATS_TEST_TMPDIR/missing.py"
}

@test "entrypoint rejects a missing cluster descriptor before configuring" {
    local missing_cluster="$BATS_TEST_TMPDIR/missing.yaml"
    local program="$BATS_TEST_TMPDIR/program.py"
    touch "$program"
    TT_METAL_MOCK_CLUSTER_DESC_PATH="$missing_cluster" \
        run -1 "$ENTRYPOINT" "$program"
    assert_output --partial "cluster descriptor not found: $missing_cluster"
}
