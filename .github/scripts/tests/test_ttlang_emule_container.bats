#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

load test_helper

SOURCE_REPO_ROOT="$TTLANG_REPO_ROOT"
RUNNER="$TTLANG_REPO_ROOT/scripts/tt-lang-emule-container.sh"
ENTRYPOINT="$TTLANG_REPO_ROOT/scripts/tt-lang-emule-entrypoint.sh"
INSTALLER="$TTLANG_REPO_ROOT/scripts/install-tt-lang-emule.sh"
DOCKERFILE="$TTLANG_REPO_ROOT/.github/containers/Dockerfile.emule"

make_stack_manifest() {
    local compiler_root="$1"
    local target="$2"
    local compiler_commit
    compiler_commit="$(git -C "$compiler_root" rev-parse HEAD)"
    sed \
        "s/\"base_commit\": \"[0-9a-f]*\"/\"base_commit\": \"$compiler_commit\"/" \
        "$SOURCE_REPO_ROOT/config/tt-lang-emule-stack.json" > "$target"
}

make_runner_fixture() {
    local root="$1"
    mkdir -p "$root/.github/containers" "$root/config" \
        "$root/examples" "$root/scripts" "$root/lib" "$root/cmake/modules"
    cp "$SOURCE_REPO_ROOT/.github/containers/Dockerfile.emule" \
        "$root/.github/containers/Dockerfile.emule"
    cp "$SOURCE_REPO_ROOT/scripts/tt-lang-emule-entrypoint.sh" \
        "$SOURCE_REPO_ROOT/scripts/tt-lang-emule-container.sh" \
        "$SOURCE_REPO_ROOT/scripts/tt-lang-emule-stack.py" \
        "$SOURCE_REPO_ROOT/scripts/shell-tt-lang-emule.sh" \
        "$SOURCE_REPO_ROOT/scripts/install-tt-lang-emule.sh" "$root/scripts/"
    cp "$SOURCE_REPO_ROOT/cmake/modules/TTLangUtils.cmake" "$root/cmake/modules/"
    touch "$root/examples/program.py"
    touch "$root/examples/eltwise_add.py"
    touch "$root/examples/compiler_only_external_call.py"
    touch "$root/lib/compiler.cpp"
    git -C "$root" init -q
    git -C "$root" add .
    git -C "$root" update-index --add --cacheinfo \
        160000,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,third-party/llvm-project
    git -C "$root" \
        -c user.name=test -c user.email=test@example.com \
        commit -q -m "Synthetic runtime inputs"
    make_stack_manifest "$root" "$root/config/tt-lang-emule-stack.json"
    git -C "$root" add config/tt-lang-emule-stack.json
    git -C "$root" \
        -c user.name=test -c user.email=test@example.com \
        commit -q -m "Pin synthetic compiler baseline"
}

pin_emulator_runtime() {
    python3 - "$TTLANG_REPO_ROOT/config/tt-lang-emule-stack.json" "$1" "$2" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
stack = json.loads(manifest.read_text())
stack["emulator"]["commit"] = sys.argv[2]
stack["metal"]["commit"] = sys.argv[3]
manifest.write_text(json.dumps(stack, indent=2) + "\n")
PY
}

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
        if [ "${MOCK_DOCKER_IMAGE_ERROR+x}" = x ]; then
            printf '%s\n' "$MOCK_DOCKER_IMAGE_ERROR" >&2
        elif [ "${MOCK_DOCKER_IMAGE_STATUS:-0}" -eq 1 ]; then
            printf 'Error response from daemon: No such image: %s\n' "$3" >&2
        fi
        if [ -n "${MOCK_DOCKER_IMAGE_OUTPUT:-}" ]; then
            printf '%s\n' "$MOCK_DOCKER_IMAGE_OUTPUT"
        fi
        if [ -n "${MOCK_DOCKER_IMAGE_SIGNAL:-}" ]; then
            kill -s "$MOCK_DOCKER_IMAGE_SIGNAL" "$$"
        fi
        exit "${MOCK_DOCKER_IMAGE_STATUS:-0}"
        ;;
    images)
        if [ -n "${MOCK_DOCKER_LIST_ERROR:-}" ]; then
            printf '%s\n' "$MOCK_DOCKER_LIST_ERROR" >&2
        fi
        if [ -n "${MOCK_DOCKER_LIST_REFERENCE:-}" ] && \
            [ "${4:-}" != "$MOCK_DOCKER_LIST_REFERENCE" ]; then
            printf '%s\n' unexpected-reference >&2
            exit 97
        fi
        if [ -n "${MOCK_DOCKER_LIST_OUTPUT:-}" ]; then
            printf '%s\n' "$MOCK_DOCKER_LIST_OUTPUT"
        fi
        exit "${MOCK_DOCKER_LIST_STATUS:-0}"
        ;;
    build)
        if [ "${MOCK_DOCKER_REQUIRE_SANITIZED_CONTEXT:-0}" = "1" ]; then
            source_context=""
            stack_context=""
            for argument in "$@"; do
                case "$argument" in
                    tt-emule-source=*)
                        source_context="${argument#tt-emule-source=}"
                        ;;
                    tt-lang-stack=*)
                        stack_context="${argument#tt-lang-stack=}"
                        ;;
                esac
            done
            [ -f "$source_context/tracked-source" ] || exit 97
            [ ! -e "$source_context/.git" ] || exit 98
            [ ! -e "$source_context/untracked-secret" ] || exit 99
            [ -f "$stack_context/tt-lang-emule-stack.json" ] || exit 96
        fi
        exit 0
        ;;
    run)
        exit "${MOCK_DOCKER_RUN_STATUS:-0}"
        ;;
    *)
        exit 99
        ;;
esac
EOF
    chmod +x "$target"
}

setup() {
    local setting
    for setting in $(compgen -A variable TTLANG_EMULE_) \
        $(compgen -A variable _TTLANG_EMULE_); do
        unset "$setting"
    done
    MOCK_DOCKER="$BATS_TEST_TMPDIR/docker"
    MOCK_DOCKER_LOG="$BATS_TEST_TMPDIR/docker.log"
    export MOCK_DOCKER_LOG
    make_mock_docker "$MOCK_DOCKER"
    unset TT_METAL_CACHE TT_EMULE_JIT_CACHE_DIR MESH_DEVICE EMULE_FABRIC8 \
        TT_METAL_ALLOCATOR_MODE_HYBRID TT_METAL_MOCK_CLUSTER_DESC_PATH
    # Keep the test manifest inside its own checkout, independent of CI depth.
    make_runner_fixture "$BATS_TEST_TMPDIR/checkout"
    TTLANG_REPO_ROOT="$(cd "$BATS_TEST_TMPDIR/checkout" && pwd -P)"
    RUNNER="$TTLANG_REPO_ROOT/scripts/tt-lang-emule-container.sh"
    ENTRYPOINT="$TTLANG_REPO_ROOT/scripts/tt-lang-emule-entrypoint.sh"
    INSTALLER="$TTLANG_REPO_ROOT/scripts/install-tt-lang-emule.sh"
    SHELL_LAUNCHER="$TTLANG_REPO_ROOT/scripts/shell-tt-lang-emule.sh"
    DOCKERFILE="$TTLANG_REPO_ROOT/.github/containers/Dockerfile.emule"
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

refute_log_contains() {
    run -1 grep -F -- "$1" "$MOCK_DOCKER_LOG"
}

make_mock_entrypoint_commands() {
    local target_dir="$1"
    mkdir -p "$target_dir"
    cat > "$target_dir/cmake" <<'EOF'
#!/usr/bin/env bash
for argument in "$@"; do
    printf 'cmake=%s\n' "$argument" >> "$MOCK_ENTRYPOINT_LOG"
done
if [ "${1:-}" = --build ]; then
    exit "${MOCK_CMAKE_BUILD_STATUS:-0}"
fi
exit "${MOCK_CMAKE_CONFIGURE_STATUS:-0}"
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

make_entrypoint_fixture() {
    export TT_METAL_ALLOCATOR_MODE_HYBRID=1 MESH_DEVICE=P150
    MOCK_ENTRYPOINT_LOG="$BATS_TEST_TMPDIR/entrypoint.log"
    export MOCK_ENTRYPOINT_LOG
    mock_bin="$BATS_TEST_TMPDIR/entrypoint-bin"
    build_dir="$BATS_TEST_TMPDIR/build"
    cluster="$BATS_TEST_TMPDIR/blackhole_P150_unharvested.yaml"
    program="$BATS_TEST_TMPDIR/program.py"
    llvm_revision_header="$BATS_TEST_TMPDIR/toolchain/include/llvm/Support/VCSRevision.h"
    memory_max_file="$BATS_TEST_TMPDIR/memory.max"
    test_entrypoint="$BATS_TEST_TMPDIR/entrypoint.sh"
    expected_llvm_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    source_fingerprint=source-fingerprint
    make_mock_entrypoint_commands "$mock_bin"
    mkdir -p "$build_dir/env" "$(dirname "$llvm_revision_header")"
    touch "$build_dir/env/activate" "$cluster" "$program"
    printf '%s\n' "$source_fingerprint" > \
        "$build_dir/.ttlang-emule-source-fingerprint"
    printf '#define LLVM_REVISION "%s"\n' "$expected_llvm_sha" > "$llvm_revision_header"
    printf '%s\n' "$((6 * 1024 * 1024 * 1024))" > "$memory_max_file"
    sed "s|/opt/ttlang-toolchain|$BATS_TEST_TMPDIR/toolchain|g" \
        "$ENTRYPOINT" > "$test_entrypoint"
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
        "test -f /opt/tt-emule/tt-metal-pin.txt" "$DOCKERFILE"
    run -0 grep -F -- \
        'grep -F -x -- "$TT_METAL_COMMIT" /opt/tt-emule/tt-metal-pin.txt' \
        "$DOCKERFILE"
    run -1 grep -F -- "github_token" "$DOCKERFILE"
}

@test "emule image records the complete runtime provenance" {
    run -0 grep -F -- \
        'io.tenstorrent.tt-lang.compiler.commit="${TT_LANG_COMPILER_BASE_COMMIT}"' \
        "$DOCKERFILE"
    run -0 grep -F -- \
        'io.tenstorrent.tt-lang.emule.commit="${TT_EMULE_COMMIT}"' \
        "$DOCKERFILE"
    run -0 grep -F -- \
        'io.tenstorrent.tt-lang.metal.commit="${TT_METAL_COMMIT}"' \
        "$DOCKERFILE"
    run -0 grep -F -- \
        'io.tenstorrent.tt-lang.runtime.manifest-sha256="${STACK_MANIFEST_SHA256}"' \
        "$DOCKERFILE"
    run -0 grep -F -- \
        'COPY --from=tt-lang-stack tt-lang-emule-stack.json' "$DOCKERFILE"
    run -0 grep -F -- \
        '/opt/tt-emule-runtime/source-manifest.json | sha256sum --check --strict' \
        "$DOCKERFILE"
    run -0 grep -F -- \
        '> /opt/tt-emule-runtime/stack.json' "$DOCKERFILE"
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
    assert_log_line "--entrypoint"
    assert_log_line "/workspace/scripts/tt-lang-emule-entrypoint.sh"
    assert_log_line \
        "TTLANG_EMULE_EXPECTED_LLVM_SHA=$(git -C "$TTLANG_REPO_ROOT" rev-parse HEAD:third-party/llvm-project)"
    [[ "$runtime_id" == 7292395c-b6c508c4-r* ]]
    assert_log_line \
        "type=volume,src=tt-lang-emule-build-${runtime_id}-${source_id},dst=/ttlang-build"
    assert_log_line \
        "type=volume,src=tt-lang-emule-cache-${runtime_id},dst=/tt-metal-cache"
    refute_log_line "build"
}

@test "linked worktree mounts its external Git metadata read-only" {
    local source_root="$BATS_TEST_TMPDIR/source"
    local worktree_root="$BATS_TEST_TMPDIR/worktree"
    local worktree_runner="$worktree_root/scripts/tt-lang-emule-container.sh"
    local git_common_dir
    make_runner_fixture "$source_root"
    git -C "$source_root" worktree add -q --detach "$worktree_root" HEAD
    git_common_dir="$(
        git -C "$worktree_root" rev-parse --path-format=absolute --git-common-dir
    )"

    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -0 "$worktree_runner" "$worktree_root/examples/program.py"

    assert_log_line \
        "type=bind,src=${git_common_dir},dst=${git_common_dir},readonly"
}

@test "compiler source identity reaches the container" {
    cd "$TTLANG_REPO_ROOT"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -0 "$RUNNER" examples/eltwise_add.py

    assert_log_line \
        "TTLANG_EMULE_COMPILER_SHA=$(git -C "$TTLANG_REPO_ROOT" rev-parse HEAD)"
    assert_log_contains "TTLANG_EMULE_SOURCE_FINGERPRINT="
    refute_log_contains "/ttlang-reports"
}

@test "installer is the only public path that enables installation" {
    cd "$TTLANG_REPO_ROOT"
    rm examples/compiler_only_external_call.py
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$INSTALLER"
    assert_output --partial "Runtime image: tt-lang-emule:"
    assert_output --partial "Compiler build volume: tt-lang-emule-build-"
    assert_output --partial "Runtime cache volume: tt-lang-emule-cache-"

    assert_log_line "TTLANG_EMULE_INSTALL=1"
    refute_log_contains "/workspace/examples/"

    run -2 "$INSTALLER" unexpected
    assert_output --partial "Usage: scripts/install-tt-lang-emule.sh"
}

@test "developer shell uses the same installed environment and working directory" {
    cd "$TTLANG_REPO_ROOT/examples"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$SHELL_LAUNCHER"

    assert_log_line "TTLANG_EMULE_SHELL=1"
    assert_log_line "--workdir"
    assert_log_line "/workspace/examples"
    assert_log_line "type=bind,src=${TTLANG_REPO_ROOT},dst=/workspace"
    assert_log_contains "dst=/ttlang-build"
    assert_log_contains "dst=/tt-metal-cache"
    assert_log_line "MESH_DEVICE=P150"
    assert_log_line "TT_METAL_ALLOCATOR_MODE_HYBRID=1"
    assert_log_contains "TTLANG_EMULE_SOURCE_FINGERPRINT="
    refute_log_line "TTLANG_EMULE_INSTALL=1"
    refute_log_line "build"

    run -2 "$SHELL_LAUNCHER" unexpected
    assert_output --partial "Usage: scripts/shell-tt-lang-emule.sh"
}

@test "installer and shell helper propagate a container failure" {
    cd "$TTLANG_REPO_ROOT"
    local launcher
    for launcher in "$INSTALLER" "$SHELL_LAUNCHER"; do
        : > "$MOCK_DOCKER_LOG"
        MOCK_DOCKER_RUN_STATUS=42 TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
            run -42 "$launcher"
        assert_log_line "run"
        refute_log_line "build"
    done
}

@test "installation and shell modes reject conflicting modes and extra arguments" {
    local launcher
    for launcher in "$RUNNER" "$ENTRYPOINT"; do
        TTLANG_EMULE_INSTALL=1 TTLANG_EMULE_SHELL=1 run -2 "$launcher"
        assert_output --partial "mutually exclusive"
        TTLANG_EMULE_INSTALL=1 run -2 "$launcher" unexpected
        assert_output --partial "do not accept script arguments"
        TTLANG_EMULE_SHELL=1 run -2 "$launcher" unexpected
        assert_output --partial "do not accept script arguments"
    done
    [ ! -e "$MOCK_DOCKER_LOG" ]
}

@test "workload edits and outputs preserve the installed compiler identity" {
    local synthetic_root="$BATS_TEST_TMPDIR/synthetic-repo"
    local synthetic_runner="$synthetic_root/scripts/tt-lang-emule-container.sh"
    local fingerprint
    make_runner_fixture "$synthetic_root"

    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$synthetic_runner" \
        "$synthetic_root/examples/program.py"
    fingerprint="$(grep '^TTLANG_EMULE_SOURCE_FINGERPRINT=' "$MOCK_DOCKER_LOG")"

    printf '# Edited workload\n' >> "$synthetic_root/examples/program.py"
    printf 'print("new workload")\n' > "$synthetic_root/program.py"
    printf '{"result": 42}\n' > "$synthetic_root/results.json"
    : > "$MOCK_DOCKER_LOG"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$synthetic_runner" \
        "$synthetic_root/program.py"

    assert_log_line "$fingerprint"
}

@test "tracked and new compiler sources change the installed compiler identity" {
    local synthetic_root="$BATS_TEST_TMPDIR/synthetic-repo"
    local synthetic_runner="$synthetic_root/scripts/tt-lang-emule-container.sh"
    local fingerprint
    local tracked_fingerprint
    local new_fingerprint
    make_runner_fixture "$synthetic_root"

    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$synthetic_runner" \
        "$synthetic_root/examples/program.py"
    fingerprint="$(grep '^TTLANG_EMULE_SOURCE_FINGERPRINT=' "$MOCK_DOCKER_LOG")"

    printf '// Compiler source edit\n' >> "$synthetic_root/lib/compiler.cpp"
    : > "$MOCK_DOCKER_LOG"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$synthetic_runner" \
        "$synthetic_root/examples/program.py"
    tracked_fingerprint="$(grep '^TTLANG_EMULE_SOURCE_FINGERPRINT=' "$MOCK_DOCKER_LOG")"
    [ "$tracked_fingerprint" != "$fingerprint" ]

    printf '// New compiler source\n' > "$synthetic_root/lib/new.cpp"
    : > "$MOCK_DOCKER_LOG"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$synthetic_runner" \
        "$synthetic_root/examples/program.py"
    new_fingerprint="$(grep '^TTLANG_EMULE_SOURCE_FINGERPRINT=' "$MOCK_DOCKER_LOG")"
    [ "$new_fingerprint" != "$tracked_fingerprint" ]
}

@test "runtime image identity changes when an image input changes" {
    local synthetic_root="$BATS_TEST_TMPDIR/synthetic-repo"
    local synthetic_runner="$synthetic_root/scripts/tt-lang-emule-container.sh"
    local first_image
    local second_image
    make_runner_fixture "$synthetic_root"

    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$synthetic_runner" \
        "$synthetic_root/examples/program.py"
    assert_log_line \
        "TTLANG_EMULE_EXPECTED_LLVM_SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    [ ! -e "$synthetic_root/third-party/llvm-project/.git" ]
    first_image="$(awk '/^tt-lang-emule:/{print; exit}' "$MOCK_DOCKER_LOG")"

    : > "$MOCK_DOCKER_LOG"
    printf '\n# changed image input\n' >> \
        "$synthetic_root/.github/containers/Dockerfile.emule"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$synthetic_runner" \
        "$synthetic_root/examples/program.py"
    second_image="$(awk '/^tt-lang-emule:/{print; exit}' "$MOCK_DOCKER_LOG")"

    [ "$first_image" != "$second_image" ]
}

@test "entrypoint edits use the mounted script without changing the runtime image" {
    cd "$TTLANG_REPO_ROOT"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$RUNNER" examples/program.py
    local first_image
    first_image="$(awk '/^tt-lang-emule:/{print; exit}' "$MOCK_DOCKER_LOG")"

    printf '\n# changed launcher input\n' >> "$ENTRYPOINT"
    : > "$MOCK_DOCKER_LOG"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$RUNNER" examples/program.py

    assert_log_line "$first_image"
    assert_log_line "/workspace/scripts/tt-lang-emule-entrypoint.sh"
    run -1 grep -F -- 'COPY tt-lang-emule-entrypoint.sh' "$DOCKERFILE"
}

@test "shallow checkout accepts its pinned HEAD but rejects an unavailable baseline" {
    local source_root="$BATS_TEST_TMPDIR/source"
    local shallow_root="$BATS_TEST_TMPDIR/shallow"
    local shallow_runner="$shallow_root/scripts/tt-lang-emule-container.sh"
    make_runner_fixture "$source_root"
    git clone -q --depth 1 "file://$source_root" "$shallow_root"
    [ "$(git -C "$shallow_root" rev-parse --is-shallow-repository)" = true ]

    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -1 "$shallow_runner" "$shallow_root/examples/program.py"
    assert_output --partial "compiler checkout does not contain the manifest compiler baseline"
    [ ! -e "$MOCK_DOCKER_LOG" ]

    make_stack_manifest "$shallow_root" "$shallow_root/config/tt-lang-emule-stack.json"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -0 "$shallow_runner" "$shallow_root/examples/program.py"
    assert_log_line "run"
    assert_log_line "TTLANG_EMULE_EXPECTED_LLVM_SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    refute_log_line "build"
}

@test "non-tty launch keeps stdin open without allocating a tty" {
    cd "$TTLANG_REPO_ROOT"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" run -0 "$RUNNER" \
        examples/eltwise_add.py

    assert_log_line "-i"
    refute_log_line "-t"
}

@test "redirected stdin prevents tty allocation when stdout is a tty" {
    cd "$TTLANG_REPO_ROOT"
    run -0 env TTLANG_EMULE_DOCKER="$MOCK_DOCKER" python3 - "$RUNNER" <<'PY'
import os
import pty
import subprocess
import sys

master, slave = pty.openpty()
try:
    result = subprocess.run(
        [sys.argv[1], "examples/eltwise_add.py"],
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=subprocess.PIPE,
        env=os.environ,
        check=False,
    )
finally:
    os.close(master)
    os.close(slave)
sys.stderr.buffer.write(result.stderr)
sys.exit(result.returncode)
PY

    assert_log_line "-i"
    refute_log_line "-t"
}

@test "missing image or explicit rebuild exports pinned source before the build" {
    local emule_source="$BATS_TEST_TMPDIR/external-emule"
    local runtime_tmp="$BATS_TEST_TMPDIR/runtime-tmp"
    local emule_commit
    local source_mode
    local source_dir
    local source_url
    local rebuild
    local inspect_status
    mkdir -p "$emule_source"
    mkdir -p "$runtime_tmp"
    git -C "$emule_source" init -q
    mkdir -p "$emule_source/cluster_descriptors"
    touch "$emule_source/tracked-source"
    touch \
        "$emule_source/cluster_descriptors/blackhole_P150_unharvested.yaml"
    printf '%s\n' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb > \
        "$emule_source/tt-metal-pin.txt"
    git -C "$emule_source" add \
        tracked-source cluster_descriptors/blackhole_P150_unharvested.yaml \
        tt-metal-pin.txt
    git -C "$emule_source" \
        -c user.name=test -c user.email=test@example.com \
        commit -q -m "Pinned source"
    touch "$emule_source/untracked-secret"
    emule_commit="$(git -C "$emule_source" rev-parse HEAD)"
    emule_source="$(cd "$emule_source" && pwd -P)"
    pin_emulator_runtime "$emule_commit" bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    cd "$TTLANG_REPO_ROOT"
    for source_mode in directory url rebuild; do
        source_dir=""
        source_url="$emule_source"
        if [ "$source_mode" != url ]; then
            source_dir="$emule_source"
            source_url="https://test-user:source-token@example.invalid/private.git"
        fi
        rebuild=0
        inspect_status=1
        if [ "$source_mode" = rebuild ]; then
            rebuild=1
            inspect_status=143
        fi
        : > "$MOCK_DOCKER_LOG"
        TMPDIR="$runtime_tmp" \
            MOCK_DOCKER_IMAGE_STATUS="$inspect_status" \
            MOCK_DOCKER_REQUIRE_SANITIZED_CONTEXT=1 \
            TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
            TTLANG_EMULE_INSTALL=1 \
            TTLANG_EMULE_REBUILD="$rebuild" \
            TTLANG_EMULE_RUNTIME_SOURCE_DIR="$source_dir" \
            TTLANG_EMULE_RUNTIME_SOURCE_URL="$source_url" \
            run -0 "$RUNNER"

        assert_log_line "build"
        assert_log_contains "Dockerfile.emule"
        assert_log_line \
            "TT_EMULE_COMMIT=$emule_commit"
        assert_log_line \
            "TT_METAL_COMMIT=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        assert_log_line \
            "TT_LANG_COMPILER_BASE_COMMIT=$(python3 -c \
                'import json, sys; print(json.load(open(sys.argv[1]))["compiler"]["base_commit"])' \
                "$TTLANG_REPO_ROOT/config/tt-lang-emule-stack.json")"
        assert_log_contains "STACK_MANIFEST_SHA256="
        assert_log_line "RUNTIME_PLATFORM=linux/amd64"
        assert_log_contains "tt-lang-stack=$runtime_tmp/tt-lang-stack-context."
        assert_log_contains "tt-emule-source="
        refute_log_line "tt-emule-source=$emule_source"
        refute_log_contains "TT_EMULE_SOURCE_URL="
        refute_log_contains "source-token"
        refute_log_contains "example.invalid/private.git"
        assert_log_line "${TTLANG_REPO_ROOT}/scripts"
        assert_log_line "run"
        if [ "$source_mode" = rebuild ]; then
            refute_log_line "image"
        fi
        shopt -s nullglob
        local retained_runtime_dirs=(
            "$runtime_tmp"/tt-lang-emule.*
            "$runtime_tmp"/tt-lang-emule-context.*
            "$runtime_tmp"/tt-lang-stack-context.*
        )
        shopt -u nullglob
        [ "${#retained_runtime_dirs[@]}" -eq 0 ]
        [ -f "$emule_source/tracked-source" ]
        [ -f "$emule_source/untracked-secret" ]
    done
}

@test "P150 target rejects an incompatible pinned runtime before Docker build" {
    local source_commit
    source_commit="$(git -C "$TTLANG_REPO_ROOT" rev-parse HEAD)"
    pin_emulator_runtime "$source_commit" bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_IMAGE_STATUS=1 \
        TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        TTLANG_EMULE_INSTALL=1 \
        TTLANG_EMULE_RUNTIME_SOURCE_DIR="$TTLANG_REPO_ROOT" \
        run -1 "$RUNNER"

    assert_output --partial "emulator target descriptor is missing"
    assert_output --partial "blackhole_P150_unharvested.yaml"
    refute_log_line "build"
    refute_log_line "run"
}

@test "an unpinned emulator checkout fails before the image build" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_IMAGE_STATUS=1 \
        TTLANG_EMULE_INSTALL=1 \
        TTLANG_EMULE_RUNTIME_SOURCE_DIR="$TTLANG_REPO_ROOT" \
        TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -1 "$RUNNER"

    assert_output --partial "emulator source must be at"
    refute_log_line "build"
    refute_log_line "run"
}

@test "a missing emulator source directory fails before the image build" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_IMAGE_STATUS=1 \
        TTLANG_EMULE_INSTALL=1 \
        TTLANG_EMULE_RUNTIME_SOURCE_DIR="$BATS_TEST_TMPDIR/missing" \
        TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -1 "$RUNNER"

    assert_output --partial "source directory not found"
    refute_log_line "build"
    refute_log_line "run"
}

@test "a missing image never causes ordinary execution to build" {
    cd "$TTLANG_REPO_ROOT"
    local diagnostic
    for diagnostic in \
        "Error response from daemon: No such image: runtime" \
        'Error response from daemon: {"message":"No such image: runtime"}' \
        '{"message":"No such image: runtime"}' \
        "image not found"; do
        : > "$MOCK_DOCKER_LOG"
        MOCK_DOCKER_IMAGE_STATUS=1 \
            MOCK_DOCKER_IMAGE_ERROR="$diagnostic" \
            MOCK_DOCKER_LIST_REFERENCE=runtime:latest \
            TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
            TTLANG_EMULE_IMAGE=runtime \
            run -1 "$RUNNER" examples/eltwise_add.py

        assert_output --partial "environment is not installed"
        assert_output --partial "scripts/install-tt-lang-emule.sh"
        assert_log_line "images"
        assert_log_line "runtime:latest"
        refute_log_line "build"
        refute_log_line "run"
    done
}

@test "a symbolic emulator revision in the manifest is rejected before Docker" {
    pin_emulator_runtime main bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    cd "$TTLANG_REPO_ROOT"
    TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        run -1 "$RUNNER" examples/eltwise_add.py

    assert_output --partial "emulator.commit must be a full lowercase commit SHA"
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

@test "image inspection daemon errors stop before sourcing building or running" {
    local diagnostic
    cd "$TTLANG_REPO_ROOT"
    for diagnostic in \
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock" \
        "permission denied while trying to connect to the Docker daemon socket" \
        $'Cannot connect to context\nError response from daemon: No such image: runtime'; do
        : > "$MOCK_DOCKER_LOG"
        MOCK_DOCKER_IMAGE_STATUS=1 \
            MOCK_DOCKER_IMAGE_ERROR="$diagnostic" \
            MOCK_DOCKER_LIST_STATUS=1 \
            MOCK_DOCKER_LIST_ERROR="$diagnostic" \
            TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
            TTLANG_EMULE_INSTALL=1 \
            TTLANG_EMULE_RUNTIME_SOURCE_DIR="$BATS_TEST_TMPDIR/never-read-source" \
            run -1 "$RUNNER"

        assert_output --partial "Docker could not list image"
        assert_output --partial "$diagnostic"
        assert_output --partial "Check the Docker daemon and selected context"
        refute_output --partial "emulator source directory not found"
        assert_log_line "image"
        refute_log_line "build"
        refute_log_line "run"
    done
}

@test "image inspection nonmissing exit statuses never trigger a build" {
    local inspect_status
    cd "$TTLANG_REPO_ROOT"
    for inspect_status in 2 125 130 137 143; do
        : > "$MOCK_DOCKER_LOG"
        MOCK_DOCKER_IMAGE_STATUS="$inspect_status" \
            MOCK_DOCKER_IMAGE_ERROR="Error response from daemon: No such image: runtime" \
            TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
            TTLANG_EMULE_RUNTIME_SOURCE_DIR="$BATS_TEST_TMPDIR/never-read-source" \
            run -"$inspect_status" "$RUNNER" examples/eltwise_add.py

        assert_output --partial "Docker could not inspect image"
        assert_output --partial "exit $inspect_status"
        refute_output --partial "emulator source directory not found"
        refute_log_line "build"
        refute_log_line "run"
    done
}

@test "terminated image inspection preserves failure without building" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_IMAGE_SIGNAL=TERM \
        TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        TTLANG_EMULE_RUNTIME_SOURCE_DIR="$BATS_TEST_TMPDIR/never-read-source" \
        run -143 "$RUNNER" examples/eltwise_add.py

    assert_output --partial "Docker could not inspect image"
    assert_output --partial "exit 143"
    refute_output --partial "emulator source directory not found"
    refute_log_line "build"
    refute_log_line "run"
}

@test "a listed image with failed inspection never triggers a build" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_IMAGE_STATUS=1 \
        MOCK_DOCKER_IMAGE_ERROR="" \
        MOCK_DOCKER_IMAGE_OUTPUT="Error response from daemon: No such image: runtime" \
        MOCK_DOCKER_LIST_OUTPUT=aaaaaaaaaaaa \
        TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
        TTLANG_EMULE_INSTALL=1 \
        TTLANG_EMULE_RUNTIME_SOURCE_DIR="$BATS_TEST_TMPDIR/never-read-source" \
        run -1 "$RUNNER"

    assert_output --partial "Docker could not inspect image"
    refute_output --partial "No such image: runtime"
    refute_output --partial "emulator source directory not found"
    refute_log_line "build"
    refute_log_line "run"
}

@test "image listing normalizes exact tags rather than accepting other tags" {
    local image
    local expected_tag
    cd "$TTLANG_REPO_ROOT"
    for image in runtime runtime:missing localhost:5000/runtime \
        docker.io/library/runtime index.docker.io/library/runtime; do
        expected_tag=runtime:latest
        case "$image" in
            runtime:*) expected_tag="$image" ;;
            localhost:*) expected_tag="$image:latest" ;;
        esac
        : > "$MOCK_DOCKER_LOG"
        MOCK_DOCKER_IMAGE_STATUS=1 \
            MOCK_DOCKER_LIST_REFERENCE="$expected_tag" \
            TTLANG_EMULE_IMAGE="$image" \
            TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
            run -1 "$RUNNER" examples/eltwise_add.py
        assert_output --partial "environment is not installed"
        assert_log_line "$expected_tag"
        refute_log_line "build"
        refute_log_line "run"
    done
}

@test "existing image IDs and digests preserve inspect semantics" {
    local image
    cd "$TTLANG_REPO_ROOT"
    for image in sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
        aaaaaaaaaaaa runtime@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; do
        : > "$MOCK_DOCKER_LOG"
        TTLANG_EMULE_IMAGE="$image" TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
            run -0 "$RUNNER" examples/eltwise_add.py
        assert_log_line "$image"
        assert_log_line run
        refute_log_line images
        refute_log_line build
    done
}

@test "failed image IDs digests and patterns never become build tags" {
    local image
    cd "$TTLANG_REPO_ROOT"
    for image in sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
        aaaaaaaaaaaa runtime@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
        'runtime:*' 'runtime:?' 'runtime:[a]'; do
        : > "$MOCK_DOCKER_LOG"
        MOCK_DOCKER_IMAGE_STATUS=1 \
            TTLANG_EMULE_IMAGE="$image" TTLANG_EMULE_DOCKER="$MOCK_DOCKER" \
            TTLANG_EMULE_INSTALL=1 \
            TTLANG_EMULE_RUNTIME_SOURCE_DIR="$BATS_TEST_TMPDIR/never-read-source" \
            run -1 "$RUNNER"
        assert_output --partial "Docker could not inspect image"
        refute_log_line images
        refute_log_line build
        refute_log_line run
    done
}

@test "image listing exit status is preserved" {
    cd "$TTLANG_REPO_ROOT"
    MOCK_DOCKER_IMAGE_STATUS=1 MOCK_DOCKER_LIST_STATUS=125 \
        TTLANG_EMULE_DOCKER="$MOCK_DOCKER" TTLANG_EMULE_INSTALL=1 \
        run -125 "$RUNNER"
    assert_output --partial "Docker could not list image"
    assert_output --partial "exit 125"
    refute_log_line build
    refute_log_line run
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

@test "entrypoint installation configures and builds without running a program" {
    make_entrypoint_fixture
    rm "$build_dir/.ttlang-emule-source-fingerprint"

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        _TTLANG_EMULE_CGROUP_MEMORY_MAX_PATH="$memory_max_file" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        TTLANG_EMULE_INSTALL=1 \
        TTLANG_EMULE_COMPILER_SHA=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
        TTLANG_EMULE_SOURCE_FINGERPRINT="$source_fingerprint" \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        TTLANG_EMULE_SOURCE_DIR="$TTLANG_REPO_ROOT" \
        run -0 /bin/bash "$test_entrypoint"

    assert_output --partial \
        "Installed compiler-backed emule environment for bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb."
    assert_output --partial "Building TT-Lang with 3 parallel job(s)."
    refute_output --partial "python="
    run -0 grep -F -x -- "cmake=--parallel" "$MOCK_ENTRYPOINT_LOG"
    run -0 grep -F -x -- "cmake=3" "$MOCK_ENTRYPOINT_LOG"
    run -0 grep -F -x -- \
        "cmake=-DTTLANG_EXTERNAL_TT_METAL_DIR=/opt/tt-emule-runtime/tt-metal" \
        "$MOCK_ENTRYPOINT_LOG"
    run -0 grep -F -x -- "$source_fingerprint" \
        "$build_dir/.ttlang-emule-source-fingerprint"
}

@test "entrypoint installation invalidates the old marker when configure or build fails" {
    make_entrypoint_fixture
    local failing_phase
    for failing_phase in MOCK_CMAKE_CONFIGURE_STATUS MOCK_CMAKE_BUILD_STATUS; do
        printf '%s\n' old-installation > "$build_dir/.ttlang-emule-source-fingerprint"
        run -42 env "$failing_phase=42" \
            PATH="$mock_bin:$PATH" \
            TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
            TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
            TTLANG_EMULE_INSTALL=1 \
            TTLANG_EMULE_COMPILER_SHA=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
            TTLANG_EMULE_SOURCE_FINGERPRINT="$source_fingerprint" \
            TTLANG_EMULE_BUILD_DIR="$build_dir" \
            TTLANG_EMULE_SOURCE_DIR="$TTLANG_REPO_ROOT" \
            /bin/bash "$test_entrypoint"

        refute_output --partial "Installed compiler-backed emule environment"
        [ ! -e "$build_dir/.ttlang-emule-source-fingerprint" ]
        [ ! -e "$build_dir/.ttlang-emule-source-fingerprint.tmp" ]
    done
}

@test "entrypoint runs from the installed environment without configuring" {
    make_entrypoint_fixture
    printf '#define LLVM_REVISION R"(%s)"\n' "$expected_llvm_sha" > "$llvm_revision_header"
    printf 'export TTLANG_SIM_ONLY=1 TTLANG_COMPILE_ONLY=1\n' > "$build_dir/env/activate"

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        TTLANG_EMULE_SOURCE_FINGERPRINT="$source_fingerprint" \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        TTLANG_COMPILE_ONLY=1 \
        TTLANG_SIM_ONLY=1 \
        run -0 /bin/bash "$test_entrypoint" "$program" "argument with spaces"

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
    [ ! -e "$MOCK_ENTRYPOINT_LOG" ]
}

@test "entrypoint honors explicitly configured cache paths" {
    make_entrypoint_fixture

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TT_METAL_CACHE="$BATS_TEST_TMPDIR/metal-cache" \
        TT_EMULE_JIT_CACHE_DIR="$BATS_TEST_TMPDIR/jit-cache" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        TTLANG_EMULE_SOURCE_FINGERPRINT="$source_fingerprint" \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        run -0 /bin/bash "$test_entrypoint" "$program"

    assert_line "emule_cache=$BATS_TEST_TMPDIR/jit-cache"
    [ ! -e "$MOCK_ENTRYPOINT_LOG" ]
}

@test "entrypoint shell activates the installed environment without running Python" {
    make_entrypoint_fixture
    printf 'export TTLANG_SIM_ONLY=1 TTLANG_COMPILE_ONLY=1\n' > "$build_dir/env/activate"

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        TTLANG_EMULE_SOURCE_FINGERPRINT="$source_fingerprint" \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        TTLANG_EMULE_SHELL=1 \
        run -0 /bin/bash "$test_entrypoint" <<'EOF'
printf 'emule=%s\nmesh=%s\nsim_only=%s\ncompile_only=%s\n' \
    "$TT_METAL_EMULE_MODE" "$MESH_DEVICE" "${TTLANG_SIM_ONLY:-}" "${TTLANG_COMPILE_ONLY:-}"
EOF

    assert_line "emule=1"
    assert_line "mesh=P150"
    assert_line "sim_only="
    assert_line "compile_only="
    [ ! -e "$MOCK_ENTRYPOINT_LOG" ]
}

@test "entrypoint shell rejects a stale compiler and preserves the shell exit status" {
    make_entrypoint_fixture
    TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_SOURCE_FINGERPRINT=stale \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        TTLANG_EMULE_SHELL=1 \
        run -1 /bin/bash "$test_entrypoint" <<< 'echo shell-must-not-run'
    assert_output --partial "installed compiler does not match this checkout"
    refute_output --partial "shell-must-not-run"

    TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_SOURCE_FINGERPRINT="$source_fingerprint" \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        TTLANG_EMULE_SHELL=1 \
        run -42 /bin/bash "$test_entrypoint" <<< 'exit 42'
    [ ! -e "$MOCK_ENTRYPOINT_LOG" ]
}

@test "entrypoint rejects an absent or stale installed compiler" {
    make_entrypoint_fixture
    rm "$build_dir/.ttlang-emule-source-fingerprint"

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        TTLANG_EMULE_SOURCE_FINGERPRINT="$source_fingerprint" \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        run -1 /bin/bash "$test_entrypoint" "$program"
    assert_output --partial "compiler environment is not installed"

    printf '%s\n' stale-fingerprint > \
        "$build_dir/.ttlang-emule-source-fingerprint"
    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        TTLANG_EMULE_SOURCE_FINGERPRINT="$source_fingerprint" \
        TTLANG_EMULE_BUILD_DIR="$build_dir" \
        run -1 /bin/bash "$test_entrypoint" "$program"
    assert_output --partial "installed compiler does not match this checkout"
}

@test "entrypoint rejects a different LLVM revision before configuring" {
    make_entrypoint_fixture
    printf '#define LLVM_REVISION "%s"\n' \
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb > "$llvm_revision_header"

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        run -1 /bin/bash "$test_entrypoint" "$program"

    assert_output --partial "runtime LLVM revision does not match"
    assert_output --partial "expected: $expected_llvm_sha"
    assert_output --partial "installed: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert_output --partial "Reinstall the compiler-backed environment"
    [ ! -e "$MOCK_ENTRYPOINT_LOG" ]
}

@test "entrypoint rejects an unknown installed LLVM revision before configuring" {
    make_entrypoint_fixture
    printf '#define LLVM_REVISION "unknown"\n' > "$llvm_revision_header"

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        run -1 /bin/bash "$test_entrypoint" "$program"

    assert_output --partial "installed: unknown"
    [ ! -e "$MOCK_ENTRYPOINT_LOG" ]
}

@test "entrypoint rejects a missing LLVM revision header before configuring" {
    make_entrypoint_fixture
    rm "$llvm_revision_header"

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="$expected_llvm_sha" \
        run -1 /bin/bash "$test_entrypoint" "$program"

    assert_output --partial "installed: unknown"
    assert_output --partial "$llvm_revision_header"
    [ ! -e "$MOCK_ENTRYPOINT_LOG" ]
}

@test "entrypoint requires the launcher's expected LLVM revision" {
    make_entrypoint_fixture

    PATH="$mock_bin:$PATH" \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster" \
        TTLANG_EMULE_EXPECTED_LLVM_SHA="" \
        run -1 /bin/bash "$test_entrypoint" "$program"

    assert_output --partial "expected LLVM revision is missing or invalid"
    [ ! -e "$MOCK_ENTRYPOINT_LOG" ]
}

@test "entrypoint requires every target setting from the launcher" {
    make_entrypoint_fixture
    local setting
    for setting in TT_METAL_MOCK_CLUSTER_DESC_PATH \
        TT_METAL_ALLOCATOR_MODE_HYBRID MESH_DEVICE; do
        run -1 env -u "$setting" /bin/bash "$test_entrypoint" "$program"
        assert_output --partial "required target setting ${setting} is missing"
        [ ! -e "$MOCK_ENTRYPOINT_LOG" ]
        export TT_METAL_MOCK_CLUSTER_DESC_PATH="$cluster"
    done
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
        TT_METAL_ALLOCATOR_MODE_HYBRID=1 MESH_DEVICE=P150 \
        run -1 "$ENTRYPOINT" "$program"
    assert_output --partial "cluster descriptor not found: $missing_cluster"
}
