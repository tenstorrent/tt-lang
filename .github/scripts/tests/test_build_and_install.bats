#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for scripts/build-and-install.sh.
#
# Focus: flag parsing and the cmake arguments produced for each mode.
# Replaces `cmake` and `pip` with stubs that record their arguments to a
# log file, so do_configure can run end-to-end without actually configuring.

load test_helper

# Fixed log path so each test can grep it after `run run_script ...`.
cmake_log() { echo "$BATS_TEST_TMPDIR/cmake.log"; }
bash_log() { echo "$BATS_TEST_TMPDIR/bash.log"; }

# Build a stubbed PATH that captures cmake and bash invocations and makes pip
# and df no-ops. The bash stub prevents finalize-mode tests from running the
# real cleanup scripts copied into the synthetic repo.
setup_stubs() {
    mkdir -p "$BATS_TEST_TMPDIR/bin"
cat > "$BATS_TEST_TMPDIR/bin/cmake" <<EOF
#!/bin/bash
printf 'ENV_TTLANG_TTNN_DEP_MODE=%s\n' "\${TTLANG_TTNN_DEP_MODE:-}" >> "$(cmake_log)"
printf '%s\n' "\$@" >> "$(cmake_log)"
echo "---END-INVOCATION---" >> "$(cmake_log)"
EOF
    cat > "$BATS_TEST_TMPDIR/bin/pip" <<'EOF'
#!/bin/bash
exit 0
EOF
    cat > "$BATS_TEST_TMPDIR/bin/df" <<'EOF'
#!/bin/bash
exit 0
EOF
    cat > "$BATS_TEST_TMPDIR/bin/bash" <<EOF
#!/bin/bash
printf '%s\n' "\$@" >> "$(bash_log)"
exit 0
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/cmake" \
        "$BATS_TEST_TMPDIR/bin/pip" \
        "$BATS_TEST_TMPDIR/bin/df" \
        "$BATS_TEST_TMPDIR/bin/bash"
}

create_build_env() {
    local build_dir="$1"
    mkdir -p "$REPO/$build_dir/env"
    cat > "$REPO/$build_dir/env/activate" <<'EOF'
# stub activate
EOF
}

run_script_with_build_dir() {
    local build_dir="$1"
    shift

    : > "$(cmake_log)"
    : > "$(bash_log)"
    setup_stubs

    # Pre-create build dir + env/activate so do_configure's `source` succeeds.
    create_build_env "$build_dir"
    mkdir -p "$BATS_TEST_TMPDIR/toolchain"
    mkdir -p "$BATS_TEST_TMPDIR/home"

    (
        cd "$REPO"
        PATH="$BATS_TEST_TMPDIR/bin:$PATH" \
            HOME="$BATS_TEST_TMPDIR/home" \
            CMAKE_BINARY_DIR="$build_dir" \
            TTLANG_TOOLCHAIN_DIR="$BATS_TEST_TMPDIR/toolchain" \
            scripts/build-and-install.sh "$@" 2>&1
    )
}

run_script_default_build_dir() {
    : > "$(cmake_log)"
    : > "$(bash_log)"
    setup_stubs
    create_build_env build-toolchain
    create_build_env build
    mkdir -p "$BATS_TEST_TMPDIR/toolchain"
    mkdir -p "$BATS_TEST_TMPDIR/home"

    (
        cd "$REPO"
        PATH="$BATS_TEST_TMPDIR/bin:$PATH" \
            HOME="$BATS_TEST_TMPDIR/home" \
            TTLANG_TOOLCHAIN_DIR="$BATS_TEST_TMPDIR/toolchain" \
            scripts/build-and-install.sh "$@" 2>&1
    )
}

run_configure() {
    run_script_with_build_dir build-test --configure-only "$@"
}

setup() {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
    mkdir -p "$REPO/scripts"
    cp "$TTLANG_REPO_ROOT/scripts/build-and-install.sh" "$REPO/scripts/"
    cp "$TTLANG_REPO_ROOT/requirements.txt" "$REPO/" 2>/dev/null || \
        echo "" > "$REPO/requirements.txt"
}

@test "default: TTLANG_USE_TOOLCHAIN_TTMETAL follows TTLANG_USE_TOOLCHAIN" {
    run run_configure
    assert_success
    # The cmake stub recorded both args. They should agree (default behavior).
    use_toolchain=$(grep '^-DTTLANG_USE_TOOLCHAIN=' "$(cmake_log)" | head -1)
    use_ttmetal=$(grep '^-DTTLANG_USE_TOOLCHAIN_TTMETAL=' "$(cmake_log)" | head -1)
    [[ -n "$use_toolchain" ]]
    [[ -n "$use_ttmetal" ]]
    # Both ON or both OFF (no toolchain dir present in test env, so OFF).
    [[ "${use_toolchain##*=}" == "${use_ttmetal##*=}" ]]
}

@test "--rebuild-ttmetal sets TTLANG_USE_TOOLCHAIN_TTMETAL=OFF" {
    # Pre-create a fake toolchain so the default _use_toolchain becomes ON,
    # which lets us show that --rebuild-ttmetal overrides it for tt-metal only.
    mkdir -p "$BATS_TEST_TMPDIR/toolchain/lib/cmake/mlir"
    : > "$BATS_TEST_TMPDIR/toolchain/lib/cmake/mlir/MLIRConfig.cmake"

    run run_configure --rebuild-ttmetal
    assert_success
    grep -q '^-DTTLANG_USE_TOOLCHAIN=ON$' "$(cmake_log)"
    grep -q '^-DTTLANG_USE_TOOLCHAIN_TTMETAL=OFF$' "$(cmake_log)"
}

@test "--force-rebuild sets both LLVM and tt-metal to OFF" {
    mkdir -p "$BATS_TEST_TMPDIR/toolchain/lib/cmake/mlir"
    : > "$BATS_TEST_TMPDIR/toolchain/lib/cmake/mlir/MLIRConfig.cmake"

    run run_configure --force-rebuild
    assert_success
    grep -q '^-DTTLANG_USE_TOOLCHAIN=OFF$' "$(cmake_log)"
    grep -q '^-DTTLANG_USE_TOOLCHAIN_TTMETAL=OFF$' "$(cmake_log)"
    grep -q '^-DTTLANG_BUILD_TOOLCHAIN=ON$' "$(cmake_log)"
}

@test "--accept-ttmetal-mismatch sets the cmake flag" {
    run run_configure --accept-ttmetal-mismatch
    assert_success
    grep -q '^-DTTLANG_ACCEPT_TTMETAL_MISMATCH=ON$' "$(cmake_log)"
}

@test "external tt-metal source and build directories are forwarded to cmake" {
    run run_configure \
        --external-tt-metal-dir "$BATS_TEST_TMPDIR/external-metal" \
        --external-tt-metal-build-dir "$BATS_TEST_TMPDIR/external-metal-build"
    assert_success
    grep -q "^-DTTLANG_EXTERNAL_TT_METAL_DIR=$BATS_TEST_TMPDIR/external-metal$" "$(cmake_log)"
    grep -q "^-DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR=$BATS_TEST_TMPDIR/external-metal-build$" "$(cmake_log)"
}

@test "external python venv is forwarded to cmake" {
    run run_configure --python-venv "$BATS_TEST_TMPDIR/metal-python-env"
    assert_success
    grep -q "^-DTTLANG_PYTHON_VENV=$BATS_TEST_TMPDIR/metal-python-env$" "$(cmake_log)"
}

@test "ttnn dep mode external is exported for setup.py" {
    run run_configure --ttnn-dep-mode external
    assert_success
    grep -q '^ENV_TTLANG_TTNN_DEP_MODE=external$' "$(cmake_log)"
}

@test "ttnn dep mode pypi is exported for setup.py" {
    run run_configure --ttnn-dep-mode pypi
    assert_success
    grep -q '^ENV_TTLANG_TTNN_DEP_MODE=pypi$' "$(cmake_log)"
}

@test "ttnn dep mode bundled is exported for setup.py" {
    run run_configure --ttnn-dep-mode bundled
    assert_success
    grep -q '^ENV_TTLANG_TTNN_DEP_MODE=bundled$' "$(cmake_log)"
}

@test "ttnn dep mode option requires an argument" {
    run run_configure --ttnn-dep-mode
    assert_failure
    assert_output --partial "ERROR: --ttnn-dep-mode requires a mode"
    [[ ! -s "$(cmake_log)" ]]
}

@test "ttnn dep mode rejects unknown values" {
    run run_configure --ttnn-dep-mode invalid
    assert_failure
    assert_output --partial "ERROR: --ttnn-dep-mode must be one of"
    [[ ! -s "$(cmake_log)" ]]
}

@test "external python venv option requires an argument" {
    run run_configure --python-venv
    assert_failure
    assert_output --partial "ERROR: --python-venv requires a path"
    [[ ! -s "$(cmake_log)" ]]
}

@test "external tt-metal directory option requires an argument" {
    run run_configure --external-tt-metal-dir
    assert_failure
    assert_output --partial "ERROR: --external-tt-metal-dir requires a path"
    [[ ! -s "$(cmake_log)" ]]
}

@test "external tt-metal build directory option requires an argument" {
    run run_configure --external-tt-metal-build-dir
    assert_failure
    assert_output --partial "ERROR: --external-tt-metal-build-dir requires a path"
    [[ ! -s "$(cmake_log)" ]]
}

@test "external tt-metal build directory requires external tt-metal directory" {
    run run_configure --external-tt-metal-build-dir "$BATS_TEST_TMPDIR/external-metal-build"
    assert_failure
    assert_output --partial "ERROR: --external-tt-metal-build-dir requires --external-tt-metal-dir"
    [[ ! -s "$(cmake_log)" ]]
}

@test "llvm-toolchain-only requires external tt-metal directory" {
    run run_script_default_build_dir --llvm-toolchain-only
    assert_failure
    assert_output --partial "ERROR: --llvm-toolchain-only requires --external-tt-metal-dir"
    [[ ! -s "$(cmake_log)" ]]
}

@test "llvm-toolchain-only configures LLVM without installing tt-metal" {
    run run_script_default_build_dir \
        --llvm-toolchain-only \
        --force-rebuild \
        --external-tt-metal-dir "$BATS_TEST_TMPDIR/external-metal"
    assert_success
    grep -q '^-B$' "$(cmake_log)"
    grep -q '^build-toolchain$' "$(cmake_log)"
    grep -q '^-DTTLANG_FORCE_TOOLCHAIN_REBUILD=ON$' "$(cmake_log)"
    grep -q '^-DTTLANG_BUILD_TOOLCHAIN=OFF$' "$(cmake_log)"
    grep -q "^-DTTLANG_EXTERNAL_TT_METAL_DIR=$BATS_TEST_TMPDIR/external-metal$" "$(cmake_log)"
    ! grep -q 'scripts/install-ttmetal.sh' "$(bash_log)"
    assert_output --partial "=== LLVM toolchain build complete ==="
}

@test "unknown flag aborts with a non-zero exit" {
    run run_configure --not-a-real-flag
    assert_failure
    assert_output --partial "Unknown argument: --not-a-real-flag"
}

@test "--python without a path -> error (exit 1)" {
    run run_configure --python
    assert_failure
    assert_output --partial "ERROR: --python requires a path"
}
