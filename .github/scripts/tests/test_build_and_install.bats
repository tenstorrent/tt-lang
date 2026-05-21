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

# Build a stubbed PATH that captures cmake invocations to cmake_log() and
# makes pip a no-op. Run the script with the given args inside $REPO.
run_script() {
    local log
    log=$(cmake_log)
    : > "$log"

    mkdir -p "$BATS_TEST_TMPDIR/bin"
    cat > "$BATS_TEST_TMPDIR/bin/cmake" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$@" >> "$log"
echo "---END-INVOCATION---" >> "$log"
EOF
    cat > "$BATS_TEST_TMPDIR/bin/pip" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/cmake" "$BATS_TEST_TMPDIR/bin/pip"

    # Pre-create build dir + env/activate so do_configure's `source` succeeds.
    mkdir -p "$REPO/build-test/env"
    cat > "$REPO/build-test/env/activate" <<'EOF'
# stub activate
EOF

    (
        cd "$REPO"
        PATH="$BATS_TEST_TMPDIR/bin:$PATH" \
            CMAKE_BINARY_DIR=build-test \
            TTLANG_TOOLCHAIN_DIR="$BATS_TEST_TMPDIR/toolchain" \
            scripts/build-and-install.sh --configure-only "$@" 2>&1
    )
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
    run run_script
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

    run run_script --rebuild-ttmetal
    assert_success
    assert_output --partial ""  # script didn't error
    grep -q '^-DTTLANG_USE_TOOLCHAIN=ON$' "$(cmake_log)"
    grep -q '^-DTTLANG_USE_TOOLCHAIN_TTMETAL=OFF$' "$(cmake_log)"
}

@test "--force-rebuild sets both LLVM and tt-metal to OFF" {
    mkdir -p "$BATS_TEST_TMPDIR/toolchain/lib/cmake/mlir"
    : > "$BATS_TEST_TMPDIR/toolchain/lib/cmake/mlir/MLIRConfig.cmake"

    run run_script --force-rebuild
    assert_success
    grep -q '^-DTTLANG_USE_TOOLCHAIN=OFF$' "$(cmake_log)"
    grep -q '^-DTTLANG_USE_TOOLCHAIN_TTMETAL=OFF$' "$(cmake_log)"
    grep -q '^-DTTLANG_BUILD_TOOLCHAIN=ON$' "$(cmake_log)"
}

@test "--accept-ttmetal-mismatch sets the cmake flag" {
    run run_script --accept-ttmetal-mismatch
    assert_success
    grep -q '^-DTTLANG_ACCEPT_TTMETAL_MISMATCH=ON$' "$(cmake_log)"
}

@test "unknown flag is rejected with a warning" {
    run run_script --not-a-real-flag
    # Script warns but does not abort on unknown args.
    assert_output --partial "Unknown argument: --not-a-real-flag"
}
