#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/run-hardware-test-phase.sh.

load test_helper

setup() {
    TEST_REPO="$BATS_TEST_TMPDIR/repo"
    SCRIPT_DIR="$TEST_REPO/.github/scripts"
    SCRIPT="$SCRIPT_DIR/run-hardware-test-phase.sh"
    BIN="$BATS_TEST_TMPDIR/bin"
    CALLS="$BATS_TEST_TMPDIR/calls"
    export CALLS

    mkdir -p "$SCRIPT_DIR" "$TEST_REPO/build/env" "$TEST_REPO/build/test" "$BIN"
    cp "$SCRIPTS_DIR/run-hardware-test-phase.sh" "$SCRIPT"
    chmod +x "$SCRIPT"
    cat > "$TEST_REPO/build/env/activate" <<'EOF'
export BUILD_ENV_ACTIVE=1
EOF

    cat > "$BIN/cmake" <<'EOF'
#!/usr/bin/env bash
printf 'cmake active:%s args:%s\n' "${BUILD_ENV_ACTIVE:-}" "$*" >> "$CALLS"
EOF
    cat > "$BIN/python" <<'EOF'
#!/usr/bin/env bash
printf 'python active:%s visible:%s home:%s runtime:%s metal:%s ld:%s args:%s\n' \
    "${BUILD_ENV_ACTIVE:-}" "${TT_VISIBLE_DEVICES:-}" "$HOME" \
    "${TT_METAL_RUNTIME_ROOT:-}" "${TT_METAL_HOME:-}" \
    "${LD_LIBRARY_PATH:-}" "$*" >> "$CALLS"
EOF
    cat > "$BIN/python3" <<'EOF'
#!/usr/bin/env bash
printf 'python3 active:%s args:%s\n' "${BUILD_ENV_ACTIVE:-}" "$*" >> "$CALLS"
EOF
    cat > "$BIN/hostname" <<'EOF'
#!/usr/bin/env bash
echo worker-0
EOF
    chmod +x "$BIN/cmake" "$BIN/python" "$BIN/python3" "$BIN/hostname"
    PATH="$BIN:$PATH"
    export PATH
    unset EXABOX_WORKER_HOME TT_METAL_RUNTIME_ROOT TT_METAL_HOME LD_LIBRARY_PATH
    unset TT_VISIBLE_DEVICES
}

@test "configure uses the release toolchain configuration" {
    run -0 "$SCRIPT" configure

    run cat "$CALLS"
    assert_output --partial "cmake active: args:-G Ninja -B build"
    assert_output --partial "-DCMAKE_BUILD_TYPE=Release"
    assert_output --partial "-DTTLANG_USE_TOOLCHAIN=ON"
    assert_output --partial "-DTTLANG_ENABLE_PERF_TRACE=ON"
}

@test "build activates the environment and builds test tools" {
    run -0 "$SCRIPT" build

    run cat "$CALLS"
    assert_line "cmake active:1 args:--build build"
    assert_line "cmake active:1 args:--build build --target ttlang-test-tools"
}

@test "Galaxy simple-add exposes one device" {
    RUNS_ON=galaxy-bh run -0 "$SCRIPT" simple-add

    run cat "$CALLS"
    assert_output --partial "python active:1 visible:0"
    assert_output --partial "args:test/python/simple_add.py"
}

@test "N150 simple-add preserves device visibility" {
    TT_VISIBLE_DEVICES=7 RUNS_ON=n150 run -0 "$SCRIPT" simple-add

    run cat "$CALLS"
    assert_output --partial "python active:1 visible:7"
    assert_output --partial "args:test/python/simple_add.py"
}

@test "Exabox worker environment discards CPU runner tt-metal overrides" {
    EXABOX_WORKER_HOME=/home/user \
        TT_METAL_RUNTIME_ROOT=/cpu/runtime \
        TT_METAL_HOME=/cpu/home \
        LD_LIBRARY_PATH=/cpu/lib \
        run -0 "$SCRIPT" simple-add

    [ "$HOME" != /home/user ]
    run cat "$CALLS"
    assert_output --partial "home:/home/user runtime: metal: ld:"
    assert_output --partial "args:test/python/simple_add.py"
}

@test "report collection preserves relative report names" {
    mkdir -p "$TEST_REPO/build/test/nested"
    echo report > "$TEST_REPO/build/test/pytest-report.xml"
    echo nested > "$TEST_REPO/build/test/nested/me2e-report.xml"
    EXABOX_REPORT_DIR="$BATS_TEST_TMPDIR/reports" run -0 "$SCRIPT" collect-exabox-reports

    assert_output --partial "Copied 2 hardware test report(s)"
    [ -f "$BATS_TEST_TMPDIR/reports/worker-0/pytest-report.xml" ]
    [ -f "$BATS_TEST_TMPDIR/reports/worker-0/nested/me2e-report.xml" ]
}

@test "unknown phase fails" {
    run -2 "$SCRIPT" unknown
    assert_output --partial "Unknown hardware test phase: unknown"
}
