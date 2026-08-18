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
    cat > "$BIN/nproc" <<'EOF'
#!/usr/bin/env bash
if [ "${NPROC_FAIL:-}" = 1 ]; then
    exit 1
fi
if [ "${1:-}" = --all ]; then
    echo 32
else
    echo 8
fi
EOF
    cat > "$BIN/hostname" <<'EOF'
#!/usr/bin/env bash
echo worker-0
EOF
    cat > "$SCRIPT_DIR/reset-tt-cards.sh" <<'EOF'
#!/usr/bin/env bash
printf 'reset active:%s args:%s\n' "${BUILD_ENV_ACTIVE:-}" "$*" >> "$CALLS"
EOF
    cat > "$SCRIPT_DIR/run-hardware-lit.sh" <<'EOF'
#!/usr/bin/env bash
printf 'lit active:%s args:%s\n' "${BUILD_ENV_ACTIVE:-}" "$*" >> "$CALLS"
EOF
    cat > "$SCRIPT_DIR/run-hardware-pytests.sh" <<'EOF'
#!/usr/bin/env bash
printf 'hardware-pytest active:%s args:%s\n' "${BUILD_ENV_ACTIVE:-}" "$*" >> "$CALLS"
EOF
    cat > "$SCRIPT_DIR/compile-and-run-examples.sh" <<'EOF'
#!/usr/bin/env bash
printf 'examples active:%s args:%s\n' "${BUILD_ENV_ACTIVE:-}" "$*" >> "$CALLS"
EOF
    chmod +x \
        "$BIN/cmake" \
        "$BIN/python" \
        "$BIN/python3" \
        "$BIN/hostname" \
        "$BIN/nproc" \
        "$SCRIPT_DIR/compile-and-run-examples.sh" \
        "$SCRIPT_DIR/reset-tt-cards.sh" \
        "$SCRIPT_DIR/run-hardware-lit.sh" \
        "$SCRIPT_DIR/run-hardware-pytests.sh"
    PATH="$BIN:$PATH"
    export PATH
    unset EXABOX_WORKER_HOME TT_METAL_RUNTIME_ROOT TT_METAL_HOME LD_LIBRARY_PATH
    unset HW_TEST_WORKERS
    unset TT_VISIBLE_DEVICES
}

@test "configure uses the release toolchain configuration" {
    run -0 "$SCRIPT" configure

    assert_output --partial "Host CPU budget: machine=32 available=8 cgroup="
    run cat "$CALLS"
    assert_output --partial "cmake active: args:-G Ninja -B build"
    assert_output --partial "-DCMAKE_BUILD_TYPE=Release"
    assert_output --partial "-DTTLANG_USE_TOOLCHAIN=ON"
    assert_output --partial "-DTTLANG_ENABLE_PERF_TRACE=ON"
}

@test "configure continues when CPU budget probing is unavailable" {
    NPROC_FAIL=1 run -0 "$SCRIPT" configure

    assert_output --partial "Host CPU budget: machine=unavailable available=unavailable"
    run cat "$CALLS"
    assert_output --partial "cmake active: args:-G Ninja -B build"
}

@test "build activates the environment and builds test tools" {
    run -0 "$SCRIPT" build

    run cat "$CALLS"
    assert_line "cmake active:1 args:--build build"
    assert_line "cmake active:1 args:--build build --target ttlang-test-tools"
}

@test "dependency and reset phases activate the build environment" {
    run -0 "$SCRIPT" install-dependencies
    run -0 "$SCRIPT" reset

    run cat "$CALLS"
    assert_line --partial "python3 active:1 args:-m pip install -r dev-requirements.txt"
    assert_line "reset active:1 args:"
}

@test "smoketest and simulator phases dispatch their Python commands" {
    run -0 "$SCRIPT" smoketest
    HW_TEST_WORKERS=8 run -0 "$SCRIPT" simulator

    run cat "$CALLS"
    assert_line --partial "python3 active:1 args:test/python/smoketest.py"
    assert_line --partial "python3 active:1 args:-m pytest test/sim -m requires_ttnn -v -n 8"
    assert_line --partial "--junitxml=build/test/sim-report.xml"
}

@test "simulator rejects an invalid worker override" {
    HW_TEST_WORKERS=0 run -2 "$SCRIPT" simulator

    assert_output --partial "worker count must be a positive integer or auto"
}

@test "hardware suite phases dispatch the shared runners" {
    run -0 "$SCRIPT" python-lit
    run -0 "$SCRIPT" python-pytests
    run -0 "$SCRIPT" me2e
    run -0 "$SCRIPT" tutorials

    run cat "$CALLS"
    assert_line "lit active:1 args:build/test/python build/test/python-lit-report"
    assert_line "hardware-pytest active:1 args:test/python build/test/pytest-report"
    assert_line "hardware-pytest active:1 args:test/me2e build/test/me2e-report"
    assert_line "hardware-pytest active:1 args:test/tutorial build/test/tutorial-report"
}

@test "examples phase dispatches the example runner" {
    run -0 "$SCRIPT" examples

    run cat "$CALLS"
    assert_line "examples active:1 args:"
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

@test "report collection rejects a relative destination" {
    EXABOX_REPORT_DIR=relative/reports run -2 "$SCRIPT" collect-exabox-reports
    assert_output --partial "EXABOX_REPORT_DIR must be absolute"
}

@test "unknown phase fails" {
    run -2 "$SCRIPT" unknown
    assert_output --partial "Unknown hardware test phase: unknown"
}
