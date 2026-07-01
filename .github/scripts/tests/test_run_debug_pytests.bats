#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/run-debug-pytests.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/run-debug-pytests.sh"
    BIN="$BATS_TEST_TMPDIR/bin"
    CALLS="$BATS_TEST_TMPDIR/pytest.calls"
    GDB_CALLS="$BATS_TEST_TMPDIR/gdb.calls"
    mkdir -p "$BIN"
    PATH="$BIN:$PATH"
    export ARTIFACT_DIR="$BATS_TEST_TMPDIR/artifacts"
    export UNDER_GDB=false
    cd "$BATS_TEST_TMPDIR"
}

# Fake gdb recording its argv. It emulates gdb's exit-status contract: the
# child's failure is reported only when invoked with --return-child-result;
# otherwise a batch run exits 0 regardless of what the inferior did.
write_fake_gdb() {
    cat > "$BIN/gdb" <<EOF
#!/usr/bin/env bash
printf 'GDB %s\n' "\$*" >> "$GDB_CALLS"
for arg in "\$@"; do
    [ "\$arg" = "--return-child-result" ] && exit 1
done
exit 0
EOF
    chmod +x "$BIN/gdb"
}

# Fake python3 recording one "CALL" marker plus each arg per invocation, exiting
# with $1 (default 0). Per-arg lines let us assert quoted args stay grouped.
write_fake_python() {
    local exit_code="${1:-0}"
    cat > "$BIN/python3" <<EOF
#!/usr/bin/env bash
echo CALL >> "$CALLS"
printf 'arg:%s\n' "\$@" >> "$CALLS"
exit $exit_code
EOF
    chmod +x "$BIN/python3"
}

call_count() {
    grep -c '^CALL$' "$CALLS"
}

@test "default: runs the selection once and succeeds" {
    write_fake_python 0

    PYTEST_ARGS="test/python/test_reduce.py" run "$SCRIPT"

    assert_success
    [ "$(call_count)" -eq 1 ]
    run cat "$CALLS"
    assert_line "arg:test/python/test_reduce.py"
    assert_line "arg:-m"
    assert_line "arg:--timeout-method=thread"
}

@test "quoted -k expression stays a single argument" {
    write_fake_python 0

    PYTEST_ARGS='test/python/test_reduce.py -k "reduce_multi_tile and fp32"' run "$SCRIPT"

    assert_success
    run cat "$CALLS"
    assert_line "arg:-k"
    assert_line "arg:reduce_multi_tile and fp32"
}

@test "repeat runs every iteration when all pass" {
    write_fake_python 0

    PYTEST_ARGS="a.py" REPEAT=3 run "$SCRIPT"

    assert_success
    [ "$(call_count)" -eq 3 ]
}

@test "stop_on_fail halts at the first crashing iteration" {
    write_fake_python 139

    PYTEST_ARGS="a.py" REPEAT=5 STOP_ON_FAIL=true run "$SCRIPT"

    assert_failure
    [ "$(call_count)" -eq 1 ]
}

@test "stop_on_fail=false keeps going through every iteration" {
    write_fake_python 1

    PYTEST_ARGS="a.py" REPEAT=4 STOP_ON_FAIL=false run "$SCRIPT"

    assert_failure
    [ "$(call_count)" -eq 4 ]
}

@test "missing PYTEST_ARGS fails fast" {
    write_fake_python 0

    run "$SCRIPT"

    assert_failure
    assert_output --partial "PYTEST_ARGS"
}

@test "under_gdb=true fails when gdb is unavailable" {
    write_fake_python 0

    PATH="$BIN" PYTEST_ARGS="a.py" UNDER_GDB=true run /usr/bin/bash "$SCRIPT"

    assert_failure 2
    assert_output --partial "requires gdb"
}

@test "rejects invalid repeat count" {
    write_fake_python 0

    PYTEST_ARGS="a.py" REPEAT=0 run "$SCRIPT"

    assert_failure 2
    assert_output --partial "REPEAT"
}

@test "under_gdb=true detects a crashing child via --return-child-result" {
    write_fake_gdb

    PYTEST_ARGS="a.py" REPEAT=3 STOP_ON_FAIL=true UNDER_GDB=true run "$SCRIPT"

    assert_failure
    run cat "$GDB_CALLS"
    assert_output --partial "--return-child-result"
    [ "$(grep -c '^GDB ' "$GDB_CALLS")" -eq 1 ]
}
