#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/run-hardware-pytests.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/run-hardware-pytests.sh"
    BIN="$BATS_TEST_TMPDIR/bin"
    CALLS="$BATS_TEST_TMPDIR/pytest.calls"
    mkdir -p "$BIN"
    PATH="$BIN:$PATH"
}

# Fake python3 recording each invocation's args, exiting with $1 (default 0).
write_fake_python() {
    local exit_code="${1:-0}"
    cat > "$BIN/python3" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "$CALLS"
exit $exit_code
EOF
    chmod +x "$BIN/python3"
}

@test "multi-chip: parallel single-device run, then serial multi_device run" {
    write_fake_python 0

    HW_PYTEST_CHIPS=4 run "$SCRIPT" test/python build/test/pytest-report

    assert_success
    run cat "$CALLS"
    assert_line --partial "pytest test/python -m not multi_device -n 4"
    assert_line --partial "pytest-report-parallel.xml"
    assert_line --partial "pytest test/python -m multi_device"
    assert_line --partial "pytest-report-multidevice.xml"
    [ "${#lines[@]}" -eq 2 ]
}

@test "single chip: one serial run over the whole suite" {
    write_fake_python 0

    HW_PYTEST_CHIPS=1 run "$SCRIPT" test/python build/test/pytest-report

    assert_success
    run cat "$CALLS"
    assert_output --partial "pytest test/python"
    refute_output --partial " -n "
    refute_output --partial "multi_device"
    assert_output --partial "pytest-report.xml"
    [ "${#lines[@]}" -eq 1 ]
}

@test "zero chips: serial run, no parallelism" {
    write_fake_python 0

    HW_PYTEST_CHIPS=0 run "$SCRIPT" test/python build/test/pytest-report

    assert_success
    run cat "$CALLS"
    [ "${#lines[@]}" -eq 1 ]
}

@test "multi-chip: both runs execute even when the parallel run fails" {
    write_fake_python 1

    HW_PYTEST_CHIPS=4 run "$SCRIPT" test/python build/test/pytest-report

    assert_failure
    run cat "$CALLS"
    [ "${#lines[@]}" -eq 2 ]
}

@test "requires test-dir and report-prefix arguments" {
    run "$SCRIPT" test/python

    assert_failure
    assert_output --partial "usage"
}
