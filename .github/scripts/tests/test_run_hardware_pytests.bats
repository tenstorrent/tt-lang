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
    ln -s "$SCRIPTS_DIR/hardware-test-common.sh" "$BATS_TEST_TMPDIR/hardware-test-common.sh"
    cp "$SCRIPT" "$BATS_TEST_TMPDIR/run-hardware-pytests.sh"
    SCRIPT="$BATS_TEST_TMPDIR/run-hardware-pytests.sh"
    PATH="$BIN:$PATH"
    unset TT_VISIBLE_DEVICES
    unset TT_METAL_CACHE
    unset TTLANG_PIN_XDIST_WORKERS_TO_DEVICES
    unset TTLANG_XDIST_TT_METAL_CACHE_ROOT
}

# Fake python3 recording each invocation's args, exiting with $1 (default 0).
write_fake_python() {
    local exit_code="${1:-0}"
    cat > "$BIN/python3" <<EOF
#!/usr/bin/env bash
printf 'env:%s cache-root:%s args:%s vis:%s\n' "\${TTLANG_PIN_XDIST_WORKERS_TO_DEVICES:-}" "\${TTLANG_XDIST_TT_METAL_CACHE_ROOT:-}" "\$*" "\${TT_VISIBLE_DEVICES:-}" >> "$CALLS"
exit $exit_code
EOF
    chmod +x "$BIN/python3"
}

write_fake_python_sequence() {
    local first_exit="$1"
    local second_exit="$2"
    local count_file="$BATS_TEST_TMPDIR/python-call-count"
    cat > "$BIN/python3" <<EOF
#!/usr/bin/env bash
call_count=0
[ -f "$count_file" ] && call_count=\$(cat "$count_file")
call_count=\$((call_count + 1))
printf '%s\n' "\$call_count" > "$count_file"
printf 'env:%s cache-root:%s args:%s vis:%s\n' "\${TTLANG_PIN_XDIST_WORKERS_TO_DEVICES:-}" "\${TTLANG_XDIST_TT_METAL_CACHE_ROOT:-}" "\$*" "\${TT_VISIBLE_DEVICES:-}" >> "$CALLS"
case "\$call_count" in
    1) exit $first_exit ;;
    *) exit $second_exit ;;
esac
EOF
    chmod +x "$BIN/python3"
}

@test "multi-chip: parallel single-device run, then serial multi_device run" {
    write_fake_python 0

    HW_PYTEST_CHIPS=4 run "$SCRIPT" test/python build/test/pytest-report

    assert_success
    run cat "$CALLS"
    assert_line --partial "env:1 cache-root:$PWD/build/test/pytest-report-tt-metal-cache args:-m pytest"
    assert_line --partial "pytest test/python -m not multi_device -n 4"
    assert_line --partial "pytest-report-parallel.xml"
    assert_line --partial "env: cache-root: args:-m pytest test/python -m multi_device"
    assert_line --partial "pytest-report-multidevice.xml"
    [ "${#lines[@]}" -eq 2 ]
}

@test "multi-chip: a preset TT_VISIBLE_DEVICES is cleared for both phases" {
    write_fake_python 0

    TT_VISIBLE_DEVICES=2 HW_PYTEST_CHIPS=4 run "$SCRIPT" test/python build/test/pytest-report

    assert_success
    run cat "$CALLS"
    refute_output --partial "vis:2"
    [ "${#lines[@]}" -eq 2 ]
}

@test "single chip: one serial run over the whole suite" {
    write_fake_python 0

    HW_PYTEST_CHIPS=1 run "$SCRIPT" test/python build/test/pytest-report

    assert_success
    run cat "$CALLS"
    assert_output --partial "pytest test/python"
    refute_output --partial "env:1"
    refute_output --partial "cache-root:build"
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

@test "multi-chip: empty parallel phase is allowed when multi_device tests run" {
    write_fake_python_sequence 5 0

    HW_PYTEST_CHIPS=4 run "$SCRIPT" test/python/only_multidevice.py build/test/pytest-report

    assert_success
    run cat "$CALLS"
    [ "${#lines[@]}" -eq 2 ]
}

@test "multi-chip: empty multi_device phase is allowed when parallel tests run" {
    write_fake_python_sequence 0 5

    HW_PYTEST_CHIPS=4 run "$SCRIPT" test/python/only_single_device.py build/test/pytest-report

    assert_success
    run cat "$CALLS"
    [ "${#lines[@]}" -eq 2 ]
}

@test "multi-chip: both phases empty is an error" {
    write_fake_python_sequence 5 5

    HW_PYTEST_CHIPS=4 run "$SCRIPT" test/python/empty.py build/test/pytest-report

    assert_failure 5
}

@test "requires test-dir and report-prefix arguments" {
    run "$SCRIPT" test/python

    assert_failure
    assert_output --partial "usage"
}

@test "rejects invalid chip override" {
    write_fake_python 0

    HW_PYTEST_CHIPS=abc run "$SCRIPT" test/python build/test/pytest-report

    assert_failure 2
    assert_output --partial "chip count"
}
