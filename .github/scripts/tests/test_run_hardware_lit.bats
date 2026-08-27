#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/run-hardware-lit.sh.

load test_helper

setup() {
    SCRIPT="$BATS_TEST_TMPDIR/run-hardware-lit.sh"
    BIN="$BATS_TEST_TMPDIR/bin"
    CALLS="$BATS_TEST_TMPDIR/lit.calls"
    mkdir -p "$BIN"
    ln -s "$SCRIPTS_DIR/hardware-test-common.sh" "$BATS_TEST_TMPDIR/hardware-test-common.sh"
    cp "$SCRIPTS_DIR/run-hardware-lit.sh" "$SCRIPT"
    PATH="$BIN:$PATH"
    unset TT_VISIBLE_DEVICES
    unset TT_METAL_CACHE
    unset HW_TEST_WORKERS
    write_fake_group_probe 1
}

write_fake_group_probe() {
    local minimum_group_size="$1"
    cat > "$BIN/python3" <<EOF
#!/usr/bin/env bash
[ "\${1:-}" = "-c" ] || exit 2
[ "\${3:-0}" -ge "$minimum_group_size" ]
EOF
    chmod +x "$BIN/python3"
}

write_fake_lit() {
    local exit_code="${1:-0}"
    cat > "$BIN/llvm-lit" <<EOF
#!/usr/bin/env bash
printf 'env:%s cache:%s exec_root:%s args:%s\n' \
    "\${TT_VISIBLE_DEVICES:-}" \
    "\${TT_METAL_CACHE:-}" \
    "\${TTLANG_LIT_TEST_EXEC_ROOT:-}" \
    "\$*" >> "$CALLS"
exit $exit_code
EOF
    chmod +x "$BIN/llvm-lit"
}

@test "multi-chip: worker cap limits lit concurrency" {
    write_fake_lit 0

    HW_LIT_CHIPS=4 HW_TEST_WORKERS=2 run "$SCRIPT" \
        build/test/python build/test/python-lit-report

    assert_success
    assert_output --partial "Detected 4 chips as 4 valid device groups: 2 Python lit shards in parallel"
    run cat "$CALLS"
    assert_line --partial "env:0"
    assert_line --partial "--num-shards 2 --run-shard 1"
    assert_line --partial "env:1"
    assert_line --partial "--num-shards 2 --run-shard 2"
    refute_output --partial "env:2"
    [ "${#lines[@]}" -eq 3 ]
}

@test "multi-chip: runs one serial lit shard per chip, then multi-device serial" {
    write_fake_lit 0

    HW_LIT_CHIPS=3 run "$SCRIPT" build/test/python build/test/python-lit-report

    assert_success
    run cat "$CALLS"
    assert_line --partial "env:0 cache:$PWD/build/test/python-lit-report-tt-metal-cache/shard-1 exec_root:$PWD/build/test/python-lit-report-lit-exec/shard-1 args:build/test/python --num-shards 3 --run-shard 1"
    assert_line --partial "env:1 cache:$PWD/build/test/python-lit-report-tt-metal-cache/shard-2 exec_root:$PWD/build/test/python-lit-report-lit-exec/shard-2 args:build/test/python --num-shards 3 --run-shard 2"
    assert_line --partial "env:2 cache:$PWD/build/test/python-lit-report-tt-metal-cache/shard-3 exec_root:$PWD/build/test/python-lit-report-lit-exec/shard-3 args:build/test/python --num-shards 3 --run-shard 3"
    assert_line --partial "env: cache:$PWD/build/test/python-lit-report-tt-metal-cache/multidevice exec_root:$PWD/build/test/python-lit-report-lit-exec/multidevice args:build/test/python --filter mesh_tensor"
    assert_line --partial "python-lit-report-shard-1.xml"
    assert_line --partial "python-lit-report-multidevice.xml"
    [ "${#lines[@]}" -eq 4 ]
}

@test "multi-chip: paired-chip topology runs one shard per valid device group" {
    write_fake_lit 0
    write_fake_group_probe 2

    HW_LIT_CHIPS=4 run "$SCRIPT" \
        build/test/python build/test/python-lit-report

    assert_success
    assert_output --partial "Detected 4 chips as 2 valid device groups: 2 Python lit shards in parallel"
    run cat "$CALLS"
    assert_line --partial "env:0,1"
    assert_line --partial "--num-shards 2 --run-shard 1"
    assert_line --partial "env:2,3"
    assert_line --partial "--num-shards 2 --run-shard 2"
    assert_line --partial "env: cache:"
    assert_line --partial "python-lit-report-multidevice.xml"
    [ "${#lines[@]}" -eq 3 ]
}

@test "multi-chip: no valid device grouping runs once with full topology visibility" {
    write_fake_lit 0
    write_fake_group_probe 5

    TT_VISIBLE_DEVICES=2 HW_LIT_CHIPS=4 run "$SCRIPT" \
        build/test/python build/test/python-lit-report

    assert_success
    assert_output --partial "No valid device grouping found"
    run cat "$CALLS"
    assert_output --partial "env:"
    assert_output --partial "python-lit-report.xml"
    refute_output --partial "env:2"
    refute_output --partial "--num-shards"
    [ "${#lines[@]}" -eq 1 ]
}

@test "single chip: one serial lit run over the whole suite" {
    write_fake_lit 0

    HW_LIT_CHIPS=1 run "$SCRIPT" build/test/python build/test/python-lit-report

    assert_success
    run cat "$CALLS"
    assert_output --partial "exec_root:$PWD/build/test/python-lit-report-lit-exec/serial"
    assert_output --partial "args:build/test/python -j1 --verbose"
    assert_output --partial "python-lit-report.xml"
    refute_output --partial "--num-shards"
    [ "${#lines[@]}" -eq 1 ]
}

@test "multi-chip: custom multi-device filter is forwarded" {
    write_fake_lit 0

    HW_LIT_CHIPS=2 HW_LIT_MULTI_DEVICE_FILTER=fabric_mesh run "$SCRIPT" build/test/python build/test/python-lit-report

    assert_success
    run cat "$CALLS"
    assert_output --partial "--filter-out fabric_mesh"
    assert_output --partial "--filter fabric_mesh"
}

@test "multi-chip: existing TT_METAL_CACHE is used as cache root" {
    write_fake_lit 0

    HW_LIT_CHIPS=2 TT_METAL_CACHE="$BATS_TEST_TMPDIR/cache-root" run "$SCRIPT" build/test/python build/test/python-lit-report

    assert_success
    run cat "$CALLS"
    assert_line --partial "cache:$BATS_TEST_TMPDIR/cache-root/shard-1"
    assert_line --partial "cache:$BATS_TEST_TMPDIR/cache-root/shard-2"
    assert_line --partial "cache:$BATS_TEST_TMPDIR/cache-root/multidevice"
}

@test "multi-chip: shard failure is reported after all shards run" {
    write_fake_lit 1

    HW_LIT_CHIPS=2 run "$SCRIPT" build/test/python build/test/python-lit-report

    assert_failure
    run cat "$CALLS"
    [ "${#lines[@]}" -eq 3 ]
}

@test "rejects invalid chip override" {
    write_fake_lit 0

    HW_LIT_CHIPS=abc run "$SCRIPT" build/test/python build/test/python-lit-report

    assert_failure 2
    assert_output --partial "chip count"
}

@test "rejects invalid worker override" {
    write_fake_lit 0

    HW_LIT_CHIPS=4 HW_TEST_WORKERS=0 run "$SCRIPT" \
        build/test/python build/test/python-lit-report
    assert_failure 2
    assert_output --partial "worker count must be a positive integer"

    HW_LIT_CHIPS=4 HW_TEST_WORKERS=5 run "$SCRIPT" \
        build/test/python build/test/python-lit-report
    assert_failure 2
    assert_output --partial "worker count 5 exceeds chip count 4"
}
