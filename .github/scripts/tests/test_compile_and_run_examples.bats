#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/compile-and-run-examples.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/compile-and-run-examples.sh"
    BIN="$BATS_TEST_TMPDIR/bin"
    REPO="$BATS_TEST_TMPDIR/repo"
    CALLS="$BATS_TEST_TMPDIR/python.calls"
    mkdir -p "$BIN" "$REPO/examples/tutorial"
    cp "$SCRIPT" "$BATS_TEST_TMPDIR/compile-and-run-examples.sh"
    SCRIPT="$BATS_TEST_TMPDIR/compile-and-run-examples.sh"
    PATH="$BIN:$PATH"
    unset TT_VISIBLE_DEVICES
    unset HW_SERIAL_TEST_VISIBLE_DEVICES
}

write_fake_python() {
    cat > "$BIN/python3" <<EOF
#!/usr/bin/env bash
printf 'args:%s vis:%s\n' "\$*" "\${TT_VISIBLE_DEVICES:-}" >> "$CALLS"
exit 0
EOF
    chmod +x "$BIN/python3"
}

write_example() {
    local path="$1"
    mkdir -p "$(dirname "$path")"
    cat > "$path" <<'EOF'
import ttl

@ttl.operation
def add(in0, in1, out):
    pass
EOF
}

@test "serial visibility override applies to each example subprocess" {
    write_fake_python
    write_example "$REPO/examples/eltwise_add.py"
    write_example "$REPO/examples/tutorial/broadcast.py"

    HW_SERIAL_TEST_VISIBLE_DEVICES=0,1 run bash "$SCRIPT" "$REPO"

    assert_success
    assert_output --partial "Restricting hardware examples to TT_VISIBLE_DEVICES=0,1"
    run cat "$CALLS"
    assert_line --partial "args:examples/eltwise_add.py vis:0,1"
    assert_line --partial "args:examples/tutorial/broadcast.py vis:0,1"
    [ "${#lines[@]}" -eq 2 ]
}
