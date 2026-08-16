#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/run-exabox-hardware-phase.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/run-exabox-hardware-phase.sh"
    WORKER_SRC=/home/user/tt-lang
    MPIRUN_CALLS="$BATS_TEST_TMPDIR/mpirun.calls"
    export WORKER_SRC MPIRUN_CALLS

    mkdir -p "$BATS_TEST_TMPDIR/bin"
    cat > "$BATS_TEST_TMPDIR/bin/mpirun" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" > "$MPIRUN_CALLS"
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/mpirun"
    PATH="$BATS_TEST_TMPDIR/bin:$PATH"
    export PATH
}

@test "phase execution uses every Exabox worker" {
    run -0 "$SCRIPT" python-pytests

    run cat "$MPIRUN_CALLS"
    assert_output "--pernode --tag-output bash /home/user/tt-lang/.github/scripts/run-hardware-test-phase.sh python-pytests"
}

@test "worker source directory must be absolute" {
    WORKER_SRC=relative run -2 "$SCRIPT" smoketest
    assert_output --partial "WORKER_SRC must be absolute"
}

@test "phase is required" {
    run "$SCRIPT"
    assert_failure
    assert_output --partial "usage: run-exabox-hardware-phase.sh <phase>"
}
