#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/prepare-exabox-workspace.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/prepare-exabox-workspace.sh"
    GITHUB_WORKSPACE="$BATS_TEST_TMPDIR/source"
    STAGE_DIR="$BATS_TEST_TMPDIR/shared/tt-lang"
    WORKER_SRC="$BATS_TEST_TMPDIR/worker/tt-lang"
    EXABOX_REPORT_DIR="$BATS_TEST_TMPDIR/shared/reports"
    MPIRUN_CALLS="$BATS_TEST_TMPDIR/mpirun.calls"
    export GITHUB_WORKSPACE STAGE_DIR WORKER_SRC EXABOX_REPORT_DIR MPIRUN_CALLS
    unset CCACHE_DIR

    mkdir -p "$GITHUB_WORKSPACE/.github/scripts" "$BATS_TEST_TMPDIR/bin"
    cp "$SCRIPT" "$GITHUB_WORKSPACE/.github/scripts/"
    chmod +x "$GITHUB_WORKSPACE/.github/scripts/prepare-exabox-workspace.sh"
    echo source > "$GITHUB_WORKSPACE/source.txt"

    cat > "$BATS_TEST_TMPDIR/bin/mpirun" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" > "$MPIRUN_CALLS"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --pernode | --tag-output) shift ;;
        --bind-to) shift 2 ;;
        *) break ;;
    esac
done
exec "$@"
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/mpirun"
    PATH="$BATS_TEST_TMPDIR/bin:$PATH"
    export PATH
}

@test "stage makes the restored ccache writable by worker UID 1001" {
    mkdir -p "$BATS_TEST_TMPDIR/shared/ccache"
    echo cached > "$BATS_TEST_TMPDIR/shared/ccache/object"
    chmod 0700 "$BATS_TEST_TMPDIR/shared/ccache"
    chmod 0600 "$BATS_TEST_TMPDIR/shared/ccache/object"

    CCACHE_DIR="$BATS_TEST_TMPDIR/shared/ccache" run -0 "$SCRIPT" stage

    [ "$(stat -c '%a' "$BATS_TEST_TMPDIR/shared/ccache")" = 777 ]
    [ "$(stat -c '%a' "$BATS_TEST_TMPDIR/shared/ccache/object")" = 666 ]
}

@test "stage copies the checkout through shared storage to the worker" {
    mkdir -p "$WORKER_SRC"
    echo stale > "$WORKER_SRC/stale.txt"

    run -0 "$SCRIPT" stage

    assert_output --partial "workspace installed at $WORKER_SRC"
    [ -f "$STAGE_DIR/source.txt" ]
    [ -f "$WORKER_SRC/source.txt" ]
    [ ! -e "$WORKER_SRC/stale.txt" ]
    [ "$(stat -c '%a' "$EXABOX_REPORT_DIR")" = 777 ]
    run cat "$MPIRUN_CALLS"
    assert_output "--pernode --bind-to none --tag-output bash $STAGE_DIR/.github/scripts/prepare-exabox-workspace.sh install"
}

@test "install does not require the CPU runner workspace" {
    mkdir -p "$STAGE_DIR"
    echo staged > "$STAGE_DIR/staged.txt"
    unset GITHUB_WORKSPACE

    run -0 "$SCRIPT" install

    [ -f "$WORKER_SRC/staged.txt" ]
}

@test "destructive targets reject the filesystem root" {
    STAGE_DIR=/ run -2 "$SCRIPT" install
    assert_output --partial "STAGE_DIR must not be /"

    WORKER_SRC=/ run -2 "$SCRIPT" install
    assert_output --partial "WORKER_SRC must not be /"

    EXABOX_REPORT_DIR=/ run -2 "$SCRIPT" stage
    assert_output --partial "EXABOX_REPORT_DIR must not be /"

    CCACHE_DIR=/ run -2 "$SCRIPT" stage
    assert_output --partial "CCACHE_DIR must not be /"
}

@test "stage and worker directories must differ" {
    WORKER_SRC="$STAGE_DIR" run -2 "$SCRIPT" install
    assert_output --partial "STAGE_DIR and WORKER_SRC must differ"
}
