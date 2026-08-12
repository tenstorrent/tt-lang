#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/reset-tt-cards.sh.

load test_helper

# Fake tt-smi. Each subcommand's exit status comes from FAKE_TT_SMI_<NAME>_FAIL;
# argv goes to $FAKE_TT_SMI_ARGS.
make_tt_smi_mock() {
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/tt-smi" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_TT_SMI_ARGS"
case "$1" in
    -glx_reset) exit "${FAKE_TT_SMI_GLX_FAIL:-0}" ;;
    -r)         exit "${FAKE_TT_SMI_R_FAIL:-0}" ;;
    --snapshot_no_tty) exit "${FAKE_TT_SMI_SNAPSHOT_FAIL:-0}" ;;
esac
exit 0
EOF
    chmod +x "$bindir/tt-smi"
    echo "$bindir"
}

setup() {
    SCRIPT="$BATS_TEST_DIRNAME/../reset-tt-cards.sh"
    FAKE_TT_SMI_ARGS="$BATS_TEST_TMPDIR/tt-smi-args"
    : > "$FAKE_TT_SMI_ARGS"
    export FAKE_TT_SMI_ARGS
    PATH="$(make_tt_smi_mock):$PATH"
    export PATH
    export TT_RESET_RETRY_SECONDS=0
}

@test "glx reset succeeds and health check passes -> exit 0" {
    run -0 "$SCRIPT"
    assert_output --partial "reset and health check passed on attempt 1"
    grep -q -- "-glx_reset" "$FAKE_TT_SMI_ARGS"
    grep -q -- "--snapshot_no_tty" "$FAKE_TT_SMI_ARGS"
}

@test "glx reset unsupported -> falls back to tt-smi -r" {
    FAKE_TT_SMI_GLX_FAIL=1 run -0 "$SCRIPT"
    grep -q '^-r' "$FAKE_TT_SMI_ARGS"
    assert_output --partial "reset and health check passed"
}

@test "successful glx reset does not also run tt-smi -r" {
    run -0 "$SCRIPT"
    run -1 grep -q '^-r$' "$FAKE_TT_SMI_ARGS"
}

@test "both resets fail -> retries then fails" {
    TT_RESET_MAX_ATTEMPTS=2 FAKE_TT_SMI_GLX_FAIL=1 FAKE_TT_SMI_R_FAIL=1 run -1 "$SCRIPT"
    assert_output --partial "Reset attempt 2 of 2"
    assert_output --partial "failed after 2 attempts"
}

@test "health check keeps failing -> non-zero exit" {
    TT_RESET_MAX_ATTEMPTS=2 FAKE_TT_SMI_SNAPSHOT_FAIL=1 run -1 "$SCRIPT"
    assert_output --partial "health check failed on attempt 1"
    assert_output --partial "failed after 2 attempts"
}

@test "health check recovers on a later attempt -> exit 0" {
    # Snapshot fails once, then succeeds.
    cat > "$BATS_TEST_TMPDIR/bin/tt-smi" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_TT_SMI_ARGS"
if [[ "$1" == "--snapshot_no_tty" ]]; then
    count=$(grep -c -- "--snapshot_no_tty" "$FAKE_TT_SMI_ARGS")
    [[ "$count" -lt 2 ]] && exit 1
fi
exit 0
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/tt-smi"
    TT_RESET_MAX_ATTEMPTS=3 run -0 "$SCRIPT"
    assert_output --partial "reset and health check passed on attempt 2"
}
