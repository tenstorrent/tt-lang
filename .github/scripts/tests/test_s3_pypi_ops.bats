#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/s3-pypi-ops.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/s3-pypi-ops.sh"
    FAKE_AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    : > "$FAKE_AWS_ARGS"
    export FAKE_AWS_ARGS
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
EOF
    chmod +x "$bindir/aws"
    export PATH="$bindir:$PATH"
}

@test "rejects a prefix outside the tt-lang allowlist" {
    run -2 "$SCRIPT" --operation delete --prefix ttnn/foo --confirm ttnn/foo --dry-run false
    assert_output --partial "not in the tt-lang allowlist"
}

@test "rejects the bucket root and bare allowlist root for delete" {
    run -2 "$SCRIPT" --operation delete --prefix tt-lang/ --confirm tt-lang/ --dry-run false
    assert_output --partial "refusing destructive op"
}

@test "rejects shell metacharacters in a prefix" {
    run -2 "$SCRIPT" --operation inspect --prefix 'tt-lang/x;rm -rf /'
    assert_output --partial "invalid characters"
}

@test "delete requires a matching confirm token" {
    run -2 "$SCRIPT" --operation delete --prefix tt-lang/ttmetal/dead --confirm wrong --dry-run false
    assert_output --partial "confirm"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "rm"
}

@test "dry-run prints the plan and performs no writes" {
    run -0 "$SCRIPT" --operation move --source tt-lang/13adda8 --dest tt-lang/ttmetal/13adda8
    assert_output --partial "DRY-RUN"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "mv"
}

@test "move (dry-run false) copies+deletes and reindexes" {
    run -0 "$SCRIPT" --operation move --source tt-lang/13adda8 --dest tt-lang/ttmetal/13adda8 --dry-run false
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 mv s3://tenstorrent-pypi/tt-lang/13adda8/ s3://tenstorrent-pypi/tt-lang/ttmetal/13adda8/ --recursive"
}

@test "readonly-cmd allows ls but rejects rm" {
    run -0 "$SCRIPT" --operation readonly-cmd -- s3 ls s3://tenstorrent-pypi/tt-lang/
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 ls s3://tenstorrent-pypi/tt-lang/"

    run -2 "$SCRIPT" --operation readonly-cmd -- s3 rm s3://tenstorrent-pypi/tt-lang/x
    assert_output --partial "read-only"
}
