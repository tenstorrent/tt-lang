#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/refresh-s3-wheel-views.sh"
    CALLS="$BATS_TEST_TMPDIR/calls"
    LISTING="$BATS_TEST_TMPDIR/listing"
    bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/aws" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$CALLS"
if [ "$1 $2" = "s3 ls" ]; then
    cat "$LISTING"
fi
exit 0
EOF
    cat > "$bindir/inject" <<'EOF'
#!/bin/sh
printf 'inject %s\n' "$*" >> "$CALLS"
EOF
    chmod +x "$bindir/aws" "$bindir/inject"
    export PATH="$bindir:$PATH"
    export CALLS LISTING
    export INJECT_S3_INDEX_README="$bindir/inject"
    : > "$CALLS"
}

@test "refresh regenerates a month that still has wheels" {
    cat > "$LISTING" <<'EOF'
                           PRE releases/
2026-07-01 00:00:00       100 tt_lang-1.2.3.dev20260701-py3-none-any.whl
EOF

    run "$SCRIPT" --months 2026-07

    assert_success
    run cat "$CALLS"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/2026-07/"
    refute_output --partial "delete-object --bucket tenstorrent-pypi --key tt-lang/2026-07"
    assert_output --partial "inject --bucket tenstorrent-pypi --key tt-lang/ --require-existing"
}

@test "refresh removes an empty month view" {
    cat > "$LISTING" <<'EOF'
                           PRE releases/
EOF

    run "$SCRIPT" --months 2026-07

    assert_success
    run cat "$CALLS"
    assert_output --partial "delete-object --bucket tenstorrent-pypi --key tt-lang/2026-07"
    assert_output --partial "delete-object --bucket tenstorrent-pypi --key tt-lang/2026-07/"
    assert_output --partial "inject --bucket tenstorrent-pypi --key tt-lang/ --require-existing"
}

@test "refresh rejects malformed months before writing" {
    run "$SCRIPT" --months 202607

    assert_failure 2
    assert_output --partial "Invalid year-month"
    [[ ! -s "$CALLS" ]]
}

@test "refresh rejects out-of-range months before writing" {
    run "$SCRIPT" --months 2026-13

    assert_failure 2
    assert_output --partial "Invalid year-month"
    [[ ! -s "$CALLS" ]]
}
