#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/publish-s3-wheels.sh.

load test_helper

# Install a fake `s3pypi` on PATH that records its full argv to
# $FAKE_S3PYPI_ARGS (one invocation per line). Echoes the bindir.
make_s3pypi_mock() {
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/s3pypi" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_S3PYPI_ARGS"
exit 0
EOF
    chmod +x "$bindir/s3pypi"
    echo "$bindir"
}

# Create a temp dist dir containing the named (empty) wheel files. Echoes the
# dir path.
make_dist_dir() {
    local dir
    dir=$(mktemp -d "$BATS_TEST_TMPDIR/dist.XXXXXX")
    for name in "$@"; do
        : > "$dir/$name"
    done
    echo "$dir"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/publish-s3-wheels.sh"
    FAKE_S3PYPI_ARGS="$BATS_TEST_TMPDIR/s3pypi_args"
    : > "$FAKE_S3PYPI_ARGS"
    export FAKE_S3PYPI_ARGS
    BINDIR=$(make_s3pypi_mock)
    export PATH="$BINDIR:$PATH"
}

@test "no arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT"
}

@test "too many arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT" dist extra
}

@test "empty dist dir -> error (exit 1)" {
    dir=$(make_dist_dir)
    run -1 "$SCRIPT" "$dir"
    assert_output --partial "No wheels found under $dir"
}

@test "single wheel uploaded with default flags" {
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    run -0 "$SCRIPT" "$dir"

    run cat "$FAKE_S3PYPI_ARGS"
    assert_output --partial "upload $dir/tt_lang-1.0-py3-none-any.whl"
    assert_output --partial "--put-root-index"
    assert_output --partial "--bucket tenstorrent-pypi"
    refute_output --partial "--force"
}

@test "multiple wheels each get their own s3pypi invocation" {
    dir=$(make_dist_dir \
        "tt_lang-1.0-py3-none-any.whl" \
        "tt_lang_sim-1.0-py3-none-any.whl")
    run -0 "$SCRIPT" "$dir"

    run wc -l < "$FAKE_S3PYPI_ARGS"
    assert_output "2"

    run cat "$FAKE_S3PYPI_ARGS"
    assert_output --partial "tt_lang-1.0-py3-none-any.whl"
    assert_output --partial "tt_lang_sim-1.0-py3-none-any.whl"
}

@test "--overwrite adds --force flag" {
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    run -0 "$SCRIPT" --overwrite "$dir"

    run cat "$FAKE_S3PYPI_ARGS"
    assert_output --partial "--force"
}

@test "s3pypi failure aborts and propagates" {
    cat > "$BINDIR/s3pypi" <<'EOF'
#!/usr/bin/env bash
exit 3
EOF
    chmod +x "$BINDIR/s3pypi"
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    run "$SCRIPT" "$dir"
    assert_equal "$status" 3
}
