#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/publish-s3-direct-wheels.sh.

load test_helper

make_aws_mock() {
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
# `s3 ls` is used to regenerate parent/child listings; return the wheels + readme
# for the SHA prefix, and the SHA dir for the parent prefix.
if [[ "$1 $2" == "s3 ls" ]]; then
    case "$3" in
        *tt-lang/ttmetal/13adda8/) cat "$LS_SHA" ;;
        *tt-lang/ttmetal/)         cat "$LS_PARENT" ;;
    esac
fi
EOF
    chmod +x "$bindir/aws"
    export PATH="$bindir:$PATH"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/publish-s3-direct-wheels.sh"
    FAKE_AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    LS_SHA="$BATS_TEST_TMPDIR/ls_sha"
    LS_PARENT="$BATS_TEST_TMPDIR/ls_parent"
    : > "$FAKE_AWS_ARGS"
    printf '2026-07-04 00:00:00 321 README.txt\n2026-07-04 00:00:00 1 %s\n' \
        "$(whl_light 1.0.0)" > "$LS_SHA"
    printf '                           PRE 13adda8/\n' > "$LS_PARENT"
    export FAKE_AWS_ARGS LS_SHA LS_PARENT
    README="$BATS_TEST_TMPDIR/README.md"
    printf '# tt-lang per-SHA wheels\n' > "$README"
    make_aws_mock
}

@test "missing prefix -> usage error (exit 2)" {
    dir=$(make_wheel_dir "$(whl_light 1.0.0)")
    run -2 "$SCRIPT" "$dir"
}

@test "empty dist dir -> error (exit 1)" {
    dir=$(make_wheel_dir)
    run -1 "$SCRIPT" --prefix tt-lang/ttmetal/13adda8 "$dir"
    assert_output --partial "No wheels found under $dir"
}

@test "uploads wheels+README as objects and regenerates slash-key listings" {
    dir=$(make_wheel_dir "$(whl_light_core_cp312 1.0.0)" "$(whl_light 1.0.0)")
    run -0 "$SCRIPT" --prefix tt-lang/ttmetal/13adda8 --readme "$README" "$dir"

    run cat "$FAKE_AWS_ARGS"
    # wheels + README uploaded as objects
    assert_output --partial "s3 cp $dir/$(whl_light 1.0.0) s3://tenstorrent-pypi/tt-lang/ttmetal/13adda8/$(whl_light 1.0.0)"
    assert_output --partial "s3://tenstorrent-pypi/tt-lang/ttmetal/13adda8/README.txt"
    # slash-key listing for the SHA prefix and the parent, via put-object
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/ttmetal/13adda8/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/ttmetal/"
    # NEVER writes an index.html object
    refute_output --partial "index.html"
}

@test "bucket is overridable via TTLANG_S3_BUCKET" {
    dir=$(make_wheel_dir "$(whl_light 1.0.0)")
    TTLANG_S3_BUCKET=other-bucket run -0 "$SCRIPT" --prefix tt-lang/ttmetal/13adda8 --readme "$README" "$dir"
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3://other-bucket/tt-lang/ttmetal/13adda8/$(whl_light 1.0.0)"
}
