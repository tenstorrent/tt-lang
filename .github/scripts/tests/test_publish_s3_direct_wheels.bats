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
# `s3 ls` is used to regenerate the parent listing; return the SHA dir.
if [[ "$1 $2" == "s3 ls" ]]; then
    case "$3" in
        *tt-lang/ttmetal/) cat "$LS_PARENT" ;;
    esac
elif [[ "$1 $2" == "s3api put-object" ]]; then
    # Capture each uploaded index body under a filename derived from --key so
    # tests can inspect its HTML content (the script deletes its tmpfile on exit).
    shift 2
    key="" body=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --key) key="$2"; shift 2 ;;
            --body) body="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    cp "$body" "$PUT_BODIES_DIR/${key//\//_}"
fi
EOF
    chmod +x "$bindir/aws"
    export PATH="$bindir:$PATH"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/publish-s3-direct-wheels.sh"
    FAKE_AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    LS_PARENT="$BATS_TEST_TMPDIR/ls_parent"
    PUT_BODIES_DIR="$BATS_TEST_TMPDIR/put_bodies"
    mkdir -p "$PUT_BODIES_DIR"
    : > "$FAKE_AWS_ARGS"
    printf '                           PRE 13adda8/\n' > "$LS_PARENT"
    export FAKE_AWS_ARGS LS_PARENT PUT_BODIES_DIR
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
    # No re-download of an uploaded wheel to compute its hash
    refute_output --regexp 's3 cp s3://[^[:space:]]*\.whl -'
}

@test "per-SHA index HTML is hashed from local wheels, not downloaded from S3" {
    empty_digest="$(sha256sum /dev/null | awk '{print $1}')"
    dir=$(make_wheel_dir "$(whl_light_core_cp312 1.0.0)" "$(whl_light 1.0.0)")
    run -0 "$SCRIPT" --prefix tt-lang/ttmetal/13adda8 --readme "$README" "$dir"

    run cat "$PUT_BODIES_DIR/tt-lang_ttmetal_13adda8_"
    assert_output --partial "<a href=\"README.txt\">README.txt</a><br>"
    assert_output --partial "<a href=\"$(whl_light_core_cp312 1.0.0)#sha256=$empty_digest\">$(whl_light_core_cp312 1.0.0)</a><br>"
    assert_output --partial "<a href=\"$(whl_light 1.0.0)#sha256=$empty_digest\">$(whl_light 1.0.0)</a><br>"
}

@test "bucket is overridable via TTLANG_S3_BUCKET" {
    dir=$(make_wheel_dir "$(whl_light 1.0.0)")
    TTLANG_S3_BUCKET=other-bucket run -0 "$SCRIPT" --prefix tt-lang/ttmetal/13adda8 --readme "$README" "$dir"
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3://other-bucket/tt-lang/ttmetal/13adda8/$(whl_light 1.0.0)"
}
