#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/lib/s3-index.sh.

load test_helper

setup() {
    LIB="$SCRIPTS_DIR/lib/s3-index.sh"
    # shellcheck disable=SC1090
    source "$LIB"
    FAKE_AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    : > "$FAKE_AWS_ARGS"
    export FAKE_AWS_ARGS
}

# Install a mock `aws` on PATH. $1 is the stdout it prints for `s3 ls`.
make_aws_mock() {
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    printf '%s' "$1" > "$BATS_TEST_TMPDIR/ls_output"
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
if [[ "$1 $2" == "s3 ls" ]]; then
    cat "$BATS_TEST_TMPDIR/ls_output"
fi
EOF
    chmod +x "$bindir/aws"
    export PATH="$bindir:$PATH"
}

# Install a mock `aws` on PATH whose `s3 ls` invocation exits 1 (simulates a
# transient AWS failure: permissions, throttling, network).
make_aws_mock_failing() {
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
if [[ "$1 $2" == "s3 ls" ]]; then
    echo "mock aws error" >&2
    exit 1
fi
EOF
    chmod +x "$bindir/aws"
    export PATH="$bindir:$PATH"
}

@test "s3_render_index wraps anchors in a Package Index document" {
    run bash -c 'source "'"$LIB"'"; printf "%s\n" "<a href=\"a/\">a</a><br>" | s3_render_index "T"'
    assert_output --partial "<!DOCTYPE html>"
    assert_output --partial "<title>T</title>"
    assert_output --partial '<a href="a/">a</a><br>'
    assert_output --partial "</html>"
}

@test "s3_child_anchors renders sub-prefixes as dir anchors and objects as file anchors" {
    make_aws_mock "                           PRE 13adda8/
2026-07-04 00:00:00       1234 tt_lang_light-1.0.0-py3-none-any.whl
2026-07-04 00:00:00        321 README.txt
2026-07-04 00:00:00        200 index.html
2026-07-04 00:00:00         42 attempt.json
2026-07-04 00:00:00         10 junk.txt
"
    run s3_child_anchors tenstorrent-pypi tt-lang/ttmetal
    assert_line '<a href="13adda8/">13adda8/</a><br>'
    assert_line '<a href="tt_lang_light-1.0.0-py3-none-any.whl">tt_lang_light-1.0.0-py3-none-any.whl</a><br>'
    assert_line '<a href="README.txt">README.txt</a><br>'
    refute_output --partial '#sha256='
    refute_output --partial 'index.html'
    refute_output --partial 'attempt.json'
    refute_output --partial 'junk.txt'
}

@test "s3_child_anchors HTML-escapes names containing '&'" {
    make_aws_mock "2026-07-04 00:00:00       1234 a&b.whl
"
    run s3_child_anchors tenstorrent-pypi tt-lang/ttmetal
    assert_line '<a href="a&amp;b.whl">a&amp;b.whl</a><br>'
    refute_output --partial 'a&b.whl'
    refute_output --partial '#sha256='
}

@test "s3_child_anchors keeps a literal '+' in a light wheel name (not percent-encoded)" {
    make_aws_mock "2026-07-04 00:00:00       1234 tt_lang-1.0.0+light-cp312-cp312-manylinux_2_34_x86_64.whl
"
    run s3_child_anchors tenstorrent-pypi tt-lang/ttmetal
    assert_success
    assert_line '<a href="tt_lang-1.0.0+light-cp312-cp312-manylinux_2_34_x86_64.whl">tt_lang-1.0.0+light-cp312-cp312-manylinux_2_34_x86_64.whl</a><br>'
    refute_output --partial "%2B"
}

@test "s3_put_index uploads to the slash-key via put-object" {
    make_aws_mock ""
    printf '<html></html>\n' > "$BATS_TEST_TMPDIR/idx.html"
    run s3_put_index tenstorrent-pypi tt-lang/ttmetal/13adda8 "$BATS_TEST_TMPDIR/idx.html"
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/ttmetal/13adda8/ --body $BATS_TEST_TMPDIR/idx.html --content-type text/html; charset=utf-8"
}

@test "s3_regenerate_index lists, renders, and puts the slash-key" {
    make_aws_mock "                           PRE 13adda8/
"
    run s3_regenerate_index tenstorrent-pypi tt-lang/ttmetal
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 ls s3://tenstorrent-pypi/tt-lang/ttmetal/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/ttmetal/"
}

@test "s3_regenerate_index writes a hash-less body from an S3-sourced listing" {
    CAPTURED_BODY="$BATS_TEST_TMPDIR/captured_body.html"
    export CAPTURED_BODY
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    printf '%s' "                           PRE 13adda8/
2026-07-04 00:00:00       1234 tt_lang_light-1.0.0-py3-none-any.whl
2026-07-04 00:00:00        321 README.txt
" > "$BATS_TEST_TMPDIR/ls_output"
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
if [[ "$1 $2" == "s3 ls" ]]; then
    cat "$BATS_TEST_TMPDIR/ls_output"
elif [[ "$1 $2" == "s3api put-object" ]]; then
    args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do
        if [[ "${args[$i]}" == "--body" ]]; then
            cp "${args[$((i + 1))]}" "$CAPTURED_BODY"
        fi
    done
fi
EOF
    chmod +x "$bindir/aws"
    export PATH="$bindir:$PATH"

    run s3_regenerate_index tenstorrent-pypi tt-lang/ttmetal
    assert_success
    [ -f "$CAPTURED_BODY" ]
    run cat "$CAPTURED_BODY"
    assert_output --partial "tt_lang_light-1.0.0-py3-none-any.whl"
    assert_output --partial "README.txt"
    refute_output --partial "#sha256="
}

@test "s3_child_anchors skips the slash-key's own listing line (empty name)" {
    make_aws_mock "2026-07-04 00:00:00        234
2026-07-04 00:00:00       1234 tt_lang_light-1.0.0-py3-none-any.whl
"
    run s3_child_anchors tenstorrent-pypi tt-lang/ttmetal
    assert_success
    assert_line '<a href="tt_lang_light-1.0.0-py3-none-any.whl">tt_lang_light-1.0.0-py3-none-any.whl</a><br>'
    refute_output --partial 'href="234"'
}

@test "s3_child_anchors fails when aws s3 ls fails" {
    make_aws_mock_failing
    run s3_child_anchors tenstorrent-pypi tt-lang/ttmetal
    assert_failure
}

@test "s3_regenerate_index refuses to write when aws s3 ls fails" {
    make_aws_mock_failing
    run s3_regenerate_index tenstorrent-pypi tt-lang/ttmetal
    assert_failure
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "put-object"
}

@test "s3_regenerate_index refuses to write when the listing is empty" {
    make_aws_mock "2026-07-04 00:00:00        234
"
    run s3_regenerate_index tenstorrent-pypi tt-lang/ttmetal
    assert_failure
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "put-object"
}

@test "s3_local_wheel_anchors hashes local wheels and always includes README.txt" {
    dist_dir="$BATS_TEST_TMPDIR/dist"
    mkdir -p "$dist_dir"
    : > "$dist_dir/tt_lang_light-1.0.0-py3-none-any.whl"
    : > "$dist_dir/tt_lang-1.0.0+light-cp312-cp312-manylinux_2_34_x86_64.whl"

    run s3_local_wheel_anchors "$dist_dir"
    assert_success
    assert_line '<a href="README.txt">README.txt</a><br>'

    digest="$(sha256sum "$dist_dir/tt_lang_light-1.0.0-py3-none-any.whl" | awk '{print $1}')"
    assert_line "<a href=\"tt_lang_light-1.0.0-py3-none-any.whl#sha256=$digest\">tt_lang_light-1.0.0-py3-none-any.whl</a><br>"
    assert_line --regexp '^<a href="tt_lang-1\.0\.0\+light-cp312-cp312-manylinux_2_34_x86_64\.whl#sha256=[0-9a-f]{64}">tt_lang-1\.0\.0\+light-cp312-cp312-manylinux_2_34_x86_64\.whl</a><br>$'
}

@test "s3_render_index HTML-escapes a title containing '&' and '\"'" {
    run bash -c 'source "'"$LIB"'"; printf "%s\n" "<a href=\"a/\">a</a><br>" | s3_render_index "AT&T \"quote\""'
    assert_success
    assert_output --partial '<title>AT&amp;T &quot;quote&quot;</title>'
    refute_output --partial '<title>AT&T'
}

@test "s3_child_anchors HTML-escapes a name containing '\"'" {
    make_aws_mock '2026-07-04 00:00:00       1234 weird"name.whl
'
    run s3_child_anchors tenstorrent-pypi tt-lang/ttmetal
    assert_success
    assert_line '<a href="weird&quot;name.whl">weird&quot;name.whl</a><br>'
}
