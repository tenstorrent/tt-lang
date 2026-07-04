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
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
if [[ "$1 $2" != "s3 cp" ]]; then
    echo "unexpected aws command" >&2
    exit 2
fi
src="$3"
dst="$4"
if [[ "$dst" == s3://tenstorrent-pypi/tt-lang/13adda8/index.html ]]; then
    cp "$src" "$CAPTURED_INDEX"
fi
if [[ "$dst" == s3://tenstorrent-pypi/tt-lang/13adda8/README.txt ]]; then
    cp "$src" "$CAPTURED_README"
fi
EOF
    chmod +x "$bindir/aws"
    echo "$bindir"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/publish-s3-direct-wheels.sh"
    FAKE_AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    CAPTURED_INDEX="$BATS_TEST_TMPDIR/index.html"
    CAPTURED_README="$BATS_TEST_TMPDIR/README.capture.txt"
    README="$BATS_TEST_TMPDIR/README.md"
    : > "$FAKE_AWS_ARGS"
    cat > "$README" <<'EOF'
# tt-lang per-SHA wheels
EOF
    export FAKE_AWS_ARGS CAPTURED_INDEX CAPTURED_README
    BINDIR=$(make_aws_mock)
    export PATH="$BINDIR:$PATH"
}

@test "missing prefix -> usage error (exit 2)" {
    dir=$(make_wheel_dir "tt_lang-1.0.0-py3-none-any.whl")
    run -2 "$SCRIPT" "$dir"
}

@test "empty dist dir -> error (exit 1)" {
    dir=$(make_wheel_dir)
    run -1 "$SCRIPT" --prefix tt-lang/13adda8 "$dir"
    assert_output --partial "No wheels found under $dir"
}

@test "uploads wheels and direct index under tt-lang sha prefix" {
    dir=$(make_wheel_dir \
        "tt_lang-1.0.0+light-cp312-cp312-manylinux_2_34_x86_64.whl" \
        "tt_lang_light-1.0.0-py3-none-any.whl")

    run -0 "$SCRIPT" --prefix tt-lang/13adda8 --readme "$README" "$dir"

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 cp $dir/tt_lang-1.0.0+light-cp312-cp312-manylinux_2_34_x86_64.whl s3://tenstorrent-pypi/tt-lang/13adda8/tt_lang-1.0.0+light-cp312-cp312-manylinux_2_34_x86_64.whl --content-type application/octet-stream"
    assert_output --partial "s3 cp $dir/tt_lang_light-1.0.0-py3-none-any.whl s3://tenstorrent-pypi/tt-lang/13adda8/tt_lang_light-1.0.0-py3-none-any.whl --content-type application/octet-stream"
    assert_output --partial "s3://tenstorrent-pypi/tt-lang/13adda8/index.html --content-type text/html; charset=utf-8"
    assert_output --partial "s3://tenstorrent-pypi/tt-lang/13adda8/README.txt --content-type text/plain; charset=utf-8"

    run cat "$CAPTURED_INDEX"
    assert_output --partial 'id="ttlang-s3-readme"'
    assert_output --partial "tt-lang per-SHA wheels"
    assert_output --partial 'href="README.txt"'
    assert_output --partial 'href="tt_lang-1.0.0%2Blight-cp312-cp312-manylinux_2_34_x86_64.whl#sha256='
    assert_output --partial 'href="tt_lang_light-1.0.0-py3-none-any.whl#sha256='
    refute_output --partial 'href="tt-lang-light/"'
    readme_line=$(grep -n 'href="README.txt"' "$CAPTURED_INDEX" | head -1 | cut -d: -f1)
    wheel_line=$(grep -n 'href="tt_lang-' "$CAPTURED_INDEX" | head -1 | cut -d: -f1)
    [ "$readme_line" -lt "$wheel_line" ]

    run cat "$CAPTURED_README"
    assert_output --partial "tt-lang per-SHA wheels"
}

@test "bucket is overridable via TTLANG_S3_BUCKET" {
    dir=$(make_wheel_dir "tt_lang-1.0.0-py3-none-any.whl")
    TTLANG_S3_BUCKET=other-bucket run -0 "$SCRIPT" --prefix tt-lang/13adda8 "$dir"

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3://other-bucket/tt-lang/13adda8/tt_lang-1.0.0-py3-none-any.whl"
}
