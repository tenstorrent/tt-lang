#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/inject-s3-index-readme.sh.

load test_helper

# Mock aws stores objects in a flat key->file map under $S3_ROOT (a key ending
# in "/" is a valid S3 key but not a valid directory-free file path, so the
# map replaces "/" with "__" rather than mkdir -p'ing the key).
setup() {
    SCRIPT="$SCRIPTS_DIR/inject-s3-index-readme.sh"
    README="$BATS_TEST_TMPDIR/README.md"
    DIST_DIR=$(make_wheel_dir \
        "tt_lang-1.0.0-py3-none-any.whl" \
        "tt_lang_light-1.0.0-py3-none-any.whl")
    AWS_LOG="$BATS_TEST_TMPDIR/aws.log"
    S3_ROOT="$BATS_TEST_TMPDIR/s3"
    mkdir -p "$S3_ROOT"
    cat > "$README" <<'EOF'
# tt-lang S3 index
EOF
    bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$AWS_LOG"
verb="$1 $2"
shift 2
bucket=""
key=""
body=""
outfile=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket)
            bucket="$2"
            shift 2
            ;;
        --key)
            key="$2"
            shift 2
            ;;
        --body)
            body="$2"
            shift 2
            ;;
        --content-type)
            shift 2
            ;;
        -*)
            shift
            ;;
        *)
            outfile="$1"
            shift
            ;;
    esac
done
keyfile="$S3_ROOT/$(printf '%s' "$key" | sed 's#/#__#g')"
case "$verb" in
    "s3api get-object")
        if [[ -f "$keyfile" ]]; then
            cp "$keyfile" "$outfile"
            echo '{"ContentLength":0}'
        else
            echo "An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist." >&2
            exit 1
        fi
        ;;
    "s3api put-object")
        cp "$body" "$keyfile"
        ;;
    *)
        echo "unexpected aws command: $verb" >&2
        exit 2
        ;;
esac
EOF
    chmod +x "$bindir/aws"
    export PATH="$bindir:$PATH"
    export AWS_LOG S3_ROOT
}

@test "missing prefixed root index is created from dist and uploaded" {
    run -0 "$SCRIPT" --key tt-lang/2026-07/ --readme "$README" --dist-dir "$DIST_DIR"

    run cat "$S3_ROOT/tt-lang__2026-07__"
    assert_output --partial 'href="tt-lang/"'
    assert_output --partial 'href="tt-lang-light/"'
    assert_output --partial 'id="ttlang-s3-readme"'

    run cat "$S3_ROOT/tt-lang__2026-07"
    assert_output --partial 'href="tt-lang/"'
    assert_output --partial 'id="ttlang-s3-readme"'

    run cat "$AWS_LOG"
    assert_output --partial 's3api put-object --bucket tenstorrent-pypi --key tt-lang/2026-07/'
    assert_output --partial 's3api put-object --bucket tenstorrent-pypi --key tt-lang/2026-07 --body'
}

@test "existing root index is preserved and uploaded" {
    cat > "$S3_ROOT/tt-lang__2026-07__" <<'EOF'
<!DOCTYPE html>
<html>
<body>
<a href="existing-package/">existing-package</a>
</body>
</html>
EOF

    run -0 "$SCRIPT" --key tt-lang/2026-07/ --readme "$README" --dist-dir "$DIST_DIR"

    run cat "$S3_ROOT/tt-lang__2026-07__"
    assert_output --partial 'href="existing-package/"'
    assert_output --partial 'id="ttlang-s3-readme"'

    run cat "$S3_ROOT/tt-lang__2026-07"
    assert_output --partial 'href="existing-package/"'
    assert_output --partial 'id="ttlang-s3-readme"'
}

@test "non-404 aws failure is propagated" {
    cat > "$BATS_TEST_TMPDIR/bin/aws" <<'EOF'
#!/usr/bin/env bash
echo "AccessDenied" >&2
exit 1
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/aws"

    run "$SCRIPT" --key tt-lang/2026-07/ --readme "$README" --dist-dir "$DIST_DIR"
    assert_equal "$status" 1
    assert_output --partial "AccessDenied"
}

@test "--require-existing fails instead of creating a missing index" {
    run "$SCRIPT" --key tt-lang/ --require-existing --readme "$README" --dist-dir "$DIST_DIR"
    assert_equal "$status" 1
    assert_output --partial "does not exist"
}

@test "no-prefix stable index round-trips at the exact index.html key" {
    run -0 "$SCRIPT" --key index.html --readme "$README" --dist-dir "$DIST_DIR"

    run cat "$S3_ROOT/index.html"
    assert_output --partial 'href="tt-lang/"'
    assert_output --partial 'href="tt-lang-light/"'
    assert_output --partial 'id="ttlang-s3-readme"'

    run cat "$AWS_LOG"
    assert_output --partial 's3api put-object --bucket tenstorrent-pypi --key index.html'
}
