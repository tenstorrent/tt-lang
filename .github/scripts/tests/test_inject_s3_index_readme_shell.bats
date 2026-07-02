#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/inject-s3-index-readme.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/inject-s3-index-readme.sh"
    README="$BATS_TEST_TMPDIR/README.md"
    DIST_DIR=$(make_wheel_dir \
        "tt_lang-1.0.0-py3-none-any.whl" \
        "tt_lang_light-1.0.0-py3-none-any.whl")
    AWS_LOG="$BATS_TEST_TMPDIR/aws.log"
    S3_ROOT="$BATS_TEST_TMPDIR/s3"
    mkdir -p "$S3_ROOT/2026-07"
    cat > "$README" <<'EOF'
# tt-lang internal index
EOF
    bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$AWS_LOG"
if [[ "$1 $2" != "s3 cp" ]]; then
    echo "unexpected aws command" >&2
    exit 2
fi
src="$3"
dst="$4"
if [[ "$src" == s3://* ]]; then
    key="${src#s3://tenstorrent-pypi/}"
    source_path="$S3_ROOT/$key"
    if [[ ! -f "$source_path" ]]; then
        echo "fatal error: An error occurred (404) when calling the HeadObject operation: Key \"$key\" does not exist" >&2
        exit 1
    fi
    cp "$source_path" "$dst"
else
    key="${dst#s3://tenstorrent-pypi/}"
    mkdir -p "$(dirname "$S3_ROOT/$key")"
    cp "$src" "$S3_ROOT/$key"
fi
EOF
    chmod +x "$bindir/aws"
    export PATH="$bindir:$PATH"
    export AWS_LOG S3_ROOT
}

@test "missing prefixed root index is created from dist and uploaded" {
    run -0 "$SCRIPT" --key 2026-07/index.html --readme "$README" --dist-dir "$DIST_DIR"

    run cat "$S3_ROOT/2026-07/index.html"
    assert_output --partial 'href="tt-lang/"'
    assert_output --partial 'href="tt-lang-light/"'
    assert_output --partial 'id="ttlang-s3-readme"'
}

@test "existing root index is preserved and uploaded" {
    cat > "$S3_ROOT/2026-07/index.html" <<'EOF'
<!DOCTYPE html>
<html>
<body>
<a href="existing-package/">existing-package</a>
</body>
</html>
EOF

    run -0 "$SCRIPT" --key 2026-07/index.html --readme "$README" --dist-dir "$DIST_DIR"

    run cat "$S3_ROOT/2026-07/index.html"
    assert_output --partial 'href="existing-package/"'
    assert_output --partial 'id="ttlang-s3-readme"'
}

@test "non-404 aws download failure is propagated" {
    cat > "$BATS_TEST_TMPDIR/bin/aws" <<'EOF'
#!/usr/bin/env bash
echo "AccessDenied" >&2
exit 1
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/aws"

    run "$SCRIPT" --key 2026-07/index.html --readme "$README" --dist-dir "$DIST_DIR"
    assert_equal "$status" 1
    assert_output --partial "AccessDenied"
}
