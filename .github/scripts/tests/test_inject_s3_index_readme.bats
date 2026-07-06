#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/inject_s3_index_readme.py.

load test_helper

# Write a minimal root index (a <body> with two package anchors) to $INDEX and
# a short README to $README.
setup() {
    SCRIPT="$SCRIPTS_DIR/inject_s3_index_readme.py"
    INDEX="$BATS_TEST_TMPDIR/index.html"
    README="$BATS_TEST_TMPDIR/README.md"
    cat > "$INDEX" <<'EOF'
<!DOCTYPE html>
<html>
<body>
<a href="tt-lang/">tt-lang</a>
<a href="tt-lang-light/">tt-lang-light</a>
</body>
</html>
EOF
    cat > "$README" <<'EOF'
# tt-lang S3 index

Install with `pip install tt-lang`.
EOF
}

run_inject() { python3 "$SCRIPT" "$README" "$INDEX"; }

@test "wrong arg count -> usage error (exit 2)" {
    run -2 python3 "$SCRIPT" "$README"
}

@test "injects the README block above the package anchor links" {
    run -0 run_inject
    run cat "$INDEX"
    assert_output --partial 'id="ttlang-s3-readme"'
    assert_output --partial "tt-lang S3 index"
    # The injected section must appear before the first package anchor.
    section_line=$(grep -n 'ttlang-s3-readme' "$INDEX" | head -1 | cut -d: -f1)
    anchor_line=$(grep -n 'href="tt-lang/"' "$INDEX" | head -1 | cut -d: -f1)
    [ "$section_line" -lt "$anchor_line" ]
}

@test "package anchor links are preserved" {
    run -0 run_inject
    run cat "$INDEX"
    assert_output --partial 'href="tt-lang/"'
    assert_output --partial 'href="tt-lang-light/"'
}

@test "re-running replaces rather than duplicating the block" {
    run -0 run_inject
    run -0 run_inject
    run grep -c 'id="ttlang-s3-readme"' "$INDEX"
    assert_output "1"
    run grep -c 'ttlang-s3-readme:start' "$INDEX"
    assert_output "1"
}

@test "index without a <body> gets the block prepended" {
    printf '<a href="tt-lang/">tt-lang</a>\n' > "$INDEX"
    run -0 run_inject
    run cat "$INDEX"
    assert_line --index 0 "$(grep -m1 . <<<'<!-- ttlang-s3-readme:start -->')"
    assert_output --partial 'href="tt-lang/"'
}

@test "--create-from-dist builds a missing root index from wheel names" {
    rm "$INDEX"
    dist_dir=$(make_wheel_dir \
        "tt_lang-1.0.0-py3-none-any.whl" \
        "tt_lang_light-1.0.0-py3-none-any.whl" \
        "tt_lang_sim-1.0.0-py3-none-any.whl")
    run -0 python3 "$SCRIPT" --create-from-dist "$dist_dir" "$README" "$INDEX"
    run cat "$INDEX"
    assert_output --partial 'href="tt-lang/"'
    assert_output --partial 'href="tt-lang-light/"'
    assert_output --partial 'href="tt-lang-sim/"'
    assert_output --partial 'id="ttlang-s3-readme"'
}

@test "--render-readme-html writes a standalone HTML README" {
    rm "$INDEX"
    run -0 python3 "$SCRIPT" --render-readme-html "$README" "$INDEX"
    run cat "$INDEX"
    assert_output --partial "<!DOCTYPE html>"
    assert_output --partial '<section id="ttlang-s3-readme">'
    assert_output --partial "tt-lang S3 index"
    refute_output --partial 'href="tt-lang/"'
}
