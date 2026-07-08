#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/detect-ttmetal-uplift.sh. The script is
# sourced (its main() is guarded) so the S3 lookups can be overridden without
# real AWS access.

load test_helper

FULL_SHA="13adda80fef07a3c5d6f2f8b9a0c1d2e3f405162"
SHORT_SHA="13adda8"
HEAD_A="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
HEAD_B="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

make_version_file() {
    local file="$BATS_TEST_TMPDIR/tt-metal-version.$RANDOM"
    write_tt_metal_version_file "$file" \
        "$TEST_TTNN_PYPI_VERSION" \
        "$TEST_TT_METAL_TAG" \
        "$TEST_TT_METAL_TAG"
    echo "$file"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/detect-ttmetal-uplift.sh"
    GH_OUT="$BATS_TEST_TMPDIR/gh_out"
    : > "$GH_OUT"
    source "$SCRIPT"
    resolve_ttmetal_tag_sha() { printf '%s\n' "$FULL_SHA"; }
}

# --- read_target_ttmetal_sha ---

@test "read_target_ttmetal_sha resolves the pinned tag" {
    version_file=$(make_version_file)
    run -0 read_target_ttmetal_sha "$version_file"
    assert_output "$FULL_SHA"
}

@test "read_target_ttmetal_sha fails on a missing file" {
    run read_target_ttmetal_sha "$BATS_TEST_TMPDIR/nope.txt"
    assert_failure
    assert_output --partial "not found"
}

@test "read_target_ttmetal_sha fails without TT_METAL_TAG" {
    f="$BATS_TEST_TMPDIR/empty.txt"
    echo 'TTNN_PYPI="0.0.0"' > "$f"
    echo 'TTNN_PYPI_TT_METAL_TAG="v0.0.0"' >> "$f"
    run read_target_ttmetal_sha "$f"
    assert_failure
    assert_output --partial "TT_METAL_TAG not set"
}

# --- classify_target ---

@test "classify_target: no prefix -> new" {
    list_prefix_objects() { printf ''; }
    run -0 classify_target "$SHORT_SHA" "$HEAD_A"
    assert_output "new"
}

@test "classify_target: a published wheel -> published" {
    list_prefix_objects() { printf 'index.html\ntt_lang-1+light.whl\n'; }
    run -0 classify_target "$SHORT_SHA" "$HEAD_A"
    assert_output "published"
}

@test "classify_target: miss marker at the same tt-lang HEAD -> doomed" {
    list_prefix_objects() { printf 'attempt.json\n'; }
    read_recorded_head() { printf '%s' "$HEAD_A"; }
    run -0 classify_target "$SHORT_SHA" "$HEAD_A"
    assert_output "doomed"
}

@test "classify_target: miss marker at an older tt-lang HEAD -> retry" {
    list_prefix_objects() { printf 'attempt.json\n'; }
    read_recorded_head() { printf '%s' "$HEAD_A"; }
    run -0 classify_target "$SHORT_SHA" "$HEAD_B"
    assert_output "retry"
}

@test "default S3 object listing reads the tt-lang/ttmetal/<ttmetal7> prefix" {
    bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    export AWS_ARGS
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$AWS_ARGS"
printf '2026-07-04 00:00:00       1234 tt_lang-1.0.0+light.whl\n'
EOF
    chmod +x "$bindir/aws"
    PATH="$bindir:$PATH" run -0 list_prefix_objects "$SHORT_SHA"
    assert_output "tt_lang-1.0.0+light.whl"

    run cat "$AWS_ARGS"
    assert_output --partial "s3 ls s3://tenstorrent-pypi/tt-lang/ttmetal/$SHORT_SHA/"
}

@test "default miss marker read uses the tt-lang/ttmetal/<ttmetal7> prefix" {
    bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    export AWS_ARGS
    cat > "$bindir/aws" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "\$AWS_ARGS"
printf '{"ttlang_head":"%s"}\n' "$HEAD_A"
EOF
    chmod +x "$bindir/aws"
    PATH="$bindir:$PATH" run -0 read_recorded_head "$SHORT_SHA"
    assert_output "$HEAD_A"

    run cat "$AWS_ARGS"
    assert_output --partial "s3 cp s3://tenstorrent-pypi/tt-lang/ttmetal/$SHORT_SHA/attempt.json -"
}

# --- main ---

@test "main: new sha -> uplift=true with sha outputs" {
    version_file=$(make_version_file)
    list_prefix_objects() { printf ''; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --version-file "$version_file" --ttlang-head "$HEAD_A"
    run cat "$GH_OUT"
    assert_output --partial "uplift=true"
    assert_output --partial "tt_metal_sha=$FULL_SHA"
    assert_output --partial "tt_metal_sha_short=$SHORT_SHA"
}

@test "main: published sha -> uplift=false" {
    version_file=$(make_version_file)
    list_prefix_objects() { printf 'tt_lang-1+light.whl\n'; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --version-file "$version_file" --ttlang-head "$HEAD_A"
    run cat "$GH_OUT"
    assert_output --partial "uplift=false"
    refute_output --partial "tt_metal_sha="
}

@test "main: doomed sha (miss at same HEAD) -> uplift=false" {
    version_file=$(make_version_file)
    list_prefix_objects() { printf 'attempt.json\n'; }
    read_recorded_head() { printf '%s' "$HEAD_A"; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --version-file "$version_file" --ttlang-head "$HEAD_A"
    run cat "$GH_OUT"
    assert_output --partial "uplift=false"
}

@test "main: recorded miss but tt-lang HEAD advanced -> uplift=true" {
    version_file=$(make_version_file)
    list_prefix_objects() { printf 'attempt.json\n'; }
    read_recorded_head() { printf '%s' "$HEAD_A"; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --version-file "$version_file" --ttlang-head "$HEAD_B"
    run cat "$GH_OUT"
    assert_output --partial "uplift=true"
}

@test "main --assume-new: uplift=true without reading S3" {
    version_file=$(make_version_file)
    # Any S3 read must abort the run; --assume-new must not reach these.
    list_prefix_objects() { echo "S3 must not be read" >&2; return 1; }
    read_recorded_head() { echo "S3 must not be read" >&2; return 1; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --version-file "$version_file" --ttlang-head "$HEAD_A" --assume-new
    run cat "$GH_OUT"
    assert_output --partial "uplift=true"
    assert_output --partial "tt_metal_sha=$FULL_SHA"
    assert_output --partial "tt_metal_sha_short=$SHORT_SHA"
}

@test "main --assume-new: unreadable version file still aborts" {
    export GITHUB_OUTPUT="$GH_OUT"
    run main --version-file "$BATS_TEST_TMPDIR/missing.txt" --assume-new
    assert_failure
    run cat "$GH_OUT"
    refute_output --partial "uplift="
}

@test "main: unreadable version file aborts without emitting uplift" {
    list_prefix_objects() { printf ''; }
    export GITHUB_OUTPUT="$GH_OUT"
    run main --version-file "$BATS_TEST_TMPDIR/missing.txt" --ttlang-head "$HEAD_A"
    assert_failure
    run cat "$GH_OUT"
    refute_output --partial "uplift="
}

@test "main: unknown argument -> exit 2" {
    run main --bogus
    assert_equal "$status" 2
}
