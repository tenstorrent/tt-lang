#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/detect-ttmlir-ttmetal-uplift.sh. The script is
# sourced (its main() is guarded) so the S3 lookups can be overridden without
# real AWS access.

load test_helper

FULL_SHA="13adda80fef07a3c5d6f2f8b9a0c1d2e3f405162"
SHORT_SHA="13adda8"
HEAD_A="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
HEAD_B="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

make_cmakelists() {
    local sha="$1"
    local file="$BATS_TEST_TMPDIR/CMakeLists.$RANDOM.txt"
    cat > "$file" <<EOF
include(ExternalProject)
set(TT_METAL_VERSION "$sha")
ExternalProject_Add(tt-metal)
EOF
    echo "$file"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/detect-ttmlir-ttmetal-uplift.sh"
    GH_OUT="$BATS_TEST_TMPDIR/gh_out"
    : > "$GH_OUT"
    source "$SCRIPT"
}

# --- read_target_ttmetal_sha ---

@test "read_target_ttmetal_sha extracts the pinned sha" {
    cml=$(make_cmakelists "$FULL_SHA")
    run -0 read_target_ttmetal_sha "$cml"
    assert_output "$FULL_SHA"
}

@test "read_target_ttmetal_sha fails on a missing file" {
    run read_target_ttmetal_sha "$BATS_TEST_TMPDIR/nope.txt"
    assert_failure
    assert_output --partial "not found"
}

@test "read_target_ttmetal_sha fails without a version line" {
    f="$BATS_TEST_TMPDIR/empty.txt"; echo "set(X 1)" > "$f"
    run read_target_ttmetal_sha "$f"
    assert_failure
    assert_output --partial "No 'set(TT_METAL_VERSION"
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

# --- main ---

@test "main: new sha -> uplift=true with sha outputs" {
    cml=$(make_cmakelists "$FULL_SHA")
    list_prefix_objects() { printf ''; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --cmakelists "$cml" --ttlang-head "$HEAD_A"
    run cat "$GH_OUT"
    assert_output --partial "uplift=true"
    assert_output --partial "tt_metal_sha=$FULL_SHA"
    assert_output --partial "tt_metal_sha_short=$SHORT_SHA"
}

@test "main: published sha -> uplift=false" {
    cml=$(make_cmakelists "$FULL_SHA")
    list_prefix_objects() { printf 'tt_lang-1+light.whl\n'; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --cmakelists "$cml" --ttlang-head "$HEAD_A"
    run cat "$GH_OUT"
    assert_output --partial "uplift=false"
    refute_output --partial "tt_metal_sha="
}

@test "main: doomed sha (miss at same HEAD) -> uplift=false" {
    cml=$(make_cmakelists "$FULL_SHA")
    list_prefix_objects() { printf 'attempt.json\n'; }
    read_recorded_head() { printf '%s' "$HEAD_A"; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --cmakelists "$cml" --ttlang-head "$HEAD_A"
    run cat "$GH_OUT"
    assert_output --partial "uplift=false"
}

@test "main: recorded miss but tt-lang HEAD advanced -> uplift=true" {
    cml=$(make_cmakelists "$FULL_SHA")
    list_prefix_objects() { printf 'attempt.json\n'; }
    read_recorded_head() { printf '%s' "$HEAD_A"; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --cmakelists "$cml" --ttlang-head "$HEAD_B"
    run cat "$GH_OUT"
    assert_output --partial "uplift=true"
}

@test "main --assume-new: uplift=true without reading S3" {
    cml=$(make_cmakelists "$FULL_SHA")
    # Any S3 read must abort the run; --assume-new must not reach these.
    list_prefix_objects() { echo "S3 must not be read" >&2; return 1; }
    read_recorded_head() { echo "S3 must not be read" >&2; return 1; }
    export GITHUB_OUTPUT="$GH_OUT"
    run -0 main --cmakelists "$cml" --ttlang-head "$HEAD_A" --assume-new
    run cat "$GH_OUT"
    assert_output --partial "uplift=true"
    assert_output --partial "tt_metal_sha=$FULL_SHA"
    assert_output --partial "tt_metal_sha_short=$SHORT_SHA"
}

@test "main --assume-new: unreadable CMakeLists still aborts" {
    export GITHUB_OUTPUT="$GH_OUT"
    run main --cmakelists "$BATS_TEST_TMPDIR/missing.txt" --assume-new
    assert_failure
    run cat "$GH_OUT"
    refute_output --partial "uplift="
}

@test "main: unreadable CMakeLists aborts without emitting uplift" {
    list_prefix_objects() { printf ''; }
    export GITHUB_OUTPUT="$GH_OUT"
    run main --cmakelists "$BATS_TEST_TMPDIR/missing.txt" --ttlang-head "$HEAD_A"
    assert_failure
    run cat "$GH_OUT"
    refute_output --partial "uplift="
}

@test "main: unknown argument -> exit 2" {
    run main --bogus
    assert_equal "$status" 2
}
