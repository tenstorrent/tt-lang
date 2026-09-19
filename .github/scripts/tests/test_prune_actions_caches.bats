#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/prune-actions-caches.sh.

load test_helper

SCRIPT_UNDER_TEST="$BATS_TEST_DIRNAME/../prune-actions-caches.sh"

# Stub gh on PATH. Listing requests print $CACHE_JSON as the API would; delete
# requests append their cache id to $BATS_TEST_TMPDIR/deleted so a test can
# assert exactly which entries the script removed.
stub_gh() {
    mkdir -p "$BATS_TEST_TMPDIR/bin"
    cat > "$BATS_TEST_TMPDIR/bin/gh" <<'STUB'
#!/usr/bin/env bash
if [[ "${GH_API_FAIL:-false}" == true ]]; then
    exit 1
fi
if [[ "$1" == "api" && "$2" == "-X" && "$3" == "DELETE" ]]; then
    echo "${4##*/}" >> "$DELETED_FILE"
    exit 0
fi
jq_filter=""
for arg in "$@"; do
    [[ -n "$take_next" ]] && { jq_filter="$arg"; take_next=""; continue; }
    [[ "$arg" == "--jq" ]] && take_next=1
done
jq -r "$jq_filter" < "$CACHE_JSON_FILE"
STUB
    chmod +x "$BATS_TEST_TMPDIR/bin/gh"
    export PATH="$BATS_TEST_TMPDIR/bin:$PATH"
    export DELETED_FILE="$BATS_TEST_TMPDIR/deleted"
    : > "$DELETED_FILE"
}

# Write an API response holding one cache per "id created ref key" line.
write_caches() {
    export CACHE_JSON_FILE="$BATS_TEST_TMPDIR/caches.json"
    {
        echo '{"actions_caches":['
        local first=1
        while read -r id created ref key; do
            [[ -z "$id" ]] && continue
            [[ "$first" == 1 ]] || echo ','
            first=0
            printf '{"id":%s,"created_at":"%s","size_in_bytes":1000,"ref":"%s","key":"%s"}' \
                "$id" "$created" "$ref" "$key"
        done
        echo ']}'
    } > "$CACHE_JSON_FILE"
}

setup() {
    unset GH_API_FAIL
    stub_gh
}

@test "keeps the newest entries of a series and deletes the rest" {
    write_caches <<'EOF'
1 2026-09-16T01:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-build-2026-09-16T01:00:00.000Z
2 2026-09-16T02:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-build-2026-09-16T02:00:00.000Z
3 2026-09-16T03:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-build-2026-09-16T03:00:00.000Z
EOF
    run "$SCRIPT_UNDER_TEST" --keep 2 --dry-run false
    assert_success
    assert_equal "$(cat "$DELETED_FILE")" "1"
}

@test "never deletes a key outside the prefix" {
    write_caches <<'EOF'
1 2026-09-16T01:00:00.000000Z refs/heads/main Linux-toolchain_llvm-aaaaaaa_ttmetal-bbbbbbb_patches-ccccccc
2 2026-09-16T02:00:00.000000Z refs/heads/main Linux-toolchain_llvm-ddddddd_ttmetal-eeeeeee_patches-fffffff
EOF
    run "$SCRIPT_UNDER_TEST" --keep 0 --dry-run false
    assert_success
    assert_equal "$(cat "$DELETED_FILE")" ""
}

@test "counts each ref separately" {
    write_caches <<'EOF'
1 2026-09-16T01:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-build-2026-09-16T01:00:00.000Z
2 2026-09-16T02:00:00.000000Z refs/pull/2/merge ccache-Linux-ttlang-build-2026-09-16T02:00:00.000Z
EOF
    run "$SCRIPT_UNDER_TEST" --keep 1 --dry-run false
    assert_success
    assert_equal "$(cat "$DELETED_FILE")" ""
}

@test "counts each job series separately" {
    write_caches <<'EOF'
1 2026-09-16T01:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-build-2026-09-16T01:00:00.000Z
2 2026-09-16T02:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-hw-n150-2026-09-16T02:00:00.000Z
EOF
    run "$SCRIPT_UNDER_TEST" --keep 1 --dry-run false
    assert_success
    assert_equal "$(cat "$DELETED_FILE")" ""
}

@test "deletes nothing without an explicit --dry-run false" {
    write_caches <<'EOF'
1 2026-09-16T01:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-build-2026-09-16T01:00:00.000Z
2 2026-09-16T02:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-build-2026-09-16T02:00:00.000Z
EOF
    run "$SCRIPT_UNDER_TEST" --keep 1
    assert_success
    assert_equal "$(cat "$DELETED_FILE")" ""
    assert_output --partial "would delete"
}

@test "fails when the cache listing request fails" {
    export GH_API_FAIL=true
    run "$SCRIPT_UNDER_TEST"
    assert_failure
    assert_output --partial "failed to list caches"
    refute_output --partial "no caches match"
}

@test "rejects a non-numeric --keep" {
    write_caches <<'EOF'
1 2026-09-16T01:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-build-2026-09-16T01:00:00.000Z
EOF
    run "$SCRIPT_UNDER_TEST" --keep two
    assert_failure
}

@test "rejects a --dry-run value that is not true or false" {
    write_caches <<'EOF'
1 2026-09-16T01:00:00.000000Z refs/pull/1/merge ccache-Linux-ttlang-build-2026-09-16T01:00:00.000Z
EOF
    run "$SCRIPT_UNDER_TEST" --dry-run yes
    assert_failure
}
