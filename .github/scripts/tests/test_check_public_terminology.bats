#!/usr/bin/env bats

load test_helper

setup() {
    TEST_REPO=$(mkrepo)
    install_scripts_in_repo "$TEST_REPO"
    git -C "$TEST_REPO" update-ref refs/remotes/origin/main HEAD
    CHECK_SCRIPT="$TEST_REPO/.github/scripts/check-public-terminology.sh"
}

@test "accepts an ordinary branch and diff" {
    git -C "$TEST_REPO" switch -q -c feature/row-fusion
    printf '%s\n' "ordinary change" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "ordinary change"

    run env -u GITHUB_HEAD_REF -u GITHUB_REF_NAME \
        bash -c 'cd "$1" && exec "$2"' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_success
}

@test "rejects the prohibited identifier in the branch name" {
    branch_name="feature/rmsnorm-bla""ze-parity"

    run env GITHUB_HEAD_REF="$branch_name" \
        bash -c 'cd "$1" && exec "$2"' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "branch name contains a non-public comparison identifier"
}

@test "branch-name matching is case insensitive" {
    branch_name="feature/rmsnorm-BLA""ZE-parity"

    run env GITHUB_HEAD_REF="$branch_name" \
        bash -c 'cd "$1" && exec "$2"' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "branch name contains a non-public comparison identifier"
}

@test "rejects the prohibited identifier in a committed diff" {
    git -C "$TEST_REPO" switch -q -c feature/row-fusion
    printf '%s\n' "non-public bla""ze reference" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "add non-public reference"

    run env -u GITHUB_HEAD_REF -u GITHUB_REF_NAME \
        bash -c 'cd "$1" && exec "$2"' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "branch diff contains a non-public comparison identifier"
}

@test "rejects the prohibited identifier in a staged diff" {
    printf '%s\n' "non-public bla""ze reference" > "$TEST_REPO/change.txt"
    git -C "$TEST_REPO" add change.txt

    run env GITHUB_HEAD_REF=feature/row-fusion \
        bash -c 'cd "$1" && exec "$2"' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "staged diff contains a non-public comparison identifier"
}

@test "rejects removal of the prohibited identifier" {
    printf '%s\n' "non-public bla""ze reference" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "add non-public reference"
    git -C "$TEST_REPO" update-ref refs/remotes/origin/main HEAD
    git -C "$TEST_REPO" switch -q -c feature/remove-reference
    printf '%s\n' "ordinary text" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "remove non-public reference"

    run env -u GITHUB_HEAD_REF -u GITHUB_REF_NAME \
        bash -c 'cd "$1" && exec "$2"' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "branch diff contains a non-public comparison identifier"
}

@test "rejects an unresolved CI base ref" {
    git -C "$TEST_REPO" update-ref -d refs/remotes/origin/main

    run env GITHUB_HEAD_REF=feature/row-fusion GITHUB_BASE_REF=main \
        bash -c 'cd "$1" && exec "$2"' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "cannot resolve required diff base"
}

@test "rejects the prohibited identifier in a pushed commit range" {
    push_base=$(git -C "$TEST_REPO" rev-parse HEAD)
    git -C "$TEST_REPO" switch -q -c feature/row-fusion
    printf '%s\n' "non-public bla""ze reference" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "add non-public reference"
    git -C "$TEST_REPO" update-ref refs/remotes/origin/main HEAD

    run env TTLANG_PUBLIC_DIFF_BASE="$push_base" \
        bash -c 'cd "$1" && exec "$2"' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "branch diff contains a non-public comparison identifier"
}
