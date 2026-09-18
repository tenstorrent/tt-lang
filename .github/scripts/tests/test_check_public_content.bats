#!/usr/bin/env bats

load test_helper

setup() {
    TEST_REPO=$(mkrepo)
    install_scripts_in_repo "$TEST_REPO"
    git -C "$TEST_REPO" update-ref refs/remotes/origin/main HEAD
    CHECK_SCRIPT="$TEST_REPO/.github/scripts/check-public-content.py"
    RESTRICTED_TEXT="private-reference"
    TEST_SIGNATURE=$(python3 -c \
        'import hashlib, sys; content = sys.argv[1].lower().encode(); print(f"{len(content)}:{hashlib.sha256(content).hexdigest()}")' \
        "$RESTRICTED_TEXT")
}

# Empty the environment: a pull_request workflow run exports GITHUB_BASE_REF
# naming a base branch the fixture repository does not have, and an allow-list
# also excludes inputs the script reads later. Each test's own `env` runs after
# this one, so its values take precedence.
run_check() {
    run env -i PATH="$PATH" HOME="$HOME" \
        TTLANG_PUBLIC_CONTENT_SIGNATURES="$TEST_SIGNATURE" "$@"
}

@test "accepts an ordinary branch and diff" {
    git -C "$TEST_REPO" switch -q -c feature/row-fusion
    printf '%s\n' "ordinary change" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "ordinary change"

    run_check env -u GITHUB_HEAD_REF -u GITHUB_REF_NAME \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_success
}

@test "rejects restricted content in the branch name" {
    run_check env GITHUB_HEAD_REF="feature/$RESTRICTED_TEXT" \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "branch name contains restricted public content"
}

@test "branch-name matching is case insensitive" {
    uppercase_text=$(printf '%s' "$RESTRICTED_TEXT" | tr '[:lower:]' '[:upper:]')

    run_check env GITHUB_HEAD_REF="feature/$uppercase_text" \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "branch name contains restricted public content"
}

@test "rejects restricted content in a committed diff" {
    git -C "$TEST_REPO" switch -q -c feature/row-fusion
    printf '%s\n' "$RESTRICTED_TEXT" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "add restricted reference"

    run_check env -u GITHUB_HEAD_REF -u GITHUB_REF_NAME \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "branch diff contains restricted public content"
}

@test "rejects restricted content in a staged diff" {
    printf '%s\n' "$RESTRICTED_TEXT" > "$TEST_REPO/change.txt"
    git -C "$TEST_REPO" add change.txt

    run_check env GITHUB_HEAD_REF=feature/row-fusion \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "staged diff contains restricted public content"
}

@test "rejects restricted content in a staged filename" {
    printf '%s\n' "ordinary text" > "$TEST_REPO/$RESTRICTED_TEXT.txt"
    git -C "$TEST_REPO" add "$RESTRICTED_TEXT.txt"

    run_check env GITHUB_HEAD_REF=feature/row-fusion \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "staged diff contains restricted public content"
}

@test "rejects removal of restricted content" {
    printf '%s\n' "$RESTRICTED_TEXT" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "add restricted reference"
    git -C "$TEST_REPO" update-ref refs/remotes/origin/main HEAD
    git -C "$TEST_REPO" switch -q -c feature/remove-reference
    printf '%s\n' "ordinary text" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "remove restricted reference"

    run_check env -u GITHUB_HEAD_REF -u GITHUB_REF_NAME \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "branch diff contains restricted public content"
}

@test "rejects an unresolved CI base ref" {
    git -C "$TEST_REPO" update-ref -d refs/remotes/origin/main

    run_check env GITHUB_HEAD_REF=feature/row-fusion GITHUB_BASE_REF=main \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "cannot resolve required diff base"
}

@test "rejects an unresolved default diff base" {
    git -C "$TEST_REPO" update-ref -d refs/remotes/origin/main

    run_check env -u GITHUB_BASE_REF -u TTLANG_PUBLIC_DIFF_BASE \
        GITHUB_HEAD_REF=feature/row-fusion \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "cannot resolve required diff base"
}

@test "accepts an ordinary commit message in Git metadata" {
    git_directory=$(git -C "$TEST_REPO" rev-parse --absolute-git-dir)
    commit_message_file="$git_directory/COMMIT_EDITMSG"
    printf '%s\n' "ordinary message" > "$commit_message_file"

    run_check bash -c 'cd "$1" && exec "$2" commit-message "$3"' \
        _ "$TEST_REPO" "$CHECK_SCRIPT" "$commit_message_file"

    assert_success
}

@test "rejects restricted content in a commit message" {
    git_directory=$(git -C "$TEST_REPO" rev-parse --absolute-git-dir)
    commit_message_file="$git_directory/COMMIT_EDITMSG"
    printf '%s\n' "reference $RESTRICTED_TEXT" > "$commit_message_file"

    run_check bash -c 'cd "$1" && exec "$2" commit-message "$3"' \
        _ "$TEST_REPO" "$CHECK_SCRIPT" "$commit_message_file"

    assert_failure
    assert_output --partial "commit message contains restricted public content"
}

@test "rejects a commit message file outside Git metadata" {
    commit_message_file="$BATS_TEST_TMPDIR/commit-message"
    printf '%s\n' "ordinary message" > "$commit_message_file"

    run_check bash -c 'cd "$1" && exec "$2" commit-message "$3"' \
        _ "$TEST_REPO" "$CHECK_SCRIPT" "$commit_message_file"

    assert_failure
    assert_output --partial "commit message file is outside Git metadata"
}

@test "rejects restricted content in the remote push branch" {
    target_ref=$(git -C "$TEST_REPO" rev-parse HEAD)

    run_check env PRE_COMMIT_FROM_REF="$target_ref" PRE_COMMIT_TO_REF="$target_ref" \
        PRE_COMMIT_LOCAL_BRANCH=refs/heads/feature/row-fusion \
        PRE_COMMIT_REMOTE_BRANCH="refs/heads/feature/$RESTRICTED_TEXT" \
        bash -c 'cd "$1" && exec "$2" push' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "remote branch name contains restricted public content"
}

@test "rejects restricted content in pushed commit messages" {
    source_ref=$(git -C "$TEST_REPO" rev-parse HEAD)
    git -C "$TEST_REPO" switch -q -c feature/row-fusion
    printf '%s\n' "ordinary change" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "mention $RESTRICTED_TEXT"
    target_ref=$(git -C "$TEST_REPO" rev-parse HEAD)

    run_check env PRE_COMMIT_FROM_REF="$source_ref" PRE_COMMIT_TO_REF="$target_ref" \
        PRE_COMMIT_LOCAL_BRANCH=refs/heads/feature/row-fusion \
        PRE_COMMIT_REMOTE_BRANCH=refs/heads/feature/row-fusion \
        bash -c 'cd "$1" && exec "$2" push' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "pushed commit-message data contains restricted public content"
}

@test "rejects restricted content removed by a later pushed commit" {
    source_ref=$(git -C "$TEST_REPO" rev-parse HEAD)
    git -C "$TEST_REPO" switch -q -c feature/row-fusion
    printf '%s\n' "$RESTRICTED_TEXT" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "add temporary content"
    printf '%s\n' "ordinary change" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "replace temporary content"
    target_ref=$(git -C "$TEST_REPO" rev-parse HEAD)

    run_check env PRE_COMMIT_FROM_REF="$source_ref" PRE_COMMIT_TO_REF="$target_ref" \
        PRE_COMMIT_LOCAL_BRANCH=refs/heads/feature/row-fusion \
        PRE_COMMIT_REMOTE_BRANCH=refs/heads/feature/row-fusion \
        bash -c 'cd "$1" && exec "$2" push' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "pushed commit-patch data contains restricted public content"
}

@test "rejects restricted commit metadata on a new branch push" {
    zero_ref=0000000000000000000000000000000000000000
    git -C "$TEST_REPO" switch -q -c feature/row-fusion
    printf '%s\n' "ordinary change" > "$TEST_REPO/change.txt"
    commit_all "$TEST_REPO" "mention $RESTRICTED_TEXT"
    target_ref=$(git -C "$TEST_REPO" rev-parse HEAD)

    run_check env PRE_COMMIT_FROM_REF="$zero_ref" PRE_COMMIT_TO_REF="$target_ref" \
        PRE_COMMIT_LOCAL_BRANCH=refs/heads/feature/row-fusion \
        PRE_COMMIT_REMOTE_BRANCH=refs/heads/feature/row-fusion \
        bash -c 'cd "$1" && exec "$2" push' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "pushed commit-message data contains restricted public content"
}

@test "permits deletion of a restricted remote branch" {
    zero_ref=0000000000000000000000000000000000000000

    run_check env PRE_COMMIT_TO_REF="$zero_ref" \
        PRE_COMMIT_REMOTE_BRANCH="refs/heads/feature/$RESTRICTED_TEXT" \
        bash -c 'cd "$1" && exec "$2" push' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_success
}

@test "rejects invalid signature configuration" {
    run env TTLANG_PUBLIC_CONTENT_SIGNATURES=invalid \
        bash -c 'cd "$1" && exec "$2" change' _ "$TEST_REPO" "$CHECK_SCRIPT"

    assert_failure
    assert_output --partial "invalid public-content signature configuration"
}
