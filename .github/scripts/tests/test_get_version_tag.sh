#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Thorough behavioral tests for .github/containers/get-version-tag.sh.

set -uo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

# Synthetic base version tag used by `fresh_tagged_repo`. Chosen well outside
# any real release range so the literals can never be confused with an actual
# production tag. Most cases mutate from this initial state and assert
# against `$BASE_TAG`.
BASE_TAG="v99.99.99"
# A later "newer" tag for the multiple-tags / nearest-tag case.
NEWER_TAG="v99.99.100"

# Run the script under test inside a given repo. Echoes the script's stdout;
# stderr is preserved so failures surface in CI logs.
get_tag() {
    local repo="$1"
    shift
    (cd "$repo" && "$@" .github/containers/get-version-tag.sh)
}

# Convenience: stage all changes and commit with a message.
commit_all() {
    local repo="$1"
    local msg="$2"
    (cd "$repo" && git add -A && git commit -q -m "$msg")
}

# === Case: no version tags in history ===
{
    start_case "exits 1 when there are no v[0-9]* tags"
    repo=$(mkrepo)
    trap 'rm -rf "$repo"' EXIT
    install_scripts_in_repo "$repo"
    set +e
    out=$(get_tag "$repo" 2>&1)
    rc=$?
    set -e
    assert_eq "$rc" "1" "no-tag: exit 1"
    assert_matches "$out" "Could not determine version tag" "no-tag: error message"
    rm -rf "$repo"
    trap - EXIT
}

# Helper for the remaining cases: spin up a repo and tag the initial commit
# with $BASE_TAG. Each case mutates from there.
fresh_tagged_repo() {
    local repo
    repo=$(mkrepo)
    install_scripts_in_repo "$repo"
    (cd "$repo" && git tag "$BASE_TAG")
    echo "$repo"
}

# === Case: clean release tag at HEAD ===
{
    start_case "clean release tag returns the tag name"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    assert_eq "$(get_tag "$repo")" "$BASE_TAG" "clean: $BASE_TAG"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: clean tag, one commit past, no uplift ===
{
    start_case "non-uplift commit past tag returns the tag (no suffix)"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    echo "kernel fix" >> "$repo/python/sim/example.py"
    commit_all "$repo" "kernel fix"
    assert_eq "$(get_tag "$repo")" "$BASE_TAG" "kernel-only diff: clean tag"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: uplift in each path separately ===
for path_to_change in \
    "third-party/tt-metal-version" \
    "third-party/llvm-project/sentinel" \
    "third-party/tt-metal/sentinel" \
    ".github/containers/Dockerfile.base" \
    "requirements-runtime.txt"; do
    start_case "uplift in $path_to_change produces -uplift-<hash> form"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    echo "modified" >> "$repo/$path_to_change"
    commit_all "$repo" "uplift $path_to_change"
    tag=$(get_tag "$repo")
    assert_matches "$tag" '^v99\.99\.99-uplift-[a-f0-9]{8}$' "$path_to_change: matches uplift form"
    rm -rf "$repo"
    trap - EXIT
done

# === Case: hash determinism — same uplift content yields same tag ===
{
    start_case "hash determinism across independent repos with same content"
    repo1=$(fresh_tagged_repo)
    repo2=$(fresh_tagged_repo)
    trap 'rm -rf "$repo1" "$repo2"' EXIT
    for r in "$repo1" "$repo2"; do
        echo "identical-content-v2" > "$r/third-party/tt-metal-version"
        commit_all "$r" "uplift"
    done
    tag1=$(get_tag "$repo1")
    tag2=$(get_tag "$repo2")
    assert_eq "$tag1" "$tag2" "same content -> same hash"
    rm -rf "$repo1" "$repo2"
    trap - EXIT
}

# === Case: revert determinism — uplift, revert, re-apply same uplift ===
{
    start_case "revert + re-apply same uplift yields same hash"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    echo "uplift-state" > "$repo/third-party/tt-metal-version"
    commit_all "$repo" "uplift"
    first_tag=$(get_tag "$repo")
    echo "v0.69.0" > "$repo/third-party/tt-metal-version"
    commit_all "$repo" "revert"
    revert_tag=$(get_tag "$repo")
    echo "uplift-state" > "$repo/third-party/tt-metal-version"
    commit_all "$repo" "re-uplift"
    second_tag=$(get_tag "$repo")
    assert_eq "$revert_tag" "$BASE_TAG" "revert restores clean tag"
    assert_eq "$first_tag" "$second_tag" "re-apply matches first hash"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: different uplift content yields different hashes ===
{
    start_case "different uplift contents yield different hashes"
    repo1=$(fresh_tagged_repo)
    repo2=$(fresh_tagged_repo)
    trap 'rm -rf "$repo1" "$repo2"' EXIT
    echo "content-A" > "$repo1/third-party/tt-metal-version"
    echo "content-B" > "$repo2/third-party/tt-metal-version"
    commit_all "$repo1" "uplift A"
    commit_all "$repo2" "uplift B"
    tag1=$(get_tag "$repo1")
    tag2=$(get_tag "$repo2")
    assert_neq "$tag1" "$tag2" "different content -> different hash"
    rm -rf "$repo1" "$repo2"
    trap - EXIT
}

# === Case: tag with `+` build metadata is translated to `-` ===
{
    start_case "tag with + build metadata gets - translation"
    repo=$(mkrepo)
    trap 'rm -rf "$repo"' EXIT
    install_scripts_in_repo "$repo"
    (cd "$repo" && git tag "${BASE_TAG}+local1")
    assert_eq "$(get_tag "$repo")" "${BASE_TAG}-local1" "+ -> -"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: tag with existing `-` (rc / dev) passes through unchanged ===
{
    start_case "tag with -rc1 passes through"
    repo=$(mkrepo)
    trap 'rm -rf "$repo"' EXIT
    install_scripts_in_repo "$repo"
    (cd "$repo" && git tag "${BASE_TAG}-rc1")
    assert_eq "$(get_tag "$repo")" "${BASE_TAG}-rc1" "rc1 unchanged"
    rm -rf "$repo"
    trap - EXIT
}

{
    start_case "tag with -dev<YYYYMMDD> passes through"
    repo=$(mkrepo)
    trap 'rm -rf "$repo"' EXIT
    install_scripts_in_repo "$repo"
    (cd "$repo" && git tag "${BASE_TAG}-dev20260515")
    assert_eq "$(get_tag "$repo")" "${BASE_TAG}-dev20260515" "dev20260515 unchanged"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: nearest tag is picked among multiple ===
{
    start_case "nearest tag is used when multiple v* tags exist"
    repo=$(mkrepo)
    trap 'rm -rf "$repo"' EXIT
    install_scripts_in_repo "$repo"
    (cd "$repo" && git tag "$BASE_TAG")
    echo "advance" >> "$repo/python/sim/example.py"
    commit_all "$repo" "advance"
    (cd "$repo" && git tag "$NEWER_TAG")
    assert_eq "$(get_tag "$repo")" "$NEWER_TAG" "nearest tag returned"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: subdir invariance — running from a subdirectory ===
{
    start_case "running from a subdirectory of the repo yields the same tag"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    echo "uplift" > "$repo/third-party/tt-metal-version"
    commit_all "$repo" "uplift"
    top_tag=$(cd "$repo" && .github/containers/get-version-tag.sh)
    sub_tag=$(cd "$repo/python/sim" && ../../.github/containers/get-version-tag.sh)
    assert_eq "$top_tag" "$sub_tag" "subdir invariance"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: locale invariance — POSIX vs UTF-8 ===
# Skipped on systems without en_US.UTF-8 installed; a silent fallback would
# make the assertion vacuous.
if locale -a 2>/dev/null | grep -qiE '^en_US\.utf-?8$'; then
    {
        start_case "tag is locale-invariant"
        repo=$(fresh_tagged_repo)
        trap 'rm -rf "$repo"' EXIT
        echo "uplift" > "$repo/third-party/tt-metal-version"
        commit_all "$repo" "uplift"
        c_tag=$(LC_ALL=C get_tag "$repo")
        en_tag=$(LC_ALL=en_US.UTF-8 get_tag "$repo")
        assert_eq "$c_tag" "$en_tag" "C vs en_US.UTF-8"
        rm -rf "$repo"
        trap - EXIT
    }
else
    echo "  SKIP: locale invariance (en_US.UTF-8 not available)"
fi

# === Case: change in a non-uplift path doesn't toggle uplift form ===
{
    start_case "change in non-uplift path stays on clean tag"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    mkdir -p "$repo/lib"
    echo "non-uplift file" > "$repo/lib/something.cpp"
    commit_all "$repo" "non-uplift change"
    assert_eq "$(get_tag "$repo")" "$BASE_TAG" "lib/ change: clean tag"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: multiple uplift paths together ===
{
    start_case "multiple uplift paths together produce a single uplift tag"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    echo "new-version" > "$repo/third-party/tt-metal-version"
    echo "new-llvm" >> "$repo/third-party/llvm-project/sentinel"
    echo "new-dep" >> "$repo/requirements-runtime.txt"
    commit_all "$repo" "multi-uplift"
    tag=$(get_tag "$repo")
    assert_matches "$tag" '^v99\.99\.99-uplift-[a-f0-9]{8}$' "multi-uplift: single tag form"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: missing UPLIFT_PATHS file fails noisily ===
{
    start_case "missing uplift-paths.sh fails noisily"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    rm "$repo/.github/scripts/uplift-paths.sh"
    set +e
    out=$(cd "$repo" && .github/containers/get-version-tag.sh 2>&1)
    rc=$?
    set -e
    assert_neq "$rc" "0" "missing helper: non-zero exit"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: hash is invariant under UPLIFT_PATHS reordering ===
# Pins the invariant that `git ls-tree HEAD -- A B` outputs in tree-position
# order (alphabetical by name within tree), NOT argument order. Anyone editing
# uplift-paths.sh to add or reorder entries must not accidentally change the
# hash for an unchanged source state.
{
    start_case "hash invariant under UPLIFT_PATHS reordering"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    # Apply an uplift change in multiple paths.
    echo "new-version" > "$repo/third-party/tt-metal-version"
    echo "new-llvm" >> "$repo/third-party/llvm-project/sentinel"
    echo "new-dep" >> "$repo/requirements-runtime.txt"
    (cd "$repo" && git add -A && git commit -q -m "multi-uplift")
    tag_forward=$(get_tag "$repo")
    # Overwrite uplift-paths.sh with the same entries in reverse order.
    cat > "$repo/.github/scripts/uplift-paths.sh" <<'EOF'
#!/bin/bash
UPLIFT_PATHS=(
    requirements-runtime.txt
    .github/containers/Dockerfile.base
    third-party/tt-metal
    third-party/llvm-project
    third-party/tt-metal-version
)
EOF
    tag_reversed=$(get_tag "$repo")
    assert_eq "$tag_forward" "$tag_reversed" "reordered array yields the same tag"
    rm -rf "$repo"
    trap - EXIT
}

# === Case: empty UPLIFT_PATHS array fails noisily ===
{
    start_case "empty UPLIFT_PATHS array fails noisily"
    repo=$(fresh_tagged_repo)
    trap 'rm -rf "$repo"' EXIT
    cat > "$repo/.github/scripts/uplift-paths.sh" <<'EOF'
#!/bin/bash
UPLIFT_PATHS=()
EOF
    set +e
    out=$(cd "$repo" && .github/containers/get-version-tag.sh 2>&1)
    rc=$?
    set -e
    assert_eq "$rc" "1" "empty array: exit 1"
    assert_matches "$out" "UPLIFT_PATHS is empty" "empty array: error message names the cause"
    rm -rf "$repo"
    trap - EXIT
}

finish_tests
