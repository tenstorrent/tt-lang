#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Thorough behavioral tests for .github/containers/get-version-tag.sh.

load test_helper

# Synthetic version tags, well outside any real release range.
BASE_TAG="v99.99.99"
NEWER_TAG="v99.99.100"

# Run the script under test inside a given repo. Args after the repo path are
# environment overrides (e.g. LC_ALL=C). Echoes stdout only.
get_tag() {
    local repo="$1"
    shift
    (cd "$repo" && "$@" .github/containers/get-version-tag.sh)
}

# Build a repo and tag the initial commit with $BASE_TAG. Echoes the repo path.
fresh_tagged_repo() {
    local repo
    repo=$(mkrepo)
    install_scripts_in_repo "$repo"
    (cd "$repo" && git tag "$BASE_TAG")
    echo "$repo"
}

# --- No version tags in history ---

@test "exits 1 when there are no v[0-9]* tags" {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
    run -1 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_output --partial "Could not determine version tag"
}

# --- Clean release tag at HEAD ---

@test "clean release tag at HEAD returns the tag name" {
    REPO=$(fresh_tagged_repo)
    run -0 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_output "$BASE_TAG"
}

# --- Clean tag, one commit past, no container input changes ---

@test "non-container commit past tag returns the tag (no suffix)" {
    REPO=$(fresh_tagged_repo)
    echo "kernel fix" >> "$REPO/python/sim/example.py"
    commit_all "$REPO" "kernel fix"
    run -0 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_output "$BASE_TAG"
}

# --- Per-path container input changes ---

container_input_one_path() {
    local path_to_change="$1"
    REPO=$(fresh_tagged_repo)
    echo "modified" >> "$REPO/$path_to_change"
    commit_all "$REPO" "container input $path_to_change"
    run -0 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    [[ "$output" =~ ^v99\.99\.99-[a-f0-9]{8}$ ]]
}

@test "container input change in third-party/tt-metal-version -> -<hash> form" {
    container_input_one_path "third-party/tt-metal-version"
}

@test "container input change in third-party/llvm-project/sentinel -> -<hash> form" {
    container_input_one_path "third-party/llvm-project/sentinel"
}

@test "container input change in third-party/tt-metal/sentinel -> -<hash> form" {
    container_input_one_path "third-party/tt-metal/sentinel"
}

@test "container input change in .github/containers/Dockerfile.base -> -<hash> form" {
    container_input_one_path ".github/containers/Dockerfile.base"
}

@test "container input change in .github/containers/Dockerfile.wheel-manylinux-2-34 -> -<hash> form" {
    container_input_one_path ".github/containers/Dockerfile.wheel-manylinux-2-34"
}

@test "container input change in requirements-runtime.txt -> -<hash> form" {
    container_input_one_path "requirements-runtime.txt"
}

# --- Hash determinism: same content yields same tag ---

@test "hash determinism across independent repos with same content" {
    REPO1=$(fresh_tagged_repo)
    REPO2=$(fresh_tagged_repo)
    for r in "$REPO1" "$REPO2"; do
        echo "identical-content-v2" > "$r/third-party/tt-metal-version"
        commit_all "$r" "container input"
    done
    tag1=$(get_tag "$REPO1")
    tag2=$(get_tag "$REPO2")
    assert_equal "$tag1" "$tag2"
}

# --- Revert determinism: modify, revert, re-apply same container input ---

@test "revert + re-apply same container input yields same hash" {
    REPO=$(fresh_tagged_repo)
    echo "container-state" > "$REPO/third-party/tt-metal-version"
    commit_all "$REPO" "container input"
    first_tag=$(get_tag "$REPO")
    # Restore the exact content mkrepo() initialized so revert matches BASE_TAG.
    write_tt_metal_version_file "$REPO/third-party/tt-metal-version" \
        "$TEST_TTNN_PYPI_VERSION" \
        "$TEST_TT_METAL_TAG" \
        "$TEST_TT_METAL_TAG"
    commit_all "$REPO" "revert"
    revert_tag=$(get_tag "$REPO")
    echo "container-state" > "$REPO/third-party/tt-metal-version"
    commit_all "$REPO" "reapply container input"
    second_tag=$(get_tag "$REPO")
    assert_equal "$revert_tag" "$BASE_TAG"
    assert_equal "$first_tag" "$second_tag"
}

# --- Different container input content yields different hashes ---

@test "different container input contents yield different hashes" {
    REPO1=$(fresh_tagged_repo)
    REPO2=$(fresh_tagged_repo)
    echo "content-A" > "$REPO1/third-party/tt-metal-version"
    echo "content-B" > "$REPO2/third-party/tt-metal-version"
    commit_all "$REPO1" "container input A"
    commit_all "$REPO2" "container input B"
    tag1=$(get_tag "$REPO1")
    tag2=$(get_tag "$REPO2")
    refute [ "$tag1" = "$tag2" ]
}

# --- Tag-format normalization ---

@test "tag with + build metadata gets - translation" {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
    (cd "$REPO" && git tag "${BASE_TAG}+local1")
    run -0 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_output "${BASE_TAG}-local1"
}

@test "tag with -rc1 passes through unchanged" {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
    (cd "$REPO" && git tag "${BASE_TAG}-rc1")
    run -0 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_output "${BASE_TAG}-rc1"
}

@test "tag with -dev<YYYYMMDD> passes through unchanged" {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
    (cd "$REPO" && git tag "${BASE_TAG}-dev20260515")
    run -0 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_output "${BASE_TAG}-dev20260515"
}

# --- Nearest tag is picked among multiple ---

@test "nearest tag is used when multiple v* tags exist" {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
    (cd "$REPO" && git tag "$BASE_TAG")
    echo "advance" >> "$REPO/python/sim/example.py"
    commit_all "$REPO" "advance"
    (cd "$REPO" && git tag "$NEWER_TAG")
    run -0 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_output "$NEWER_TAG"
}

# --- Subdir invariance: running from a subdirectory ---

@test "running from a subdirectory of the repo yields the same tag" {
    REPO=$(fresh_tagged_repo)
    echo "container input" > "$REPO/third-party/tt-metal-version"
    commit_all "$REPO" "container input"
    top_tag=$(cd "$REPO" && .github/containers/get-version-tag.sh)
    sub_tag=$(cd "$REPO/python/sim" && ../../.github/containers/get-version-tag.sh)
    assert_equal "$top_tag" "$sub_tag"
}

# --- Locale invariance: POSIX vs UTF-8 ---

@test "tag is locale-invariant (POSIX vs en_US.UTF-8)" {
    if ! locale -a 2>/dev/null | grep -qiE '^en_US\.utf-?8$'; then
        skip "en_US.UTF-8 locale not installed"
    fi
    REPO=$(fresh_tagged_repo)
    echo "container input" > "$REPO/third-party/tt-metal-version"
    commit_all "$REPO" "container input"
    c_tag=$(LC_ALL=C get_tag "$REPO")
    en_tag=$(LC_ALL=en_US.UTF-8 get_tag "$REPO")
    assert_equal "$c_tag" "$en_tag"
}

# --- Change outside container inputs does not toggle hashed form ---

@test "change outside container inputs stays on clean tag" {
    REPO=$(fresh_tagged_repo)
    mkdir -p "$REPO/lib"
    echo "non-container file" > "$REPO/lib/something.cpp"
    commit_all "$REPO" "non-container change"
    run -0 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_output "$BASE_TAG"
}

# --- Multiple container input paths together produce a single hashed tag ---

@test "multiple container input paths together produce a single hashed tag" {
    REPO=$(fresh_tagged_repo)
    echo "new-version" > "$REPO/third-party/tt-metal-version"
    echo "new-llvm" >> "$REPO/third-party/llvm-project/sentinel"
    echo "new-dep" >> "$REPO/requirements-runtime.txt"
    commit_all "$REPO" "multi-container-input"
    run -0 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    [[ "$output" =~ ^v99\.99\.99-[a-f0-9]{8}$ ]]
}

# --- Missing UPLIFT_PATHS file fails noisily ---

@test "missing uplift-paths.sh fails noisily" {
    REPO=$(fresh_tagged_repo)
    rm "$REPO/.github/scripts/uplift-paths.sh"
    run bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_failure
}

# Pins the invariant that `git ls-tree HEAD -- A B` outputs in tree-position
# order (alphabetical by name within tree), NOT argument order. Anyone editing
# uplift-paths.sh to add or reorder entries must not accidentally change the
# hash for an unchanged source state.
@test "hash invariant under UPLIFT_PATHS reordering" {
    REPO=$(fresh_tagged_repo)
    echo "new-version" > "$REPO/third-party/tt-metal-version"
    echo "new-llvm" >> "$REPO/third-party/llvm-project/sentinel"
    echo "new-dep" >> "$REPO/requirements-runtime.txt"
    commit_all "$REPO" "multi-container-input"
    tag_forward=$(get_tag "$REPO")
    cat > "$REPO/.github/scripts/uplift-paths.sh" <<'EOF'
#!/bin/bash
UPLIFT_PATHS=(
    requirements-runtime.txt
    .github/containers/Dockerfile.base
    .github/containers/Dockerfile.wheel-manylinux-2-34
    third-party/tt-metal
    third-party/llvm-project
    third-party/tt-metal-version
)
EOF
    tag_reversed=$(get_tag "$REPO")
    assert_equal "$tag_forward" "$tag_reversed"
}

@test "empty UPLIFT_PATHS array fails noisily" {
    REPO=$(fresh_tagged_repo)
    cat > "$REPO/.github/scripts/uplift-paths.sh" <<'EOF'
#!/bin/bash
UPLIFT_PATHS=()
EOF
    run -1 bash -c "cd '$REPO' && .github/containers/get-version-tag.sh"
    assert_output --partial "UPLIFT_PATHS is empty"
}
