#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/wheel-or-container-changed.sh.

load test_helper

# Run the script inside $REPO with the given base/head SHAs. Echoes
# "true" / "false" parsed from GITHUB_OUTPUT, or "EXITNZ" on non-zero exit.
run_changed() {
    local base="$1"
    local head="$2"
    local gh_out
    gh_out="$BATS_TEST_TMPDIR/gh_out.$RANDOM"
    : > "$gh_out"
    (cd "$REPO" && GITHUB_OUTPUT="$gh_out" \
        .github/scripts/wheel-or-container-changed.sh "$base" "$head") >/dev/null 2>&1
    local rc=$?
    if [[ "$rc" -ne 0 ]]; then
        echo "EXITNZ"
    else
        grep '^wheel_or_container=' "$gh_out" | sed 's/^wheel_or_container=//'
    fi
}

setup() {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
    BASE=$(cd "$REPO" && git rev-parse HEAD)
    # Create initial versions of the WHEEL_OR_CONTAINER files so each test
    # below modifies existing content (covers the "edit existing file" diff
    # form, not only "add new file").
    (
        cd "$REPO"
        mkdir -p .github/containers .github/scripts bin examples packaging python
        echo "FROM ubuntu:22.04" > .github/containers/Dockerfile
        echo "# run-tutorials" > .github/scripts/run-tutorials.sh
        echo "# smoke-test" > .github/scripts/smoke-test-wheel.py
        echo "#!/bin/sh" > bin/ttlang-sim
        echo "cmake_minimum_required(VERSION 3.20)" > CMakeLists.txt
        echo "# example" > examples/foo.py
        echo "# packaging" > packaging/setup.cfg
        echo "[project]" > pyproject.toml
        echo "# py cmake" > python/CMakeLists.txt
        echo "# setup" > python/setup.py
        git add -A
        git commit -q -m "seed wheel/container files"
    )
    BASE=$(cd "$REPO" && git rev-parse HEAD)
}

# --- Per-path detection: each WHEEL_OR_CONTAINER path triggers true. ---

@test "diff in .github/containers/Dockerfile -> true" {
    echo "RUN echo x" >> "$REPO/.github/containers/Dockerfile"
    commit_all "$REPO" "dockerfile"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "true"
}

@test "diff in .github/containers/Dockerfile.base -> true" {
    echo "RUN echo base" >> "$REPO/.github/containers/Dockerfile.base"
    commit_all "$REPO" "dockerfile.base"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "true"
}

@test "diff in bin/ttlang-sim -> true" {
    echo "echo new" >> "$REPO/bin/ttlang-sim"
    commit_all "$REPO" "bin"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "true"
}

@test "diff in CMakeLists.txt -> true" {
    echo "project(tt-lang)" >> "$REPO/CMakeLists.txt"
    commit_all "$REPO" "cmake"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "true"
}

@test "diff in examples/ -> true" {
    echo "# new example" >> "$REPO/examples/foo.py"
    commit_all "$REPO" "examples"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "true"
}

@test "diff in packaging/ -> true" {
    echo "more=1" >> "$REPO/packaging/setup.cfg"
    commit_all "$REPO" "packaging"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "true"
}

@test "diff in pyproject.toml -> true" {
    echo "name = 'x'" >> "$REPO/pyproject.toml"
    commit_all "$REPO" "pyproject"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "true"
}

@test "diff in python/setup.py -> true" {
    echo "# more" >> "$REPO/python/setup.py"
    commit_all "$REPO" "setup.py"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "true"
}

# --- Negative: diff only outside the path list. ---

@test "diff in unrelated kernel file -> false" {
    mkdir -p "$REPO/lib"
    echo "// kernel" > "$REPO/lib/foo.cpp"
    commit_all "$REPO" "kernel"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "false"
}

@test "diff only in third-party uplift path -> false (covered by detect-uplift, not this script)" {
    echo "v0.70.0" > "$REPO/third-party/tt-metal-version"
    commit_all "$REPO" "uplift only"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "false"
}

@test "no diff (same base and head) -> false" {
    assert_equal "$(run_changed "$BASE" "$BASE")" "false"
}

# --- Mixed diff: wheel path + unrelated path -> true ---

@test "mixed wheel-path + unrelated diff -> true" {
    mkdir -p "$REPO/lib"
    echo "// kernel" > "$REPO/lib/foo.cpp"
    echo "echo extra" >> "$REPO/bin/ttlang-sim"
    commit_all "$REPO" "mixed"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_changed "$BASE" "$head")" "true"
}

# --- No-PR-context safety: empty BASE / empty HEAD / zero-SHA BASE -> true ---

@test "empty BASE -> true (safety mode)" {
    assert_equal "$(run_changed "" "$BASE")" "true"
}

@test "empty HEAD -> true (safety mode)" {
    assert_equal "$(run_changed "$BASE" "")" "true"
}

@test "all-zeros BASE (first push of new branch) -> true (safety mode)" {
    assert_equal "$(run_changed "0000000000000000000000000000000000000000" "$BASE")" "true"
}

# --- CWD-invariance regression (mirrors detect-uplift.sh test). ---

@test "wheel/container detection is CWD-invariant (subdir regression)" {
    echo "RUN echo x" >> "$REPO/.github/containers/Dockerfile"
    commit_all "$REPO" "dockerfile"
    head=$(cd "$REPO" && git rev-parse HEAD)
    mkdir -p "$REPO/lib/subdir"
    gh_out="$BATS_TEST_TMPDIR/gh_out.subdir"
    : > "$gh_out"
    (cd "$REPO/lib/subdir" && GITHUB_OUTPUT="$gh_out" \
        "$REPO/.github/scripts/wheel-or-container-changed.sh" "$BASE" "$head") >/dev/null 2>&1
    val=$(grep '^wheel_or_container=' "$gh_out" | sed 's/^wheel_or_container=//')
    assert_equal "$val" "true"
}

# --- GITHUB_OUTPUT write contract ---

@test "writes wheel_or_container key to GITHUB_OUTPUT" {
    gh_out="$BATS_TEST_TMPDIR/gh_out.contract"
    : > "$gh_out"
    (cd "$REPO" && GITHUB_OUTPUT="$gh_out" \
        .github/scripts/wheel-or-container-changed.sh "$BASE" "$BASE") >/dev/null 2>&1
    run -0 cat "$gh_out"
    assert_output --partial "wheel_or_container="
}
