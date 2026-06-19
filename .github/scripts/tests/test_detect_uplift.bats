#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/detect-uplift.sh.

load test_helper

# Run detect-uplift.sh inside $REPO with the given base and head SHAs.
# Echoes the uplift value ("true" / "false") parsed from GITHUB_OUTPUT writes,
# or "EXITNZ" if the script exited non-zero.
run_detect() {
    local base="$1"
    local head="$2"
    local gh_out
    gh_out="$BATS_TEST_TMPDIR/gh_out.$RANDOM"
    : > "$gh_out"
    (cd "$REPO" && GITHUB_OUTPUT="$gh_out" .github/scripts/detect-uplift.sh "$base" "$head") >/dev/null 2>&1
    local rc=$?
    if [[ "$rc" -ne 0 ]]; then
        echo "EXITNZ"
    else
        grep '^uplift=' "$gh_out" | sed 's/^uplift=//'
    fi
}

setup() {
    REPO=$(mkrepo)
    install_scripts_in_repo "$REPO"
    BASE=$(cd "$REPO" && git rev-parse HEAD)
}

# --- Per-path uplift detection: each uplift path separately. ---

@test "diff in third-party/tt-metal-version marks uplift=true" {
    echo "modified" >> "$REPO/third-party/tt-metal-version"
    commit_all "$REPO" "uplift"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "true"
}

@test "diff in third-party/llvm-project marks uplift=true" {
    echo "modified" >> "$REPO/third-party/llvm-project/sentinel"
    commit_all "$REPO" "uplift"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "true"
}

@test "diff in third-party/tt-metal marks uplift=true" {
    echo "modified" >> "$REPO/third-party/tt-metal/sentinel"
    commit_all "$REPO" "uplift"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "true"
}

@test "diff in .github/containers/Dockerfile.base marks uplift=true" {
    echo "modified" >> "$REPO/.github/containers/Dockerfile.base"
    commit_all "$REPO" "uplift"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "true"
}

@test "diff in .github/containers/Dockerfile.wheel-manylinux-2-34 marks uplift=true" {
    echo "modified" >> "$REPO/.github/containers/Dockerfile.wheel-manylinux-2-34"
    commit_all "$REPO" "uplift"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "true"
}

@test "diff in requirements-runtime.txt marks uplift=true" {
    echo "modified" >> "$REPO/requirements-runtime.txt"
    commit_all "$REPO" "uplift"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "true"
}

# --- No-diff case ---

@test "same base and head -> uplift=false" {
    assert_equal "$(run_detect "$BASE" "$BASE")" "false"
}

# --- Diff in non-uplift path only ---

@test "diff in non-uplift path -> uplift=false" {
    mkdir -p "$REPO/lib"
    echo "kernel change" > "$REPO/lib/something.cpp"
    commit_all "$REPO" "kernel"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "false"
}

# --- Regression: tt-mlir is NOT uplift (built fresh by call-build.yml) ---
# Guards against a future "is tt-mlir uplift?" mistake re-adding it to
# UPLIFT_PATHS.
@test "diff in third-party/tt-mlir alone -> uplift=false" {
    mkdir -p "$REPO/third-party/tt-mlir"
    echo "tt-mlir bump" > "$REPO/third-party/tt-mlir/sentinel"
    commit_all "$REPO" "tt-mlir-only"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "false"
}

# --- Regression: pyproject.toml is NOT uplift (covered by wheel filter,
# not container content) ---
@test "diff in pyproject.toml alone -> uplift=false" {
    echo "[project]" > "$REPO/pyproject.toml"
    commit_all "$REPO" "pyproject-only"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "false"
}

# --- Mixed: uplift path + non-uplift path -> uplift=true ---

@test "diff in both uplift and non-uplift paths -> uplift=true" {
    echo "new tt-metal" > "$REPO/third-party/tt-metal-version"
    mkdir -p "$REPO/lib"
    echo "kernel" > "$REPO/lib/foo.cpp"
    commit_all "$REPO" "mixed"
    head=$(cd "$REPO" && git rev-parse HEAD)
    assert_equal "$(run_detect "$BASE" "$head")" "true"
}

# Without `cd $(git rev-parse --show-toplevel)`, `git diff -- <paths>` from a
# subdir interprets the paths relative to the subdir and silently produces an
# empty result.
@test "uplift detection is CWD-invariant (subdir regression)" {
    echo "new tt-metal" > "$REPO/third-party/tt-metal-version"
    commit_all "$REPO" "uplift"
    head=$(cd "$REPO" && git rev-parse HEAD)
    mkdir -p "$REPO/lib/subdir"
    gh_out="$BATS_TEST_TMPDIR/gh_out.subdir"
    : > "$gh_out"
    (cd "$REPO/lib/subdir" && GITHUB_OUTPUT="$gh_out" "$REPO/.github/scripts/detect-uplift.sh" "$BASE" "$head") >/dev/null 2>&1
    val=$(grep '^uplift=' "$gh_out" | sed 's/^uplift=//')
    assert_equal "$val" "true"
}

# --- Missing arguments are rejected ---

@test "missing base sha errors out" {
    GITHUB_OUTPUT="$BATS_TEST_TMPDIR/gh_out" run bash -c "cd '$REPO' && .github/scripts/detect-uplift.sh"
    assert_failure
}

@test "missing head sha errors out" {
    GITHUB_OUTPUT="$BATS_TEST_TMPDIR/gh_out" run bash -c "cd '$REPO' && .github/scripts/detect-uplift.sh '$BASE'"
    assert_failure
}
