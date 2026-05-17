#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/detect-uplift.sh.

set -uo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

# Run detect-uplift.sh inside a given repo with the given base and head SHAs,
# returning the resulting uplift value ("true" / "false") parsed from the
# script's GITHUB_OUTPUT writes.
run_detect() {
    local repo="$1"
    local base="$2"
    local head="$3"
    local gh_out
    gh_out=$(mktemp)
    (cd "$repo" && GITHUB_OUTPUT="$gh_out" .github/scripts/detect-uplift.sh "$base" "$head") >/dev/null 2>&1
    local rc=$?
    local val
    val=$(grep '^uplift=' "$gh_out" | sed 's/^uplift=//' || echo "ERR")
    rm -f "$gh_out"
    if [ "$rc" -ne 0 ]; then
        echo "EXITNZ"
    else
        echo "$val"
    fi
}

# --- Per-path uplift detection ---
for path_to_change in \
    "third-party/tt-metal-version" \
    "third-party/llvm-project/sentinel" \
    "third-party/tt-metal/sentinel" \
    ".github/containers/Dockerfile.base" \
    "requirements-runtime.txt"; do
    start_case "diff in $path_to_change marks uplift=true"
    repo=$(mkrepo)
    trap 'rm -rf "$repo"' EXIT
    install_scripts_in_repo "$repo"
    base=$(cd "$repo" && git rev-parse HEAD)
    echo "modified" >> "$repo/$path_to_change"
    (cd "$repo" && git add -A && git commit -q -m "uplift")
    head=$(cd "$repo" && git rev-parse HEAD)
    assert_eq "$(run_detect "$repo" "$base" "$head")" "true" "$path_to_change: uplift=true"
    rm -rf "$repo"
    trap - EXIT
done

# --- No-diff case ---
start_case "same base and head -> uplift=false"
repo=$(mkrepo)
trap 'rm -rf "$repo"' EXIT
install_scripts_in_repo "$repo"
sha=$(cd "$repo" && git rev-parse HEAD)
assert_eq "$(run_detect "$repo" "$sha" "$sha")" "false" "BASE == HEAD"
rm -rf "$repo"
trap - EXIT

# --- Diff in non-uplift path only ---
start_case "diff in non-uplift path -> uplift=false"
repo=$(mkrepo)
trap 'rm -rf "$repo"' EXIT
install_scripts_in_repo "$repo"
base=$(cd "$repo" && git rev-parse HEAD)
mkdir -p "$repo/lib"
echo "kernel change" > "$repo/lib/something.cpp"
(cd "$repo" && git add -A && git commit -q -m "kernel")
head=$(cd "$repo" && git rev-parse HEAD)
assert_eq "$(run_detect "$repo" "$base" "$head")" "false" "lib/ change"
rm -rf "$repo"
trap - EXIT

# --- Regression: tt-mlir is NOT uplift (built fresh by call-build.yml) ---
# Guards against a future "is tt-mlir uplift?" mistake re-adding it to
# UPLIFT_PATHS.
start_case "diff in third-party/tt-mlir alone -> uplift=false"
repo=$(mkrepo)
trap 'rm -rf "$repo"' EXIT
install_scripts_in_repo "$repo"
base=$(cd "$repo" && git rev-parse HEAD)
mkdir -p "$repo/third-party/tt-mlir"
echo "tt-mlir bump" > "$repo/third-party/tt-mlir/sentinel"
(cd "$repo" && git add -A && git commit -q -m "tt-mlir-only")
head=$(cd "$repo" && git rev-parse HEAD)
assert_eq "$(run_detect "$repo" "$base" "$head")" "false" "tt-mlir-only change"
rm -rf "$repo"
trap - EXIT

# --- Regression: pyproject.toml is NOT uplift (covered by wheel filter,
# not container content) ---
start_case "diff in pyproject.toml alone -> uplift=false"
repo=$(mkrepo)
trap 'rm -rf "$repo"' EXIT
install_scripts_in_repo "$repo"
base=$(cd "$repo" && git rev-parse HEAD)
echo "[project]" > "$repo/pyproject.toml"
(cd "$repo" && git add -A && git commit -q -m "pyproject-only")
head=$(cd "$repo" && git rev-parse HEAD)
assert_eq "$(run_detect "$repo" "$base" "$head")" "false" "pyproject-only change"
rm -rf "$repo"
trap - EXIT

# --- Mixed: uplift path + non-uplift path -> uplift=true ---
start_case "diff in both uplift and non-uplift paths -> uplift=true"
repo=$(mkrepo)
trap 'rm -rf "$repo"' EXIT
install_scripts_in_repo "$repo"
base=$(cd "$repo" && git rev-parse HEAD)
echo "new tt-metal" > "$repo/third-party/tt-metal-version"
mkdir -p "$repo/lib"
echo "kernel" > "$repo/lib/foo.cpp"
(cd "$repo" && git add -A && git commit -q -m "mixed")
head=$(cd "$repo" && git rev-parse HEAD)
assert_eq "$(run_detect "$repo" "$base" "$head")" "true" "mixed change"
rm -rf "$repo"
trap - EXIT

# --- Subdir-CWD regression: same result from a subdirectory ---
# Without `cd $(git rev-parse --show-toplevel)`, `git diff -- <paths>` from a
# subdir interprets the paths relative to the subdir and silently produces an
# empty result.
start_case "uplift detection is CWD-invariant (subdir regression)"
repo=$(mkrepo)
trap 'rm -rf "$repo"' EXIT
install_scripts_in_repo "$repo"
base=$(cd "$repo" && git rev-parse HEAD)
echo "new tt-metal" > "$repo/third-party/tt-metal-version"
(cd "$repo" && git add -A && git commit -q -m "uplift")
head=$(cd "$repo" && git rev-parse HEAD)
mkdir -p "$repo/lib/subdir"
gh_out=$(mktemp)
(cd "$repo/lib/subdir" && GITHUB_OUTPUT="$gh_out" "$repo/.github/scripts/detect-uplift.sh" "$base" "$head") >/dev/null 2>&1
val=$(grep '^uplift=' "$gh_out" | sed 's/^uplift=//' || echo "ERR")
rm -f "$gh_out"
assert_eq "$val" "true" "subdir CWD still detects uplift"
rm -rf "$repo"
trap - EXIT

# --- Missing arguments are rejected ---
start_case "missing base sha errors out"
repo=$(mkrepo)
trap 'rm -rf "$repo"' EXIT
install_scripts_in_repo "$repo"
gh_out=$(mktemp)
set +e
(cd "$repo" && GITHUB_OUTPUT="$gh_out" .github/scripts/detect-uplift.sh) >/dev/null 2>&1
rc=$?
set -e
rm -f "$gh_out"
assert_neq "$rc" "0" "no args: non-zero exit"
rm -rf "$repo"
trap - EXIT

start_case "missing head sha errors out"
repo=$(mkrepo)
trap 'rm -rf "$repo"' EXIT
install_scripts_in_repo "$repo"
base=$(cd "$repo" && git rev-parse HEAD)
gh_out=$(mktemp)
set +e
(cd "$repo" && GITHUB_OUTPUT="$gh_out" .github/scripts/detect-uplift.sh "$base") >/dev/null 2>&1
rc=$?
set -e
rm -f "$gh_out"
assert_neq "$rc" "0" "one arg: non-zero exit"
rm -rf "$repo"
trap - EXIT

finish_tests
