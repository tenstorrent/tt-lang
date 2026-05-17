#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared test helpers for .github/scripts/tests/. Source from each test file.

set -uo pipefail

# Where the scripts under test live, relative to this file.
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$TESTS_DIR")"
CONTAINERS_DIR="$(dirname "$SCRIPTS_DIR")/containers"

# Track pass/fail counts. Test files use these and exit with status 1 if any
# assertion failed.
TEST_PASSES=0
TEST_FAILURES=0
CURRENT_TEST=""

# Begin a named test case.
start_case() {
    CURRENT_TEST="$1"
}

# assert_eq <actual> <expected> [label]
assert_eq() {
    local actual="$1"
    local expected="$2"
    local label="${3:-${CURRENT_TEST:-assertion}}"
    if [[ "$actual" == "$expected" ]]; then
        echo "  PASS: $label"
        TEST_PASSES=$((TEST_PASSES + 1))
    else
        echo "  FAIL: $label"
        echo "    expected: $(printf '%q' "$expected")"
        echo "    actual:   $(printf '%q' "$actual")"
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
}

# assert_neq <actual> <unexpected> [label]
assert_neq() {
    local actual="$1"
    local unexpected="$2"
    local label="${3:-${CURRENT_TEST:-assertion}}"
    if [[ "$actual" != "$unexpected" ]]; then
        echo "  PASS: $label"
        TEST_PASSES=$((TEST_PASSES + 1))
    else
        echo "  FAIL: $label"
        echo "    must differ from: $(printf '%q' "$unexpected")"
        echo "    actual:           $(printf '%q' "$actual")"
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
}

# assert_matches <actual> <regex> [label]
assert_matches() {
    local actual="$1"
    local regex="$2"
    local label="${3:-${CURRENT_TEST:-assertion}}"
    if [[ "$actual" =~ $regex ]]; then
        echo "  PASS: $label"
        TEST_PASSES=$((TEST_PASSES + 1))
    else
        echo "  FAIL: $label"
        echo "    regex:  $regex"
        echo "    actual: $(printf '%q' "$actual")"
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
}

# assert_exit <expected_code> -- command args...
# Runs the command, captures its exit code, asserts.
assert_exit() {
    local expected="$1"
    shift
    [[ "$1" == "--" ]] && shift
    set +e
    "$@" >/dev/null 2>&1
    local rc=$?
    set -e
    assert_eq "$rc" "$expected" "${CURRENT_TEST:-exit}: exit code"
}

# Print summary and exit 0 (pass) or 1 (fail).
finish_tests() {
    local total=$((TEST_PASSES + TEST_FAILURES))
    echo ""
    echo "--- $(basename "${0}") summary: $TEST_PASSES/$total passed, $TEST_FAILURES failed ---"
    [[ $TEST_FAILURES -eq 0 ]] || return 1
    return 0
}

# Build a synthetic git repo. Creates a temp dir, inits a repo, commits a
# minimal initial set of files (the five uplift paths + a non-uplift sentinel
# under python/sim/ used by tests that need a non-uplift change).
# Echos the path to the repo. Caller is responsible for cleanup (use `trap`).
#
# Usage:
#   repo=$(mkrepo)
#   trap 'rm -rf "$repo"' EXIT
mkrepo() {
    local tmpdir
    tmpdir=$(mktemp -d)
    (
        cd "$tmpdir"
        git init -q -b main
        git config user.email t@t
        git config user.name t
        mkdir -p third-party/llvm-project third-party/tt-metal .github/containers python/sim
        echo v0.69.0 > third-party/tt-metal-version
        echo "llvm-content-v1" > third-party/llvm-project/sentinel
        echo "tt-metal-content-v1" > third-party/tt-metal/sentinel
        cat > .github/containers/Dockerfile.base <<'EOF'
FROM ubuntu:22.04
RUN echo "base v1"
EOF
        echo "greenlet>=3.0.0" > requirements-runtime.txt
        echo "// kernel placeholder" > python/sim/example.py
        git add -A
        git commit -q -m "initial"
    )
    echo "$tmpdir"
}

# Copy the scripts under test (and their dependencies) into the repo so they
# resolve their own paths correctly. The synthetic repo must look like a
# tt-lang checkout to the scripts.
install_scripts_in_repo() {
    local repo="$1"
    mkdir -p "$repo/.github/scripts" "$repo/.github/containers"
    cp "$SCRIPTS_DIR/uplift-paths.sh"          "$repo/.github/scripts/"
    cp "$SCRIPTS_DIR/detect-uplift.sh"         "$repo/.github/scripts/"
    cp "$SCRIPTS_DIR/require-release-tag.sh"   "$repo/.github/scripts/"
    cp "$SCRIPTS_DIR/verify-wheel-version.sh"  "$repo/.github/scripts/"
    cp "$CONTAINERS_DIR/get-version-tag.sh"    "$repo/.github/containers/"
}
