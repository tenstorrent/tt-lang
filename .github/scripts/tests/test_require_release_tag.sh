#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/require-release-tag.sh — refusal of non-tag refs
# and PEP 440 normalization of accepted forms.

set -uo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

SCRIPT="$SCRIPTS_DIR/require-release-tag.sh"

# Run the script with a given GITHUB_REF and capture stdout, stderr, and exit.
# Echos "<rc>|<stdout>|<stderr>" delimited by '|'.
run_ref() {
    local ref="$1"
    local out err rc
    out=$(GITHUB_REF="$ref" GITHUB_OUTPUT=/dev/null "$SCRIPT" 2>/tmp/rrt_stderr)
    rc=$?
    err=$(cat /tmp/rrt_stderr)
    rm -f /tmp/rrt_stderr
    printf '%s|%s|%s' "$rc" "$out" "$err"
}

# Helper: assert that a given GITHUB_REF input produces the given normalized
# stdout and exit 0.
assert_accepts() {
    local ref="$1"
    local expected="$2"
    local label="$3"
    local result rc out
    result=$(run_ref "$ref")
    rc="${result%%|*}"
    rest="${result#*|}"
    out="${rest%%|*}"
    assert_eq "$rc" "0" "$label: exit 0"
    assert_eq "$out" "$expected" "$label: $ref -> $expected"
}

# Helper: assert that the given GITHUB_REF input is rejected (non-zero exit).
assert_rejects() {
    local ref="$1"
    local label="$2"
    local result rc
    result=$(run_ref "$ref")
    rc="${result%%|*}"
    assert_neq "$rc" "0" "$label: rejected"
}

# --- Rejection cases ---

start_case "rejects branch ref"
assert_rejects "refs/heads/main" "branch ref"

start_case "rejects empty ref"
assert_rejects "" "empty ref"

start_case "rejects non-version tag"
assert_rejects "refs/tags/somefeature" "non-version tag"

start_case "rejects tag without leading v"
assert_rejects "refs/tags/1.0.0" "no leading v"

# --- PEP 440 normalization cases ---

start_case "final release"
assert_accepts "refs/tags/v1.2.0" "1.2.0" "final"

start_case "patch release"
assert_accepts "refs/tags/v1.2.3" "1.2.3" "patch"

start_case "dev pre-release (tt-metal style)"
assert_accepts "refs/tags/v1.2.0-dev20260515" "1.2.0.dev20260515" "dev YYYYMMDD"

start_case "rc pre-release"
assert_accepts "refs/tags/v1.2.0-rc1" "1.2.0rc1" "rc1"

start_case "alpha pre-release"
assert_accepts "refs/tags/v1.2.0-alpha3" "1.2.0a3" "alpha3"

start_case "beta pre-release"
assert_accepts "refs/tags/v1.2.0-beta2" "1.2.0b2" "beta2"

start_case "post release"
assert_accepts "refs/tags/v1.2.0-post1" "1.2.0.post1" "post1"

start_case "local version label"
assert_accepts "refs/tags/v1.2.0+uplift" "1.2.0+uplift" "+uplift"

start_case "dev + local combined"
assert_accepts "refs/tags/v1.2.0-dev20260515+ci123" "1.2.0.dev20260515+ci123" "dev + local"

# --- GITHUB_OUTPUT writes ---

start_case "writes tag_version to GITHUB_OUTPUT when set"
gh_out=$(mktemp)
trap 'rm -f "$gh_out"' EXIT
GITHUB_REF="refs/tags/v1.2.0-rc1" GITHUB_OUTPUT="$gh_out" "$SCRIPT" >/dev/null
assert_eq "$(cat "$gh_out")" "tag_version=1.2.0rc1" "GITHUB_OUTPUT contents"
rm -f "$gh_out"
trap - EXIT

start_case "stdout matches GITHUB_OUTPUT line"
gh_out=$(mktemp)
trap 'rm -f "$gh_out"' EXIT
stdout=$(GITHUB_REF="refs/tags/v1.2.0-dev20260515" GITHUB_OUTPUT="$gh_out" "$SCRIPT")
out_line=$(grep '^tag_version=' "$gh_out" | sed 's/^tag_version=//')
assert_eq "$stdout" "$out_line" "stdout matches output file"
rm -f "$gh_out"
trap - EXIT

finish_tests
