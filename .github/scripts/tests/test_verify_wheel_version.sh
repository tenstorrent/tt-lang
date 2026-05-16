#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/verify-wheel-version.sh.

set -uo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

SCRIPT="$SCRIPTS_DIR/verify-wheel-version.sh"

# Helper: create a temp dir, drop in zero or more empty wheel files with the
# given names, run verify-wheel-version.sh, return exit code.
run_verify() {
    local expected="$1"
    shift
    local dir
    dir=$(mktemp -d)
    for name in "$@"; do
        : > "$dir/$name"
    done
    set +e
    "$SCRIPT" "$expected" "$dir" >/dev/null 2>&1
    local rc=$?
    set -e
    rm -rf "$dir"
    echo "$rc"
}

# --- Argument-validation cases ---

start_case "no args -> usage error (exit 2)"
set +e
"$SCRIPT" >/dev/null 2>&1
rc=$?
set -e
assert_eq "$rc" "2" "no args"

start_case "one arg -> usage error (exit 2)"
set +e
"$SCRIPT" "1.2.0" >/dev/null 2>&1
rc=$?
set -e
assert_eq "$rc" "2" "one arg"

# --- Wheel-presence cases ---

start_case "empty wheel directory -> error (exit 1)"
rc=$(run_verify "1.2.0")
assert_eq "$rc" "1" "empty dir"

# --- Match cases ---

start_case "single matching wheel passes"
rc=$(run_verify "1.2.0" "tt_lang-1.2.0-cp312-cp312-linux_x86_64.whl")
assert_eq "$rc" "0" "single match"

start_case "multiple matching wheels pass"
rc=$(run_verify "1.2.0" \
    "tt_lang-1.2.0-cp312-cp312-linux_x86_64.whl" \
    "tt_lang_sim-1.2.0-py3-none-any.whl")
assert_eq "$rc" "0" "multiple matches"

# --- Mismatch cases ---

start_case "single mismatched wheel fails"
rc=$(run_verify "1.2.0" "tt_lang-1.1.0-cp312-cp312-linux_x86_64.whl")
assert_eq "$rc" "1" "single mismatch"

start_case "one match + one mismatch -> fails"
rc=$(run_verify "1.2.0" \
    "tt_lang-1.2.0-cp312-cp312-linux_x86_64.whl" \
    "tt_lang_sim-1.1.0-py3-none-any.whl")
assert_eq "$rc" "1" "mixed pass/fail"

# --- PEP 440 version forms ---

start_case "dotted dev version (PEP 440 normalized) matches"
rc=$(run_verify "1.2.0.dev20260515" \
    "tt_lang-1.2.0.dev20260515-cp312-cp312-linux_x86_64.whl")
assert_eq "$rc" "0" "dev YYYYMMDD"

start_case "rc version (PEP 440 normalized) matches"
rc=$(run_verify "1.2.0rc1" \
    "tt_lang-1.2.0rc1-cp312-cp312-linux_x86_64.whl")
assert_eq "$rc" "0" "rc1"

start_case "local-label version matches"
rc=$(run_verify "1.2.0+uplift" \
    "tt_lang-1.2.0+uplift-cp312-cp312-linux_x86_64.whl")
assert_eq "$rc" "0" "+uplift"

start_case "post-release matches"
rc=$(run_verify "1.2.0.post1" \
    "tt_lang-1.2.0.post1-cp312-cp312-linux_x86_64.whl")
assert_eq "$rc" "0" "post1"

# PEP 427 wheel filenames may carry an optional build-number field between
# the version and the python tag: {name}-{version}-{build}-{python}-{abi}-{plat}.whl.
# The verifier extracts field 2 (version); field 3 (build number) must not
# be mistaken for the version.
start_case "PEP 427 build-number suffix: version still extracted from field 2"
rc=$(run_verify "1.2.0" "tt_lang-1.2.0-1-cp312-cp312-linux_x86_64.whl")
assert_eq "$rc" "0" "build-number wheel matches expected version"

start_case "PEP 427 build-number suffix: mismatch on version (not build-number) still fails"
rc=$(run_verify "1.2.0" "tt_lang-1.1.0-1-cp312-cp312-linux_x86_64.whl")
assert_eq "$rc" "1" "version 1.1.0 with build 1 must not satisfy expected 1.2.0"

start_case "PEP 427 build-number suffix: expected version must not match the build-number field"
# If the verifier mistakenly compared field 3, this wheel's '7' would match
# expected '7'. Field 2 is the real version (1.2.0), so this must NOT match.
rc=$(run_verify "7" "tt_lang-1.2.0-7-cp312-cp312-linux_x86_64.whl")
assert_eq "$rc" "1" "expected '7' must not match the build-number field"

finish_tests
