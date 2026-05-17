#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/verify-wheel-version.sh.

set -uo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

SCRIPT="$SCRIPTS_DIR/verify-wheel-version.sh"

# Synthetic version under test and the wheel-filename template that wraps it.
# `VER` is chosen well outside any real release range so the literals can
# never be confused with a production version. `WRONG_VER` is the mismatch
# input used in fail-path cases. `whl <version>` and `whl_sim <version>`
# produce the matching filenames for the two distributions; tests pass these
# to `run_verify`.
VER="99.99.99"
WRONG_VER="99.99.98"
PYTAG="cp312-cp312-linux_x86_64"
whl()     { printf 'tt_lang-%s-%s.whl' "$1" "$PYTAG"; }
whl_sim() { printf 'tt_lang_sim-%s-py3-none-any.whl' "$1"; }
whl_build() { printf 'tt_lang-%s-%s-%s.whl' "$1" "$2" "$PYTAG"; }  # <ver> <build>


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
"$SCRIPT" "$VER" >/dev/null 2>&1
rc=$?
set -e
assert_eq "$rc" "2" "one arg"

# --- Wheel-presence cases ---

start_case "empty wheel directory -> error (exit 1)"
rc=$(run_verify "$VER")
assert_eq "$rc" "1" "empty dir"

# --- Match cases ---

start_case "single matching wheel passes"
rc=$(run_verify "$VER" "$(whl "$VER")")
assert_eq "$rc" "0" "single match"

start_case "multiple matching wheels pass"
rc=$(run_verify "$VER" "$(whl "$VER")" "$(whl_sim "$VER")")
assert_eq "$rc" "0" "multiple matches"

# --- Mismatch cases ---

start_case "single mismatched wheel fails"
rc=$(run_verify "$VER" "$(whl "$WRONG_VER")")
assert_eq "$rc" "1" "single mismatch"

start_case "one match + one mismatch -> fails"
rc=$(run_verify "$VER" "$(whl "$VER")" "$(whl_sim "$WRONG_VER")")
assert_eq "$rc" "1" "mixed pass/fail"

# --- PEP 440 version forms ---

for form in "${VER}.dev20260515:dev YYYYMMDD" \
            "${VER}rc1:rc1" \
            "${VER}+uplift:+uplift" \
            "${VER}.post1:post1"; do
    v="${form%%:*}"
    label="${form#*:}"
    start_case "PEP 440 form: $label matches"
    rc=$(run_verify "$v" "$(whl "$v")")
    assert_eq "$rc" "0" "$label"
done

# PEP 427 wheel filenames may carry an optional build-number field between
# the version and the python tag: {name}-{version}-{build}-{python}-{abi}-{plat}.whl.
# The verifier extracts field 2 (version); field 3 (build number) must not
# be mistaken for the version.
start_case "PEP 427 build-number suffix: version still extracted from field 2"
rc=$(run_verify "$VER" "$(whl_build "$VER" 1)")
assert_eq "$rc" "0" "build-number wheel matches expected version"

start_case "PEP 427 build-number suffix: mismatch on version (not build-number) still fails"
rc=$(run_verify "$VER" "$(whl_build "$WRONG_VER" 1)")
assert_eq "$rc" "1" "version $WRONG_VER with build 1 must not satisfy expected $VER"

start_case "PEP 427 build-number suffix: expected version must not match the build-number field"
# If the verifier mistakenly compared field 3, this wheel's '7' would match
# expected '7'. Field 2 is the real version ($VER), so this must NOT match.
rc=$(run_verify "7" "$(whl_build "$VER" 7)")
assert_eq "$rc" "1" "expected '7' must not match the build-number field"

finish_tests
