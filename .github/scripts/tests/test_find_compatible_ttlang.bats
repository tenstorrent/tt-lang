#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/find-compatible-ttlang.sh.
#
# The per-candidate build + device gate is CI-only, so the tests source the
# script (its main() is guarded from running on source) and override the heavy
# hooks (evaluate_candidate, commit_epoch, resolve_wheel_version) to exercise
# the walk / date-gap / cap / first-pass-wins logic without a toolchain.

load test_helper

# A synthetic tt-lang repo with a linear first-parent history newest..oldest
# = c5 c4 c3 c2 c1. Echoes the repo path.
make_ttlang_repo() {
    local dir
    dir=$(mktemp -d "$BATS_TEST_TMPDIR/ttlang.XXXXXX")
    (
        cd "$dir"
        git init -q -b main
        git config user.email t@t
        git config user.name t
        for idx in 1 2 3 4 5; do
            echo "commit $idx" > file.txt
            git add -A
            git commit -q -m "c$idx"
        done
    )
    echo "$dir"
}

# A tt-lang repo with one commit per given ISO date (oldest first). Echoes the
# repo path. Committer dates are fixed so date anchoring is deterministic.
make_dated_ttlang_repo() {
    local dir
    dir=$(mktemp -d "$BATS_TEST_TMPDIR/ttlang-dated.XXXXXX")
    (
        cd "$dir"
        git init -q -b main
        git config user.email t@t
        git config user.name t
        for iso in "$@"; do
            echo "$iso" > file.txt
            git add -A
            GIT_AUTHOR_DATE="${iso}T00:00:00+0000" \
            GIT_COMMITTER_DATE="${iso}T00:00:00+0000" \
                git commit -q -m "$iso"
        done
    )
    echo "$dir"
}

FUTURE_EPOCH=9999999999   # anchor past every synthetic commit

setup() {
    SCRIPT="$SCRIPTS_DIR/find-compatible-ttlang.sh"
    GH_OUT="$BATS_TEST_TMPDIR/gh_out"
    : > "$GH_OUT"
    source "$SCRIPT"
}

# --- Pure helpers ---

@test "abs_day_gap is symmetric and in whole days" {
    assert_equal "$(abs_day_gap 1000000 1000000)" "0"
    assert_equal "$(abs_day_gap 1086400 1000000)" "1"
    assert_equal "$(abs_day_gap 1000000 1086400)" "1"
}

@test "iso_to_epoch parses an ISO-8601 committer date" {
    expected=$(date -d "2026-06-15T00:00:00+00:00" +%s)
    run -0 iso_to_epoch "2026-06-15T00:00:00+00:00"
    assert_output "$expected"
}

@test "candidate_shas returns first-parent newest->oldest, honoring the cap" {
    repo=$(make_ttlang_repo)
    run -0 candidate_shas "$repo" HEAD 5 "$FUTURE_EPOCH"
    assert_equal "${#lines[@]}" "5"
    # Newest commit (c5) is first.
    top=$(git -C "$repo" rev-parse HEAD)
    assert_equal "${lines[0]}" "$top"

    run -0 candidate_shas "$repo" HEAD 2 "$FUTURE_EPOCH"
    assert_equal "${#lines[@]}" "2"
}

@test "candidate_shas anchors at the date window, skipping newer commits" {
    repo=$(make_dated_ttlang_repo 2020-01-01 2020-02-01 2020-03-01 2020-04-01)
    # Upper edge 2020-02-15 excludes the Mar and Apr commits.
    before=$(date -d "2020-02-15T00:00:00+00:00" +%s)
    run -0 candidate_shas "$repo" HEAD 40 "$before"
    assert_equal "${#lines[@]}" "2"
    feb=$(git -C "$repo" log --format=%H --all -1 --grep 2020-02-01)
    assert_equal "${lines[0]}" "$feb"
}

# --- Selection walk (stubbed evaluation) ---

@test "select_winner: first candidate that passes wins" {
    CANDIDATES=(c5 c4 c3 c2 c1)
    TTLANG_DIR="unused"
    commit_epoch() { echo 1000000; }        # all in-window
    evaluate_candidate() { [[ "$1" == "c4" ]]; }   # c5 fails, c4 passes

    if select_winner 1000000 14; then found=1; else found=0; fi
    assert_equal "$found" "1"
    assert_equal "$WINNER_SHA" "c4"
}

@test "select_winner: date-gap stops the walk before an out-of-window build" {
    CANDIDATES=(c5 c4 c3)
    TTLANG_DIR="unused"
    # c5,c4 in-window but fail; c3 is ~11 days older than tt-metal.
    commit_epoch() {
        case "$2" in
            c5) echo 1000000 ;;
            c4) echo 1000000 ;;
            c3) echo 100 ;;
        esac
    }
    evaluate_candidate() { return 1; }
    calls=0
    evaluate_candidate() { calls=$((calls + 1)); return 1; }

    if select_winner 1000000 1; then found=1; else found=0; fi
    assert_equal "$found" "0"
    assert_equal "$WINNER_SHA" ""
    # c3 must NOT have been evaluated (date-gap stops first).
    assert_equal "$calls" "2"
}

@test "select_winner: no pass within the cap returns failure" {
    CANDIDATES=(c5 c4 c3)
    TTLANG_DIR="unused"
    commit_epoch() { echo 1000000; }
    evaluate_candidate() { return 1; }

    if select_winner 1000000 14; then found=1; else found=0; fi
    assert_equal "$found" "0"
    assert_equal "$WINNER_SHA" ""
}

# --- main() end-to-end with stubbed evaluation ---

@test "main emits winner outputs when a candidate passes" {
    repo=$(make_dated_ttlang_repo 2026-06-10 2026-06-11 2026-06-12 2026-06-13 2026-06-14)
    winner=$(git -C "$repo" rev-parse HEAD~1)
    resolve_wheel_version() { echo "9.9.9.dev20260630"; }
    # Pass only on the second-newest commit; real commit_epoch keeps it in-window.
    evaluate_candidate() { [[ "$1" == "$winner" ]]; }
    export GITHUB_OUTPUT="$GH_OUT"

    run -0 main --ttmetal-install-dir /some/install \
        --ttmetal-date "2026-06-15T00:00:00+00:00" \
        --ttlang-dir "$repo"

    run cat "$GH_OUT"
    assert_output --partial "found=true"
    assert_output --partial "winner_sha=$winner"
    assert_output --partial "winner_version=9.9.9.dev20260630"
}

@test "main emits found=false when nothing passes" {
    repo=$(make_dated_ttlang_repo 2026-06-10 2026-06-11 2026-06-12 2026-06-13 2026-06-14)
    evaluate_candidate() { return 1; }
    export GITHUB_OUTPUT="$GH_OUT"

    run -0 main --ttmetal-install-dir /some/install \
        --ttmetal-date "2026-06-15T00:00:00+00:00" \
        --ttlang-dir "$repo"

    run cat "$GH_OUT"
    assert_output --partial "found=false"
    refute_output --partial "winner_sha="
}

@test "main matches an OLD tt-metal against old tt-lang (anchored walk)" {
    # HEAD is 2020-04; the tt-metal is 2020-02. The walk must anchor at the
    # tt-metal era and pick the newest in-window commit (2020-02), not report
    # incompatible because HEAD is far away.
    repo=$(make_dated_ttlang_repo 2020-01-01 2020-02-01 2020-03-01 2020-04-01)
    feb=$(git -C "$repo" log --format=%H --all -1 --grep 2020-02-01)
    resolve_wheel_version() { echo "1.2.3.dev20200201"; }
    evaluate_candidate() { [[ "$1" == "$feb" ]]; }   # real commit_epoch used
    export GITHUB_OUTPUT="$GH_OUT"

    run -0 main --ttmetal-install-dir /some/install \
        --ttmetal-date "2020-02-10T00:00:00+00:00" \
        --max-age-days 14 \
        --ttlang-dir "$repo"

    run cat "$GH_OUT"
    assert_output --partial "found=true"
    assert_output --partial "winner_sha=$feb"
}

@test "main requires --ttmetal-install-dir and --ttmetal-date (exit 2)" {
    run main --ttmetal-date "2026-06-15T00:00:00+00:00"
    assert_equal "$status" 2
}
