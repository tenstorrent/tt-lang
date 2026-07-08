#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Find the newest tt-lang commit on a first-parent history that builds against a
# given tt-metal install (TTLANG_EXTERNAL_TT_METAL_DIR) and passes a fast device
# gate. The first commit that builds and gates wins.
#
# Compatibility is a band in tt-lang history around the tt-metal's date (newer
# tt-lang expects a newer tt-metal API, older tt-lang an older one). The walk is
# anchored to that band, not to HEAD, so an old tt-metal SHA is matched against
# old tt-lang rather than reported incompatible:
#   - upper anchor: start at the newest first-parent commit dated at or before
#     ttmetal_date + max_age_days, then walk newest -> oldest. First pass wins,
#     which yields the newest compatible commit even if compatibility is not a
#     single contiguous run (a flaky gate or one-commit hole below does not end
#     the walk, unlike an oldest-first "stop at first break").
#   - lower stop (date-gap): stop once |candidate_date - ttmetal_date| exceeds
#     max_age_days; older commits are only farther.
#   - candidate cap: never consider more than --max-candidates commits, so a
#     misparsed date cannot expand into an unbounded build sweep.
# The anchor, date-gap stop, and cap are all logged; nothing is silently dropped.
#
# Writes to $GITHUB_OUTPUT (or stdout):
#   found=true|false
#   winner_sha=<sha>       (only when found=true)
#   winner_version=<ver>   (only when found=true)
#
# Usage:
#   find-compatible-ttlang.sh --ttmetal-install-dir <dir> --ttmetal-date <iso>
#       [--max-age-days <n>] [--max-candidates <n>] [--ttlang-dir <dir>]
#       [--ref <ref>] [--build-dir <dir>] [--gate-cmd <cmd>]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_repo_root="$(cd "$script_dir/../.." && pwd)"

log() { echo "[find-compatible] $*" >&2; }

iso_to_epoch() {
    date -d "$1" +%s
}

abs_day_gap() {
    local epoch_a="$1" epoch_b="$2" diff
    diff=$(( epoch_a - epoch_b ))
    (( diff < 0 )) && diff=$(( -diff ))
    echo $(( diff / 86400 ))
}

# First-parent commits at or older than $before_epoch, newest -> oldest, capped
# at $cap. The date anchor keeps the walk near the tt-metal's era for old SHAs.
candidate_shas() {
    local ttlang_dir="$1" ref="$2" cap="$3" before_epoch="$4"
    git -C "$ttlang_dir" rev-list --first-parent \
        --before="@$before_epoch" --max-count="$cap" "$ref"
}

commit_epoch() {
    git -C "$1" show -s --format=%ct "$2"
}

resolve_wheel_version() {
    local ttlang_dir="$1"
    ( cd "$ttlang_dir" && python3 .github/scripts/compute-nightly-version.py )
}

# Build the checked-out candidate against the external tt-metal and run the gate.
# Returns 0 when the candidate builds and gates cleanly. CI-only: needs the
# toolchain, submodules, and a device.
evaluate_candidate() {
    local sha="$1"
    log "evaluating $sha: checkout"
    git -C "$TTLANG_DIR" checkout --quiet --detach "$sha"

    log "evaluating $sha: configure + build"
    rm -rf "$BUILD_DIR"
    (
        cd "$TTLANG_DIR"
        TTLANG_EXTERNAL_TT_METAL_DIR="$TTMETAL_INSTALL_DIR" \
            .github/scripts/configure-ttlang-build.sh "$BUILD_DIR"
        cmake --build "$BUILD_DIR"
    ) || return 1

    log "evaluating $sha: fast device gate"
    (
        cd "$TTLANG_DIR"
        # env/activate is not nounset-safe; the gate runs like the hardware
        # workflow, without -u.
        set +u
        # shellcheck disable=SC1091
        source "$BUILD_DIR/env/activate"
        eval "$GATE_CMD"
    )
}

# Walk $CANDIDATES newest -> oldest, honoring the date-gap stop, and return the
# first that evaluate_candidate accepts. Sets WINNER_SHA on success.
select_winner() {
    local ttmetal_epoch="$1" max_age="$2"
    WINNER_SHA=""
    local sha cand_epoch gap
    for sha in "${CANDIDATES[@]}"; do
        cand_epoch="$(commit_epoch "$TTLANG_DIR" "$sha")"
        gap="$(abs_day_gap "$cand_epoch" "$ttmetal_epoch")"
        if (( gap > max_age )); then
            log "stop: $sha is ${gap}d from tt-metal (> ${max_age}d window); older commits are farther"
            return 1
        fi
        if evaluate_candidate "$sha"; then
            WINNER_SHA="$sha"
            return 0
        fi
        log "$sha failed build or gate; trying the next older commit"
    done
    log "no compatible commit within the ${#CANDIDATES[@]}-candidate cap"
    return 1
}

emit() {
    printf '%s\n' "$1" >> "${GITHUB_OUTPUT:-/dev/stdout}"
}

main() {
    TTMETAL_INSTALL_DIR=""
    local ttmetal_date=""
    local max_age_days=14
    local max_candidates=40
    TTLANG_DIR="$default_repo_root"
    local ref="HEAD"
    BUILD_DIR="build"
    GATE_CMD='python3 test/python/smoketest.py && python3 test/python/simple_add.py'

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --ttmetal-install-dir) TTMETAL_INSTALL_DIR="$2"; shift 2 ;;
            --ttmetal-date)        ttmetal_date="$2";        shift 2 ;;
            --max-age-days)        max_age_days="$2";        shift 2 ;;
            --max-candidates)      max_candidates="$2";      shift 2 ;;
            --ttlang-dir)          TTLANG_DIR="$2";          shift 2 ;;
            --ref)                 ref="$2";                 shift 2 ;;
            --build-dir)           BUILD_DIR="$2";           shift 2 ;;
            --gate-cmd)            GATE_CMD="$2";            shift 2 ;;
            *) echo "Unknown argument: $1" >&2; return 2 ;;
        esac
    done

    if [[ -z "$TTMETAL_INSTALL_DIR" || -z "$ttmetal_date" ]]; then
        echo "Usage: $0 --ttmetal-install-dir <dir> --ttmetal-date <iso> [options]" >&2
        return 2
    fi

    local ttmetal_epoch upper_epoch
    ttmetal_epoch="$(iso_to_epoch "$ttmetal_date")"
    upper_epoch=$(( ttmetal_epoch + max_age_days * 86400 ))

    mapfile -t CANDIDATES < <(candidate_shas "$TTLANG_DIR" "$ref" "$max_candidates" "$upper_epoch")
    log "considering ${#CANDIDATES[@]} first-parent candidate(s) from $ref at or before $(date -d "@$upper_epoch" -u +%Y-%m-%d) (cap $max_candidates)"

    if select_winner "$ttmetal_epoch" "$max_age_days"; then
        local version
        version="$(resolve_wheel_version "$TTLANG_DIR")"
        log "winner: $WINNER_SHA (version $version)"
        emit "found=true"
        emit "winner_sha=$WINNER_SHA"
        emit "winner_version=$version"
        return 0
    fi

    log "no compatible tt-lang commit found for this tt-metal"
    emit "found=false"
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
