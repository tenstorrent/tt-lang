#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Delete superseded compiler caches so they stop evicting the toolchain cache.
#
# GitHub Actions caches are immutable, so hendrikmuhs/ccache-action appends a
# timestamp when it saves an updated cache. Each entry can be restored until a
# newer prefix match supersedes it. The older entries still count against the
# repository's 10 GB cache budget and can cause unrelated toolchain caches to
# be evicted. Pull requests restore the default-branch caches without saving;
# this script bounds the history produced by rolling default-branch caches.
#
# Entries are grouped by ref and by key with the trailing timestamp removed; the
# newest KEEP entries of each group survive. Only keys matching PREFIX are ever
# considered, so toolchain caches are out of reach of this script.
#
# Usage:
#   .github/scripts/prune-actions-caches.sh                  # report only
#   .github/scripts/prune-actions-caches.sh --dry-run false  # delete
#   .github/scripts/prune-actions-caches.sh --keep 3 --prefix ccache-

set -euo pipefail

REPO="${GITHUB_REPOSITORY:-tenstorrent/tt-lang}"
KEEP=2
PREFIX="ccache-"
DRY_RUN=true

usage() {
    sed -n '/^# Usage:/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --keep)
        KEEP="${2:?--keep requires a count}"
        shift 2
        ;;
    --prefix)
        PREFIX="${2:?--prefix requires a key prefix}"
        shift 2
        ;;
    --dry-run)
        DRY_RUN="${2:?--dry-run requires true or false}"
        shift 2
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "prune-actions-caches.sh: unknown argument '$1'" >&2
        usage >&2
        exit 2
        ;;
    esac
done

case "$KEEP" in
'' | *[!0-9]*)
    echo "prune-actions-caches.sh: --keep must be a non-negative integer, got '$KEEP'" >&2
    exit 2
    ;;
esac

case "$DRY_RUN" in
true | false) ;;
*)
    echo "prune-actions-caches.sh: --dry-run must be true or false, got '$DRY_RUN'" >&2
    exit 2
    ;;
esac

# id, created_at, size, ref, key and series for every cache whose key starts
# with PREFIX. The series is the key without the action's timestamp suffix, so
# every run of one job on one ref lands in the same series. Sorted by series and
# then by creation time descending, so the entries restore-keys would pick come
# first in each group.
timestamp_suffix='-[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}Z$'
cache_rows=""
if ! cache_rows="$(
    gh api --paginate "repos/$REPO/actions/caches?per_page=100" \
        --jq ".actions_caches[] | select(.key | startswith(\"$PREFIX\")) |
              [.id, .created_at, .size_in_bytes, .ref, .key,
               (.key | sub(\"$timestamp_suffix\"; \"\"))] | @tsv" |
        sort -t"$(printf '\t')" -k4,4 -k6,6 -k2,2r
)"; then
    echo "prune-actions-caches.sh: failed to list caches for '$REPO'" >&2
    exit 1
fi

entries=()
if [[ -n "$cache_rows" ]]; then
    mapfile -t entries <<<"$cache_rows"
fi

if ((${#entries[@]} == 0)); then
    echo "no caches match prefix '$PREFIX'"
    exit 0
fi

deleted=0
freed=0
kept=0
previous_group=""
group_count=0

for entry in "${entries[@]}"; do
    IFS=$'\t' read -r id created size ref key series <<<"$entry"
    group="$ref|$series"

    if [[ "$group" != "$previous_group" ]]; then
        previous_group="$group"
        group_count=0
    fi
    group_count=$((group_count + 1))

    if ((group_count <= KEEP)); then
        kept=$((kept + 1))
        continue
    fi

    freed=$((freed + size))
    deleted=$((deleted + 1))
    if [[ "$DRY_RUN" == true ]]; then
        echo "would delete  $ref  $key  ($created)"
    else
        echo "deleting      $ref  $key  ($created)"
        gh api -X DELETE "repos/$REPO/actions/caches/$id" >/dev/null
    fi
done

gigabytes=$(awk "BEGIN {printf \"%.2f\", $freed / 1073741824}")
if [[ "$DRY_RUN" == true ]]; then
    echo "$REPO: kept $kept, would delete $deleted entries, would free $gigabytes GB"
else
    echo "$REPO: kept $kept, deleted $deleted entries, freed $gigabytes GB"
fi
