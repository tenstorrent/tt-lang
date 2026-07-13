#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Resolve a tt-lang IRD Docker image tag against the tags present in GHCR.
#
# Exact matches always win. With --allow-version-prefix-fallback, a missing
# bare release tag such as v1.1.2 may resolve to a single existing v1.1.2-*
# image tag. Multiple matches are ambiguous and require an explicit docker_tag.

set -euo pipefail

candidate=""
owner="${GITHUB_REPOSITORY_OWNER:-}"
tags_file=""
allow_version_prefix_fallback=false

usage() {
    echo "usage: resolve-ird-docker-tag.sh --candidate <tag> [--owner <org>] [--tags-file <file>] [--allow-version-prefix-fallback]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --candidate) [[ $# -ge 2 ]] || usage; candidate="$2"; shift 2 ;;
        --owner) [[ $# -ge 2 ]] || usage; owner="$2"; shift 2 ;;
        --tags-file) [[ $# -ge 2 ]] || usage; tags_file="$2"; shift 2 ;;
        --allow-version-prefix-fallback) allow_version_prefix_fallback=true; shift ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

trim() { printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
candidate="$(trim "$candidate")"
owner="$(trim "$owner")"

if [[ -z "$candidate" ]]; then
    echo "error: --candidate is required" >&2
    usage
fi

list_tags() {
    if [[ -n "$tags_file" ]]; then
        sed '/^[[:space:]]*$/d' "$tags_file"
        return
    fi

    if [[ -z "$owner" && -n "${GITHUB_REPOSITORY:-}" ]]; then
        owner="${GITHUB_REPOSITORY%%/*}"
    fi
    if [[ -z "$owner" ]]; then
        echo "error: --owner is required unless GITHUB_REPOSITORY_OWNER or GITHUB_REPOSITORY is set" >&2
        exit 2
    fi
    if ! command -v gh >/dev/null 2>&1; then
        echo "error: gh is required to query GHCR tags" >&2
        exit 2
    fi

    gh api --paginate \
        "/orgs/${owner}/packages/container/tt-lang%2Ftt-lang-ird-ubuntu-24-04/versions?per_page=100" \
        --jq '.[].metadata.container.tags[]?'
}

tags_tmp="$(mktemp)"
trap 'rm -f "$tags_tmp"' EXIT
if ! list_tags > "$tags_tmp"; then
    exit 1
fi
# Normalize before matching: strip CRLF carriage returns and surrounding
# whitespace so a tags file with trailing spaces or CRLF still matches the
# trimmed candidate, then drop any lines left empty.
mapfile -t tags < <(
    sed -e 's/\r$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' "$tags_tmp" \
        | sort -u
)

for tag in "${tags[@]}"; do
    if [[ "$tag" == "$candidate" ]]; then
        printf '%s\n' "$candidate"
        exit 0
    fi
done

if [[ "$allow_version_prefix_fallback" != true ]]; then
    echo "error: IRD builder image tag does not exist: $candidate" >&2
    exit 1
fi

if [[ ! "$candidate" =~ ^v[0-9]+[.][0-9]+[.][0-9]+$ ]]; then
    echo "error: IRD builder image tag does not exist: $candidate" >&2
    exit 1
fi

matches=()
for tag in "${tags[@]}"; do
    if [[ "$tag" == "$candidate"-* ]]; then
        matches+=("$tag")
    fi
done

case "${#matches[@]}" in
    0)
        echo "error: no existing IRD image tag matches $candidate or $candidate-*" >&2
        exit 1
        ;;
    1)
        echo "Exact IRD image tag $candidate is missing; using ${matches[0]}" >&2
        printf '%s\n' "${matches[0]}"
        ;;
    *)
        echo "error: multiple IRD image tags match $candidate-*; pass docker_tag explicitly:" >&2
        printf '  %s\n' "${matches[@]}" >&2
        exit 1
        ;;
esac
