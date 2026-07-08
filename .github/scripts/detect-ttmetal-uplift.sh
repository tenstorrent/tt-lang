#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Detect whether tt-lang should build a light wheel for the tt-metal tag
# recorded in third-party/tt-metal-version, keyed by the
# tt-lang/ttmetal/<ttmetal7> S3 prefix.
#
# A SHA is skipped when a wheel is already published, and also when a prior
# search recorded no compatible tt-lang AND the tt-lang HEAD it searched is
# unchanged (re-searching the same history would fail identically). Once tt-lang
# HEAD advances, a recorded miss is retried because newer commits may now match.
#
# --assume-new treats the pinned SHA as buildable without reading S3. Dry runs
# do not publish, and branch runs cannot use main-only OIDC credentials.
#
# Writes to $GITHUB_OUTPUT (or stdout):
#   uplift=true|false
#   tt_metal_sha=<full 40-char sha>   (only when uplift=true)
#   tt_metal_sha_short=<7-char>       (only when uplift=true)
#
# Usage: detect-ttmetal-uplift.sh [--version-file <path>] [--ttlang-head <sha>] [--assume-new]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
# shellcheck source=lib/tt-metal-version-utils.sh
. "$script_dir/lib/tt-metal-version-utils.sh"

S3_BUCKET="${TTLANG_S3_BUCKET:-tenstorrent-pypi}"
TT_METAL_REMOTE="${TT_METAL_REMOTE:-https://github.com/tenstorrent/tt-metal}"

resolve_ttmetal_tag_sha() {
    local tag="$1"
    git ls-remote --tags "$TT_METAL_REMOTE" \
        "refs/tags/$tag" "refs/tags/$tag^{}" \
        | awk '$2 ~ /\^\{\}$/ {deref=$1} $2 !~ /\^\{\}$/ {direct=$1} END {print (deref ? deref : direct)}'
}

read_target_ttmetal_sha() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "tt-metal version file not found: $file" >&2
        return 1
    fi
    load_tt_metal_version "$file"
    local sha
    sha="$(resolve_ttmetal_tag_sha "$TT_METAL_TAG")"
    if [[ -z "$sha" ]]; then
        echo "tt-metal has no release tag $TT_METAL_TAG" >&2
        return 1
    fi
    printf '%s\n' "$sha"
}

# Object basenames under the per-SHA wheel prefix. Overridable in tests.
list_prefix_objects() {
    aws s3 ls "s3://$S3_BUCKET/tt-lang/ttmetal/$1/" 2>/dev/null | awk '{print $NF}'
}

# tt-lang HEAD from a prior miss marker, or empty. Overridable in tests.
read_recorded_head() {
    aws s3 cp "s3://$S3_BUCKET/tt-lang/ttmetal/$1/attempt.json" - 2>/dev/null \
        | sed -n -E 's/.*"ttlang_head"[[:space:]]*:[[:space:]]*"([a-f0-9]+)".*/\1/p' \
        | head -n1
}

# Echo one of: published | doomed | retry | new.
classify_target() {
    local short="$1" head="$2" objects
    objects="$(list_prefix_objects "$short")"
    if grep -q '\.whl$' <<<"$objects"; then
        echo "published"
        return
    fi
    if grep -qx 'attempt.json' <<<"$objects"; then
        local recorded
        recorded="$(read_recorded_head "$short")"
        if [[ -n "$recorded" && "$recorded" == "$head" ]]; then
            echo "doomed"
        else
            echo "retry"
        fi
        return
    fi
    echo "new"
}

emit() {
    printf '%s\n' "$1" >> "${GITHUB_OUTPUT:-/dev/stdout}"
}

main() {
    local version_file="$repo_root/third-party/tt-metal-version"
    local ttlang_head=""
    local assume_new=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version-file)
                [[ $# -ge 2 ]] || { echo "Usage: $0 [--version-file <path>] [--ttlang-head <sha>] [--assume-new]" >&2; return 2; }
                version_file="$2"; shift 2 ;;
            --ttlang-head)
                [[ $# -ge 2 ]] || { echo "Usage: $0 [--version-file <path>] [--ttlang-head <sha>] [--assume-new]" >&2; return 2; }
                ttlang_head="$2"; shift 2 ;;
            --assume-new)
                assume_new=true; shift ;;
            *)
                echo "Unknown argument: $1" >&2
                echo "Usage: $0 [--version-file <path>] [--ttlang-head <sha>] [--assume-new]" >&2
                return 2 ;;
        esac
    done

    if [[ -z "$ttlang_head" ]]; then
        ttlang_head="$(git -C "$repo_root" rev-parse HEAD)"
    fi

    local target_sha short_sha class
    if ! target_sha="$(read_target_ttmetal_sha "$version_file")"; then
        return 1
    fi
    short_sha="${target_sha:0:7}"
    echo "tt-lang targets tt-metal $target_sha (prefix tt-lang/ttmetal/$short_sha); tt-lang HEAD $ttlang_head" >&2

    if [[ "$assume_new" == true ]]; then
        echo "Assuming $short_sha is unbuilt (--assume-new); skipping S3 idempotency check." >&2
        class="new"
    else
        class="$(classify_target "$short_sha" "$ttlang_head")"
    fi
    case "$class" in
        published)
            echo "Wheel already published under tt-lang/ttmetal/$short_sha; skipping." >&2
            emit "uplift=false" ;;
        doomed)
            echo "Recorded miss under tt-lang/ttmetal/$short_sha for the current tt-lang HEAD; skipping." >&2
            emit "uplift=false" ;;
        retry|new)
            echo "Building for $short_sha ($class)." >&2
            emit "uplift=true"
            emit "tt_metal_sha=$target_sha"
            emit "tt_metal_sha_short=$short_sha" ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
