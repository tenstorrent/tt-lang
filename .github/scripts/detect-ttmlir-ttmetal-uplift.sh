#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Detect whether tt-lang should build a light wheel for tt-mlir's pinned
# tt-metal SHA, keyed by the tt-lang/<ttmetal7> S3 prefix.
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
# Usage: detect-ttmlir-ttmetal-uplift.sh [--cmakelists <path>] [--ttlang-head <sha>] [--assume-new]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

S3_BUCKET="${TTLANG_S3_BUCKET:-tenstorrent-pypi}"

read_target_ttmetal_sha() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "tt-mlir third_party CMakeLists not found: $file" >&2
        echo "Is the tt-mlir submodule checked out?" >&2
        return 1
    fi
    local sha
    sha="$(sed -n -E 's/.*set\(TT_METAL_VERSION[[:space:]]+"([a-f0-9]+)".*/\1/p' "$file" \
        | head -n1)"
    if [[ -z "$sha" ]]; then
        echo "No 'set(TT_METAL_VERSION \"<sha>\")' found in $file" >&2
        return 1
    fi
    printf '%s\n' "$sha"
}

# Object basenames under the per-SHA wheel prefix. Overridable in tests.
list_prefix_objects() {
    aws s3 ls "s3://$S3_BUCKET/tt-lang/$1/" 2>/dev/null | awk '{print $NF}'
}

# tt-lang HEAD from a prior miss marker, or empty. Overridable in tests.
read_recorded_head() {
    aws s3 cp "s3://$S3_BUCKET/tt-lang/$1/attempt.json" - 2>/dev/null \
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
    local cmakelists="$repo_root/third-party/tt-mlir/third_party/CMakeLists.txt"
    local ttlang_head=""
    local assume_new=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cmakelists)
                [[ $# -ge 2 ]] || { echo "Usage: $0 [--cmakelists <path>] [--ttlang-head <sha>] [--assume-new]" >&2; return 2; }
                cmakelists="$2"; shift 2 ;;
            --ttlang-head)
                [[ $# -ge 2 ]] || { echo "Usage: $0 [--cmakelists <path>] [--ttlang-head <sha>] [--assume-new]" >&2; return 2; }
                ttlang_head="$2"; shift 2 ;;
            --assume-new)
                assume_new=true; shift ;;
            *)
                echo "Unknown argument: $1" >&2
                echo "Usage: $0 [--cmakelists <path>] [--ttlang-head <sha>] [--assume-new]" >&2
                return 2 ;;
        esac
    done

    if [[ -z "$ttlang_head" ]]; then
        ttlang_head="$(git -C "$repo_root" rev-parse HEAD)"
    fi

    local target_sha short_sha class
    if ! target_sha="$(read_target_ttmetal_sha "$cmakelists")"; then
        return 1
    fi
    short_sha="${target_sha:0:7}"
    echo "tt-mlir targets tt-metal $target_sha (prefix tt-lang/$short_sha); tt-lang HEAD $ttlang_head" >&2

    if [[ "$assume_new" == true ]]; then
        echo "Assuming $short_sha is unbuilt (--assume-new); skipping S3 idempotency check." >&2
        class="new"
    else
        class="$(classify_target "$short_sha" "$ttlang_head")"
    fi
    case "$class" in
        published)
            echo "Wheel already published under tt-lang/$short_sha; skipping." >&2
            emit "uplift=false" ;;
        doomed)
            echo "Recorded miss under tt-lang/$short_sha for the current tt-lang HEAD; skipping." >&2
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
