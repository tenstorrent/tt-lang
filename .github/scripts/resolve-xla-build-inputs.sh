#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Resolve the builder image tag and wheel version for an XLA light-wheel build
# from a pinned tt-lang checkout. Both are computed from that checkout using its
# own get-version-tag.sh and compute-nightly-version.py, so an older ref builds
# against the builder image and version that match it rather than the current
# workflow commit. Explicit --docker-tag / --version override the computation.
#
# Usage:
#   resolve-xla-build-inputs.sh --target-dir <dir> [--docker-tag <tag>] [--version <ver>]
#                                [--resolve-existing-docker-tag]
#                                [--docker-tags-file <file>]
#
# Emits to $GITHUB_OUTPUT (or stdout):
#   tag=<builder image tag>
#   version=<wheel base version>

set -euo pipefail

target_dir=""
docker_tag=""
version=""
resolve_existing_docker_tag=false
docker_tags_file=""

usage() {
    echo "usage: resolve-xla-build-inputs.sh --target-dir <dir> [--docker-tag <tag>] [--version <ver>] [--resolve-existing-docker-tag] [--docker-tags-file <file>]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-dir) [[ $# -ge 2 ]] || usage; target_dir="$2"; shift 2 ;;
        --docker-tag) [[ $# -ge 2 ]] || usage; docker_tag="$2"; shift 2 ;;
        --version)    [[ $# -ge 2 ]] || usage; version="$2";    shift 2 ;;
        --resolve-existing-docker-tag) resolve_existing_docker_tag=true; shift ;;
        --docker-tags-file) [[ $# -ge 2 ]] || usage; docker_tags_file="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

trim() { printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
docker_tag="$(trim "$docker_tag")"
version="$(trim "$version")"
docker_tag_override="$docker_tag"

if [[ -z "$target_dir" || ! -d "$target_dir" ]]; then
    echo "error: --target-dir must be an existing tt-lang checkout" >&2
    usage
fi

# The target ref's own get-version-tag.sh reproduces the tag its builder image
# was pushed under; the current workflow commit's tag would point at a newer,
# mismatched image.
if [[ -z "$docker_tag" ]]; then
    docker_tag="$(cd "$target_dir" && .github/containers/get-version-tag.sh)"
fi
if [[ -z "$version" ]]; then
    version="$(cd "$target_dir" && python3 .github/scripts/compute-nightly-version.py)"
fi

if [[ "$resolve_existing_docker_tag" == true ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    resolve_args=(--candidate "$docker_tag")
    if [[ -n "$docker_tags_file" ]]; then
        resolve_args+=(--tags-file "$docker_tags_file")
    else
        resolve_args+=(--owner "${TTLANG_IRD_DOCKER_OWNER:-${GITHUB_REPOSITORY_OWNER:-tenstorrent}}")
    fi
    if [[ -z "$docker_tag_override" ]]; then
        resolve_args+=(--allow-version-prefix-fallback)
    fi
    docker_tag="$("$script_dir/resolve-ird-docker-tag.sh" "${resolve_args[@]}")"
fi

emit() { printf '%s\n' "$1" >> "${GITHUB_OUTPUT:-/dev/stdout}"; }
emit "tag=$docker_tag"
emit "version=$version"
