#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Validate the public PyPI publish target and emit the wheel version used by
# the shared manylinux build.

set -euo pipefail

: "${EVENT_NAME:?EVENT_NAME is required}"
: "${RELEASE_SOURCE:?RELEASE_SOURCE is required}"

dry_run="${DRY_RUN:-false}"
docker_tag="${DOCKER_TAG:-}"
ttlang_sha="${TTLANG_SHA:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/docker-image-utils.sh
. "$script_dir/lib/docker-image-utils.sh"
output_file="${GITHUB_OUTPUT:-/dev/stdout}"

case "$EVENT_NAME" in
    push | workflow_dispatch) ;;
    *)
        echo "Unsupported public PyPI event: $EVENT_NAME" >&2
        exit 1
        ;;
esac
case "$dry_run" in
    true | false) ;;
    *)
        echo "DRY_RUN must be true or false" >&2
        exit 2
        ;;
esac
if [[ -n "$docker_tag" ]] && ! ttlang_validate_docker_tag "$docker_tag"; then
    echo "DOCKER_TAG is not a valid Docker tag" >&2
    exit 2
fi

if [[ "$EVENT_NAME" == "workflow_dispatch" ]]; then
    echo "dry_run=$dry_run"
    echo "docker_tag=$docker_tag"
    echo "ttlang_sha=$ttlang_sha"

    if [[ ! "$ttlang_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "ttlang_sha must be a full 40-character commit SHA." >&2
        exit 1
    fi
    checked_out_sha="$(git -C "$RELEASE_SOURCE" rev-parse HEAD)"
    if [[ "$checked_out_sha" != "${ttlang_sha,,}" ]]; then
        echo "Checked-out commit does not match ttlang_sha." >&2
        exit 1
    fi
fi

if [[ "$EVENT_NAME" == "workflow_dispatch" && "$dry_run" != true ]]; then
    if [[ "${GITHUB_REF:-}" != refs/heads/main ]]; then
        echo "Public PyPI publishing is restricted to workflow dispatches from refs/heads/main." >&2
        exit 1
    fi
    if ! git -C "$RELEASE_SOURCE" merge-base --is-ancestor "$ttlang_sha" "${GITHUB_SHA:?GITHUB_SHA is required}"; then
        echo "ttlang_sha must be an ancestor of the dispatching main commit." >&2
        exit 1
    fi
fi

tag_version=""
if [[ "$EVENT_NAME" == push || "$dry_run" != true ]]; then
    release_ref="${GITHUB_REF:-}"
    if [[ -n "$ttlang_sha" ]]; then
        mapfile -t release_tags < <(
            git -C "$RELEASE_SOURCE" tag --list 'v[0-9]*' --points-at "$ttlang_sha"
        )
        if [[ "${#release_tags[@]}" -ne 1 ]]; then
            echo "ttlang_sha must have exactly one v* release tag; found ${#release_tags[@]}." >&2
            exit 1
        fi
        release_ref="refs/tags/${release_tags[0]}"
    fi
    tag_version="$(GITHUB_OUTPUT='' "$script_dir/require-release-tag.sh" "$release_ref")"
    wheel_version="$tag_version"
else
    wheel_version="$(
        cd "$RELEASE_SOURCE"
        python3 "$script_dir/compute-nightly-version.py"
    )"
fi

{
    echo "tag_version=$tag_version"
    echo "wheel_version=$wheel_version"
} >> "$output_file"

echo "Resolved public PyPI wheel version: $wheel_version"
