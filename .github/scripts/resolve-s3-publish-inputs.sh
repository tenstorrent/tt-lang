#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Resolve the inputs of the S3 PyPI publish workflow into a single set of
# step outputs. Stable tag pushes use the tag version, scheduled runs compute
# a nightly version, and scheduled runs force `overwrite_releases=true`.
#
# Required env:
#   DISPATCH_DOCKER_TAG          May be empty (workflow_dispatch input).
#   DISPATCH_DRY_RUN             "true"|"false" (workflow_dispatch input).
#   DISPATCH_OVERWRITE_RELEASES  "true"|"false" (workflow_dispatch input).
#   DISPATCH_VERSION_OVERRIDE    PEP 440 string, may be empty.
#   DISPATCH_WHEEL_VARIANT       pypi|light|bundled|bundled-and-light, may be
#                                empty for non-dispatch events.
#   EVENT_NAME                   github.event_name.
#   GITHUB_REF                   github.ref, required for push events.
#   GITHUB_OUTPUT                Path that receives the resolved outputs.
#                                Falls back to stdout when unset.
#
# Outputs (written to $GITHUB_OUTPUT):
#   docker_tag, dry_run, overwrite_releases, version_override, wheel_variant,
#   wheel_variants, wheel_matrix, allow_final_internal_version

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/tt-metal-version-utils.sh
. "$script_dir/lib/tt-metal-version-utils.sh"

stable_tag_version() {
    local ref="$1"

    if [[ "$ref" =~ ^refs/tags/v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
        printf '%s.%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
        return 0
    fi

    echo "S3 release-tag publish requires a stable tag like refs/tags/vX.Y.Z (got '$ref')." >&2
    return 1
}

is_stable_version() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

variant_includes_bundled() {
    [[ "$1" == "bundled" || "$1" == "bundled-and-light" ]]
}

pypi_aligned() {
    load_tt_metal_version || exit 1
    ttnn_pypi_aligned
}

: "${DISPATCH_DRY_RUN:?DISPATCH_DRY_RUN is required}"
: "${DISPATCH_OVERWRITE_RELEASES:?DISPATCH_OVERWRITE_RELEASES is required}"
: "${EVENT_NAME:?EVENT_NAME is required}"
docker_tag="${DISPATCH_DOCKER_TAG:-}"
dry_run="$DISPATCH_DRY_RUN"
overwrite_releases="$DISPATCH_OVERWRITE_RELEASES"
version_override="${DISPATCH_VERSION_OVERRIDE:-}"
wheel_variant="${DISPATCH_WHEEL_VARIANT:-}"

if [[ -z "$version_override" ]]; then
    if [[ "$EVENT_NAME" == "push" ]]; then
        version_override=$(stable_tag_version "${GITHUB_REF:-}")
    else
        version_override=$(python3 "$script_dir/compute-nightly-version.py")
    fi
fi

if [[ -z "$wheel_variant" ]]; then
    case "$EVENT_NAME" in
        push)
            if pypi_aligned; then
                wheel_variant=light
            else
                wheel_variant=bundled-and-light
            fi
            ;;
        schedule)
            wheel_variant=bundled-and-light
            ;;
        *)
            echo "DISPATCH_WHEEL_VARIANT is required for $EVENT_NAME events" >&2
            exit 1
            ;;
    esac
fi

case "$wheel_variant" in
    bundled)
        wheel_variants='["bundled"]'
        wheel_matrix='{"include":[{"wheel_variant":"bundled","ttnn_dep_mode":"bundled"}]}'
        ;;
    light)
        wheel_variants='["light"]'
        wheel_matrix='{"include":[{"wheel_variant":"light","ttnn_dep_mode":"external"}]}'
        ;;
    bundled-and-light)
        wheel_variants='["bundled","light"]'
        wheel_matrix='{"include":[{"wheel_variant":"bundled","ttnn_dep_mode":"bundled"},{"wheel_variant":"light","ttnn_dep_mode":"external"}]}'
        ;;
    pypi)
        wheel_variants='["pypi"]'
        wheel_matrix='{"include":[{"wheel_variant":"pypi","ttnn_dep_mode":"pypi"}]}'
        ;;
    *)
        echo "Unknown S3 wheel variant: $wheel_variant" >&2
        exit 2
        ;;
esac

if is_stable_version "$version_override" && variant_includes_bundled "$wheel_variant" && pypi_aligned; then
    echo "Refusing to publish bundled tt-lang==$version_override to S3 because public PyPI publishing is aligned for this tt-metal tag." >&2
    echo "Use the light or pypi S3 variant, or use a distinct internal version." >&2
    exit 1
fi

allow_final_internal_version=false
if is_stable_version "$version_override"; then
    allow_final_internal_version=true
fi

if [[ "$EVENT_NAME" == "schedule" ]]; then
    overwrite_releases=true
fi

output_file="${GITHUB_OUTPUT:-/dev/stdout}"
{
    echo "docker_tag=$docker_tag"
    echo "dry_run=$dry_run"
    echo "overwrite_releases=$overwrite_releases"
    echo "version_override=$version_override"
    echo "wheel_variant=$wheel_variant"
    echo "wheel_variants=$wheel_variants"
    echo "wheel_matrix=$wheel_matrix"
    echo "allow_final_internal_version=$allow_final_internal_version"
} >> "$output_file"

echo "Resolved wheel_variant=$wheel_variant"
echo "Resolved wheel_variants=$wheel_variants"
echo "Resolved version_override=$version_override"
echo "Resolved allow_final_internal_version=$allow_final_internal_version"
echo "Resolved dry_run=$dry_run"
echo "Resolved overwrite_releases=$overwrite_releases"
if [[ -n "$docker_tag" ]]; then
    echo "Using existing docker_tag=$docker_tag"
else
    echo "No docker_tag provided; build-docker will create one"
fi
