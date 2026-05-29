#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Resolve the inputs of the S3 PyPI publish workflow into a single set of
# step outputs. Computes a nightly version when none was supplied, forces
# `overwrite_releases=true` for scheduled runs, and prints a one-line
# summary per resolved input.
#
# Required env:
#   DISPATCH_DOCKER_TAG          May be empty (workflow_dispatch input).
#   DISPATCH_DRY_RUN             "true"|"false" (workflow_dispatch input).
#   DISPATCH_OVERWRITE_RELEASES  "true"|"false" (workflow_dispatch input).
#   DISPATCH_VERSION_OVERRIDE    PEP 440 string, may be empty.
#   DISPATCH_TTNN_DEP_MODE       pypi|external|bundled.
#   EVENT_NAME                   github.event_name.
#   GITHUB_OUTPUT                Path that receives the resolved outputs.
#                                Falls back to stdout when unset.
#
# Outputs (written to $GITHUB_OUTPUT):
#   docker_tag, dry_run, overwrite_releases, version_override, ttnn_dep_mode

set -euo pipefail

: "${DISPATCH_DRY_RUN:?DISPATCH_DRY_RUN is required}"
: "${DISPATCH_OVERWRITE_RELEASES:?DISPATCH_OVERWRITE_RELEASES is required}"
: "${DISPATCH_TTNN_DEP_MODE:?DISPATCH_TTNN_DEP_MODE is required}"
: "${EVENT_NAME:?EVENT_NAME is required}"
docker_tag="${DISPATCH_DOCKER_TAG:-}"
dry_run="$DISPATCH_DRY_RUN"
overwrite_releases="$DISPATCH_OVERWRITE_RELEASES"
version_override="${DISPATCH_VERSION_OVERRIDE:-}"
ttnn_dep_mode="$DISPATCH_TTNN_DEP_MODE"

if [[ "$EVENT_NAME" == "schedule" ]]; then
    overwrite_releases=true
fi

if [[ -z "$version_override" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    version_override=$(python3 "$script_dir/compute-nightly-version.py")
fi

output_file="${GITHUB_OUTPUT:-/dev/stdout}"
{
    echo "docker_tag=$docker_tag"
    echo "dry_run=$dry_run"
    echo "overwrite_releases=$overwrite_releases"
    echo "version_override=$version_override"
    echo "ttnn_dep_mode=$ttnn_dep_mode"
} >> "$output_file"

echo "Resolved ttnn_dep_mode=$ttnn_dep_mode"
echo "Resolved version_override=$version_override"
echo "Resolved dry_run=$dry_run"
echo "Resolved overwrite_releases=$overwrite_releases"
if [[ -n "$docker_tag" ]]; then
    echo "Using existing docker_tag=$docker_tag"
else
    echo "No docker_tag provided; build-docker will create one"
fi
