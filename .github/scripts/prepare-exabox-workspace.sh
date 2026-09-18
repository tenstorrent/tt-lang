#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Copy the Actions checkout to Exabox shared storage and then to every worker.
#
# Env: STAGE_DIR and WORKER_SRC are required. GITHUB_WORKSPACE is required by
# the stage operation. EXABOX_REPORT_DIR optionally selects a shared report
# directory that every worker can write. CCACHE_DIR optionally selects a cache
# restored by the controller for worker builds.
# Usage: prepare-exabox-workspace.sh <stage|install>

set -euo pipefail

MODE="${1:?usage: prepare-exabox-workspace.sh <stage|install>}"
: "${STAGE_DIR:?STAGE_DIR is required}"
: "${WORKER_SRC:?WORKER_SRC is required}"

validate_target() {
    local name="$1"
    local target="$2"
    case "$target" in
        /)
            echo "$name must not be /" >&2
            return 2
            ;;
        /*) ;;
        *)
            echo "$name must be absolute: $target" >&2
            return 2
            ;;
    esac
}

validate_target STAGE_DIR "$STAGE_DIR"
validate_target WORKER_SRC "$WORKER_SRC"
[ "$STAGE_DIR" != "$WORKER_SRC" ] || {
    echo "STAGE_DIR and WORKER_SRC must differ" >&2
    exit 2
}
if [ -n "${EXABOX_REPORT_DIR:-}" ]; then
    validate_target EXABOX_REPORT_DIR "$EXABOX_REPORT_DIR"
fi
if [ -n "${CCACHE_DIR:-}" ]; then
    validate_target CCACHE_DIR "$CCACHE_DIR"
fi

case "$MODE" in
    stage)
        : "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
        [ -d "$GITHUB_WORKSPACE" ] || {
            echo "GITHUB_WORKSPACE is not a directory: $GITHUB_WORKSPACE" >&2
            exit 2
        }
        [ "$GITHUB_WORKSPACE" != "$STAGE_DIR" ] || {
            echo "GITHUB_WORKSPACE and STAGE_DIR must differ" >&2
            exit 2
        }

        rm -rf -- "$STAGE_DIR"
        mkdir -p "$(dirname "$STAGE_DIR")"
        cp -r "$GITHUB_WORKSPACE" "$STAGE_DIR"
        if [ -n "${EXABOX_REPORT_DIR:-}" ]; then
            mkdir -p "$EXABOX_REPORT_DIR"
            chmod 0777 "$EXABOX_REPORT_DIR"
        fi
        if [ -n "${CCACHE_DIR:-}" ]; then
            mkdir -p "$CCACHE_DIR"
            # The controller and worker use different UIDs on job-scoped shared
            # storage, so both must be able to update restored cache entries.
            chmod -R ugo+rwX "$CCACHE_DIR"
        fi
        # Use the same unbound worker-shell contract as hardware test phases.
        mpirun --pernode --bind-to none --tag-output \
            bash "$STAGE_DIR/.github/scripts/prepare-exabox-workspace.sh" install
        ;;
    install)
        [ -d "$STAGE_DIR" ] || {
            echo "STAGE_DIR is not a directory: $STAGE_DIR" >&2
            exit 2
        }
        rm -rf -- "$WORKER_SRC"
        mkdir -p "$(dirname "$WORKER_SRC")"
        cp -r "$STAGE_DIR" "$WORKER_SRC"
        echo "$(hostname): workspace installed at $WORKER_SRC"
        ;;
    *)
        echo "Unknown Exabox workspace operation: $MODE" >&2
        exit 2
        ;;
esac
