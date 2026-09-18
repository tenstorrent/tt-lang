#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run one hardware test phase on every Exabox worker host.
#
# Env: WORKER_SRC is required.
# Usage: run-exabox-hardware-phase.sh <phase>

set -euo pipefail

PHASE="${1:?usage: run-exabox-hardware-phase.sh <phase>}"
: "${WORKER_SRC:?WORKER_SRC is required}"

case "$WORKER_SRC" in
    /*) ;;
    *)
        echo "WORKER_SRC must be absolute: $WORKER_SRC" >&2
        exit 2
        ;;
esac

# OpenMPI otherwise binds a single per-host rank to one core, and every build
# and test process inherits that affinity restriction.
exec mpirun --pernode --bind-to none --tag-output \
    bash "$WORKER_SRC/.github/scripts/run-hardware-test-phase.sh" "$PHASE"
