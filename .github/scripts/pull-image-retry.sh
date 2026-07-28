#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Pull a Docker image, retrying on transient registry failures such as GHCR
# header-wait timeouts that intermittently fail an otherwise healthy pull.
# Retries with linear backoff (N * base seconds before retry N), then fails.
#
# Tuning via env: PULL_RETRY_ATTEMPTS (default 3), PULL_RETRY_BASE_DELAY
# seconds (default 10; set 0 to disable sleeps, e.g. in tests).
#
# Usage: pull-image-retry.sh <image-ref>

set -euo pipefail

# shellcheck source=lib/docker-image-utils.sh
. "$(dirname "$0")/lib/docker-image-utils.sh"

IMAGE="${1:?usage: pull-image-retry.sh <image-ref>}"
attempts="${PULL_RETRY_ATTEMPTS:-3}"
base_delay="${PULL_RETRY_BASE_DELAY:-10}"

for ((attempt = 1; attempt <= attempts; attempt++)); do
    if ttlang_docker pull "$IMAGE"; then
        exit 0
    fi
    if ((attempt < attempts)); then
        delay=$((base_delay * attempt))
        echo "docker pull failed for $IMAGE (attempt ${attempt}/${attempts}); retrying in ${delay}s" >&2
        sleep "$delay"
    fi
done

echo "::error::docker pull failed for $IMAGE after ${attempts} attempts" >&2
exit 1
