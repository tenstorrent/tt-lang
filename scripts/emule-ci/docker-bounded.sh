#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
exec "${TTLANG_CI_HOST_PYTHON:?}" \
    "$(dirname "${BASH_SOURCE[0]}")/worker.py" --docker-proxy "$@"
