#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Usage: scripts/shell-tt-lang-emule.sh" >&2
    exit 2
fi

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly _SCRIPT_DIR

export TTLANG_EMULE_SHELL=1
exec "${_SCRIPT_DIR}/tt-lang-emule-container.sh"
