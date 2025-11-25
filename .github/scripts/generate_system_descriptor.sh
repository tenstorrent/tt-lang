#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

HARDWARE_TYPE="${1:-n150}"
TTRT=/opt/ttmlir-toolchain/venv/bin/ttrt

source build/env/activate
if [ ! -f "${TTRT}" ]; then
  echo "Error: ttrt binary not found (${TTRT})"
  exit 1
fi

mkdir -p ttrt-artifacts
cd ttrt-artifacts

if [ "${HARDWARE_TYPE}" == "tg" ] || [ "${HARDWARE_TYPE}" == "p150" ]; then
  ${TTRT} query --save-artifacts --disable-eth-dispatch
else
  ${TTRT} query --save-artifacts
fi

SYSTEM_DESC=$(find . -name "*.ttsys" -type f | head -1)
if [ -z "$SYSTEM_DESC" ]; then
  echo "Error: No system descriptor found"
  exit 1
fi

echo "SYSTEM_DESC_PATH=$SYSTEM_DESC" >> $GITHUB_ENV
echo "Using system descriptor: $SYSTEM_DESC"
