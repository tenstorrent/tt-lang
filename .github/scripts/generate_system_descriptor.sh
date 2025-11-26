#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

HARDWARE_TYPE="${1:-n150}"

source build/env/activate

# For hardware testing, try to get real system descriptor using ttrt (like tt-mlir does)
if [ -n "${TT_MLIR_DIR}" ] && [ -f "${TT_MLIR_DIR}/bin/ttrt" ]; then
  TTRT="${TT_MLIR_DIR}/bin/ttrt"
elif [ -f "/opt/ttmlir-toolchain/venv/bin/ttrt" ]; then
  TTRT="/opt/ttmlir-toolchain/venv/bin/ttrt"
elif command -v ttrt >/dev/null 2>&1; then
  TTRT="ttrt"
else
  TTRT=""
fi

mkdir -p ttrt-artifacts
cd ttrt-artifacts

if [ -n "${TTRT}" ]; then
  echo "Using ttrt from: ${TTRT}"
  if [ "${HARDWARE_TYPE}" == "tg" ] || [ "${HARDWARE_TYPE}" == "p150" ]; then
    ${TTRT} query --save-artifacts --disable-eth-dispatch
  else
    ${TTRT} query --save-artifacts
  fi
else
  echo "Error: ttrt not found. For hardware testing, ttrt must be available to query the real system descriptor."
  echo "Install ttrt or ensure it's in PATH."
  exit 1
fi

SYSTEM_DESC=$(find . -name "*.ttsys" -type f | head -1)
if [ -z "$SYSTEM_DESC" ]; then
  echo "Error: No system descriptor found"
  exit 1
fi

echo "SYSTEM_DESC_PATH=$SYSTEM_DESC" >> $GITHUB_ENV
echo "Using system descriptor: $SYSTEM_DESC"
