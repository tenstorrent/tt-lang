#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

HARDWARE_TYPE="${1:-n150}"

source build/env/activate

# Ensure PYTHONPATH includes tt-mlir installation for runtime modules
export PYTHONPATH="${TT_MLIR_DIR}/python_packages:${PYTHONPATH}"

# For hardware testing, try to get real system descriptor using ttrt (like tt-mlir does)
echo "Looking for ttrt binary..."
echo "TT_MLIR_DIR: ${TT_MLIR_DIR}"

if [ -n "${TT_MLIR_DIR}" ] && [ -f "${TT_MLIR_DIR}/ttrt/bin/ttrt" ]; then
  TTRT="${TT_MLIR_DIR}/ttrt/bin/ttrt"
  echo "Found ttrt at: ${TTRT}"
elif [ -n "${TT_MLIR_DIR}" ] && [ -f "${TT_MLIR_DIR}/bin/ttrt" ]; then
  TTRT="${TT_MLIR_DIR}/bin/ttrt"
  echo "Found ttrt at: ${TTRT}"
elif [ -f "/opt/ttmlir-toolchain/venv/bin/ttrt" ]; then
  TTRT="/opt/ttmlir-toolchain/venv/bin/ttrt"
  echo "Found ttrt at: ${TTRT}"
elif command -v ttrt >/dev/null 2>&1; then
  TTRT="ttrt"
  echo "Found ttrt in PATH: $(which ttrt)"
else
  TTRT=""
fi

if [ -z "${TTRT}" ]; then
  echo "Error: ttrt not found. For hardware testing, ttrt must be available to query the real system descriptor."
  echo "Install ttrt or ensure it's in PATH."
  echo ""
  echo "Searched locations:"
  echo "  ${TT_MLIR_DIR}/ttrt/bin/ttrt"
  echo "  ${TT_MLIR_DIR}/bin/ttrt"
  echo "  /opt/ttmlir-toolchain/venv/bin/ttrt"
  echo "  PATH: ${PATH}"
  echo ""
  echo "Directory contents of ${TT_MLIR_DIR}/bin:"
  ls -la "${TT_MLIR_DIR}/bin" || echo "  Directory does not exist"
  echo ""
  echo "Searching for ttrt in ${TT_MLIR_DIR}:"
  find "${TT_MLIR_DIR}" -name "ttrt" -type f 2>/dev/null || echo "  No ttrt files found"
  exit 1
fi

mkdir -p ttrt-artifacts
cd ttrt-artifacts

echo "Using ttrt from: ${TTRT}"
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
