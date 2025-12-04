#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

HARDWARE_TYPE="${1:-n150}"

source build/env/activate

# Ensure PYTHONPATH includes tt-mlir installation for runtime modules and ttrt
# The ttrt package with .so files is in ${TT_MLIR_DIR} directly, not in python_packages
export PYTHONPATH="${TT_MLIR_DIR}:${TT_MLIR_DIR}/python_packages:${PYTHONPATH}"

# For hardware testing, use ttrt Python package to query the real system descriptor
echo "Setting up ttrt via Python..."
echo "TT_MLIR_DIR: ${TT_MLIR_DIR}"

# Try to use ttrt binary first, then fall back to Python module
if [ -f "/opt/ttmlir-toolchain/venv/bin/ttrt" ]; then
  TTRT_CMD="/opt/ttmlir-toolchain/venv/bin/ttrt"
  echo "Found ttrt binary at: ${TTRT_CMD}"
elif command -v ttrt >/dev/null 2>&1; then
  TTRT_CMD="ttrt"
  echo "Found ttrt in PATH: $(which ttrt)"
else
  # Use Python to invoke ttrt module directly
  TTRT_CMD="python3 -m ttrt"
  echo "Using ttrt Python module: ${TTRT_CMD}"
  echo "PYTHONPATH: ${PYTHONPATH}"
fi

mkdir -p ttrt-artifacts
cd ttrt-artifacts

echo "Using ttrt command: ${TTRT_CMD}"
if [ "${HARDWARE_TYPE}" == "tg" ] || [ "${HARDWARE_TYPE}" == "p150" ]; then
  ${TTRT_CMD} query --save-artifacts --disable-eth-dispatch
else
  ${TTRT_CMD} query --save-artifacts
fi

SYSTEM_DESC=$(find . -name "*.ttsys" -type f | head -1)
if [ -z "$SYSTEM_DESC" ]; then
  echo "Error: No system descriptor found"
  exit 1
fi

echo "SYSTEM_DESC_PATH=$SYSTEM_DESC" >> $GITHUB_ENV
echo "Using system descriptor: $SYSTEM_DESC"
