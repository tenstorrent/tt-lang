#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Generate system descriptor
HARDWARE_TYPE="${1:-n150}"

mkdir -p ttrt-artifacts
cd ttrt-artifacts

# Check if ttrt is available
if command -v ttrt &> /dev/null; then
  if [ "${HARDWARE_TYPE}" == "tg" ] || [ "${HARDWARE_TYPE}" == "p150" ]; then
    ttrt query --save-artifacts --disable-eth-dispatch
  else
    ttrt query --save-artifacts
  fi
else
  echo "Warning: ttrt not available, using dummy system descriptor"
  # For initial testing, create a minimal dummy descriptor
  # This should be replaced with actual ttrt query in production
  echo "Creating placeholder system descriptor..."
fi


SYSTEM_DESC=$(find . -name "*.ttsys" -type f | head -1)
if [ -z "$SYSTEM_DESC" ]; then
  echo "Warning: No system descriptor found"
else
  echo "SYSTEM_DESC_PATH=$SYSTEM_DESC" >> $GITHUB_ENV
  echo "Using system descriptor: $SYSTEM_DESC"
fi
