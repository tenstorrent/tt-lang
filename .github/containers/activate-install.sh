#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# tt-lang environment activation for installed location
# This script is used when tt-lang is installed via cmake --install

# Guard against double activation
if [ "${TTLANG_ENV_ACTIVATED:-0}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi

# Determine the install prefix (parent of env/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="$(dirname "$SCRIPT_DIR")"

# Default TTLANG_TOOLCHAIN_DIR if not set (assume same as install prefix for Docker)
: ${TTLANG_TOOLCHAIN_DIR:=$INSTALL_PREFIX}
export TTLANG_TOOLCHAIN_DIR

# Activate toolchain venv directly (do not source the venv's own activate
# script — it hardcodes the path where the venv was originally created,
# which breaks when the venv is relocated into the toolchain directory).
export VIRTUAL_ENV="${TTLANG_TOOLCHAIN_DIR}/venv"

# Set paths for installed tt-lang
export TT_LANG_HOME="$INSTALL_PREFIX"
export PATH="${INSTALL_PREFIX}/bin:${TTLANG_TOOLCHAIN_DIR}/bin:${VIRTUAL_ENV}/bin:$PATH"
export PYTHONPATH="${INSTALL_PREFIX}/python_packages:${TTLANG_TOOLCHAIN_DIR}/python_packages:${TTLANG_TOOLCHAIN_DIR}/tt-metal/python_packages/ttnn:${TTLANG_TOOLCHAIN_DIR}/tt-metal/python_packages/tools:$PYTHONPATH"
export LD_LIBRARY_PATH="${TTLANG_TOOLCHAIN_DIR}/lib:${TTLANG_TOOLCHAIN_DIR}/tt-metal/lib:$LD_LIBRARY_PATH"

# Set TT_METAL_RUNTIME_ROOT
export TT_METAL_RUNTIME_ROOT="${TTLANG_TOOLCHAIN_DIR}/tt-metal"
export TT_METAL_HOME="$TT_METAL_RUNTIME_ROOT"

export TTLANG_ENV_ACTIVATED=1

cat << 'EOF'

████████╗████████╗       ██╗      █████╗  ███╗   ██╗  ██████╗
╚══██╔══╝╚══██╔══╝       ██║     ██╔══██╗ ████╗  ██║ ██╔════╝
   ██║      ██║   █████╗ ██║     ███████║ ██╔██╗ ██║ ██║  ███╗
   ██║      ██║   ╚════╝ ██║     ██╔══██║ ██║╚██╗██║ ██║   ██║
   ██║      ██║          ███████╗██║  ██║ ██║ ╚████║ ╚██████╔╝
   ╚═╝      ╚═╝          ╚══════╝╚═╝  ╚═╝ ╚═╝  ╚═══╝  ╚═════╝
EOF
echo ""
echo "  Toolchain: ${TTLANG_TOOLCHAIN_DIR}"
echo "  Examples:  ${TTLANG_TOOLCHAIN_DIR}/examples"
echo ""
echo "  Run an example on:"
echo "   - Python simulator: ttlang-sim $TTLANG_TOOLCHAIN_DIR/examples/tutorial/multinode_grid_auto.py"
echo "   - TT hardware:      python $TTLANG_TOOLCHAIN_DIR/examples/tutorial/multinode_grid_auto.py"
