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

# Default TTMLIR_TOOLCHAIN_DIR if not set (assume same as install prefix for Docker)
: ${TTMLIR_TOOLCHAIN_DIR:=$INSTALL_PREFIX}

# Activate tt-mlir toolchain venv
if [ -f "${TTMLIR_TOOLCHAIN_DIR}/venv/bin/activate" ]; then
  . "${TTMLIR_TOOLCHAIN_DIR}/venv/bin/activate"
fi

# Set paths for installed tt-lang
export TT_LANG_HOME="$INSTALL_PREFIX"
export PATH="${INSTALL_PREFIX}/bin:${TTMLIR_TOOLCHAIN_DIR}/bin:$PATH"
export PYTHONPATH="${INSTALL_PREFIX}/python_packages:${TTMLIR_TOOLCHAIN_DIR}/python_packages:${TTMLIR_TOOLCHAIN_DIR}/python_packages/ttrt/runtime/ttnn:$PYTHONPATH"
export LD_LIBRARY_PATH="${TTMLIR_TOOLCHAIN_DIR}/lib:$LD_LIBRARY_PATH"

# Set TT_METAL_RUNTIME_ROOT
export TT_METAL_RUNTIME_ROOT="${TTMLIR_TOOLCHAIN_DIR}/tt-metal"
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
echo "  Toolchain: ${TTMLIR_TOOLCHAIN_DIR}"
echo "  Examples:  ${TTMLIR_TOOLCHAIN_DIR}/examples"
echo ""
echo "  Run an example:  python \$TTMLIR_TOOLCHAIN_DIR/examples/demo_one.py"
