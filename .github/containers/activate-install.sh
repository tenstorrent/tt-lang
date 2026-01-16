#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# tt-lang environment activation for installed location
# This script is used when tt-lang is installed via cmake --install

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

echo "tt-lang environment activated"
echo "  TT-Lang install:       ${INSTALL_PREFIX}"
echo "  Python:                ${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python3"
echo "  TT-MLIR:               ${TTMLIR_TOOLCHAIN_DIR}"
echo "  Toolchain:             ${TTMLIR_TOOLCHAIN_DIR}"
echo "  TT_METAL_RUNTIME_ROOT: ${TT_METAL_RUNTIME_ROOT}"
echo "  LD_LIBRARY_PATH:       ${LD_LIBRARY_PATH}"
