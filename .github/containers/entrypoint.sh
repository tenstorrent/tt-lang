#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint for tt-lang images
# Activates the toolchain environment and executes the provided command

# Activate tt-mlir toolchain venv if available
if [ -f "${TTMLIR_TOOLCHAIN_DIR}/venv/bin/activate" ]; then
    source "${TTMLIR_TOOLCHAIN_DIR}/venv/bin/activate"
fi

# Setup tt-lang paths if installed
if [ -d "${TTLANG_INSTALL_DIR}" ]; then
    export PATH="${TTLANG_INSTALL_DIR}/bin:$PATH"
    export PYTHONPATH="${TTLANG_INSTALL_DIR}/python_packages:${TTLANG_INSTALL_DIR}/python:$PYTHONPATH"
fi

# If command is bash or sh, run it; otherwise wrap in bash to preserve environment
if [[ "$1" == "bash" ]] || [[ "$1" == "/bin/bash" ]] || [[ "$1" == "sh" ]] || [[ "$1" == "/bin/sh" ]]; then
    exec "$@"
else
    exec bash -c "$*"
fi
