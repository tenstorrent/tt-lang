#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint for tt-lang images
# Activates the toolchain and tt-lang environments and executes the provided command

# Activate tt-lang environment (includes toolchain activation)
# tt-lang env/activate is installed to the toolchain
if [ -f "${TTMLIR_TOOLCHAIN_DIR}/env/activate" ]; then
    source "${TTMLIR_TOOLCHAIN_DIR}/env/activate"
fi

# Execute the command directly (environment is already activated above)
exec "$@"
