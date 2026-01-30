#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Example configuration overrides for tt-lang VM testing.
#
# Usage:
#   1. Copy this file to config.local.sh
#   2. Uncomment and modify the settings you want to change
#   3. config.local.sh is gitignored and will be sourced automatically
#
# You can also set these as environment variables before running scripts.

# =============================================================================
# Example: Non-standard repository layout
# =============================================================================
# If your repositories aren't in ~/tt/, specify the root:
# export TT_ROOT="/path/to/your/repos"

# Or specify each repository individually:
# export TTSIM_DIR="/custom/path/to/ttsim-private"
# export TT_MLIR_DIR="/custom/path/to/tt-mlir"
# export TT_LANG_DIR="/custom/path/to/tt-lang"

# =============================================================================
# Example: VM resource tuning
# =============================================================================
# More CPUs for faster builds (if your Mac has them):
# export VM_CPUS=12

# More memory for large builds:
# export VM_MEMORY="24GiB"

# More disk space if needed:
# export VM_DISK="200GiB"

# Use a different VM name (useful for multiple VMs):
# export VM_NAME="ttlang-dev"

# =============================================================================
# Example: Use Ubuntu 24.04 instead of 22.04
# =============================================================================
# export UBUNTU_VERSION="24.04"

# =============================================================================
# Example: Use Blackhole chip instead of Wormhole
# =============================================================================
# export CHIP_TYPE="bh"

# =============================================================================
# Example: Custom simulator directory
# =============================================================================
# export SIM_DIR="/opt/ttsim"

# =============================================================================
# Example: Debug builds
# =============================================================================
# export CMAKE_BUILD_TYPE="Debug"
