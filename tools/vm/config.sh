#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Central configuration for tt-lang VM testing infrastructure.
# All scripts source this file to get consistent settings.
#
# To customize settings without modifying this file:
#   1. Copy config.example.sh to config.local.sh
#   2. Edit config.local.sh with your settings
#   3. config.local.sh is gitignored and will be sourced automatically
#
# Alternatively, set environment variables before running scripts.

set -euo pipefail

# =============================================================================
# Path Configuration
# =============================================================================

# Root directory containing tt-* repositories (tt-lang, tt-mlir)
# Default assumes ~/tt/ layout: ~/tt/tt-lang, ~/tt/tt-mlir
TT_ROOT="${TT_ROOT:-$HOME/tt}"

# Individual repository locations (relative to TT_ROOT by default)
TT_MLIR_DIR="${TT_MLIR_DIR:-$TT_ROOT/tt-mlir}"
TT_LANG_DIR="${TT_LANG_DIR:-$TT_ROOT/tt-lang}"

# =============================================================================
# VM Configuration
# =============================================================================

# VM instance name (used by Lima and for identification)
VM_NAME="${VM_NAME:-ttlang-ttsim}"

# VM resource allocation
VM_CPUS="${VM_CPUS:-8}"           # Number of CPUs (8+ recommended for parallel builds)
VM_MEMORY="${VM_MEMORY:-16GiB}"   # RAM allocation (16GB minimum, 24GB+ recommended)
VM_DISK="${VM_DISK:-128GiB}"      # Disk size (tt-mlir + tt-metal need ~80GB)

# Ubuntu version for the VM guest
# Supported: 22.04 (recommended, tested with tt-metal), 24.04
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"

# =============================================================================
# Simulator Configuration
# =============================================================================

# Chip type to simulate
# Supported: wh (Wormhole), bh (Blackhole)
CHIP_TYPE="${CHIP_TYPE:-wh}"

# Directory for simulator files (libttsim.so and soc_descriptor.yaml)
# This is where libttsim.so will be copied along with matching SOC descriptor
SIM_DIR="${SIM_DIR:-$HOME/sim}"

# Linux VM build artifacts directory - keeps Linux binaries separate from macOS
# This persists across VM recreations since it's in the shared mount
LINUX_VM_DIR="${LINUX_VM_DIR:-$TT_ROOT/linux-vm}"

# Toolchain directory - stores LLVM/MLIR built from source (Linux ARM64 binaries)
# Located under LINUX_VM_DIR to avoid confusion with macOS toolchain
TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-$LINUX_VM_DIR/ttmlir-toolchain}"

# =============================================================================
# Mount Configuration
# =============================================================================

# Where TT_ROOT is mounted inside the VM
# Lima uses <macOS-username>.linux as the default username
# This allows editing files on macOS while building/testing in Linux VM
VM_MOUNT_POINT="${VM_MOUNT_POINT:-/home/${USER}.linux/tt}"

# =============================================================================
# Build Configuration
# =============================================================================

# CMake build type for tt-mlir and tt-lang
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"

# Enable ccache if available (speeds up rebuilds significantly)
USE_CCACHE="${USE_CCACHE:-ON}"

# =============================================================================
# Derived Paths (computed from above, generally don't need to change)
# =============================================================================

# Script directory (where this config.sh lives)
VM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bundled ttsim binaries directory (included in tt-lang repo)
TTSIM_BIN_DIR="${VM_SCRIPTS_DIR}/bin"

# tt-metal source location (fetched by tt-mlir ExternalProject)
TT_METAL_DIR="${TT_MLIR_DIR}/third_party/tt-metal/src/tt-metal"

# SOC descriptor mapping (function-based for Bash 3.x compatibility)
_get_soc_descriptor_name() {
    case "$1" in
        wh) echo "wormhole_b0_80_arch.yaml" ;;
        bh) echo "blackhole_140_arch.yaml" ;;
        *)  echo "" ;;
    esac
}

# Bundled libttsim.so path mapping (function-based for Bash 3.x compatibility)
_get_bundled_libttsim_path() {
    case "$1" in
        wh) echo "${TTSIM_BIN_DIR}/wh/libttsim.so" ;;
        bh) echo "${TTSIM_BIN_DIR}/bh/libttsim.so" ;;
        *)  echo "" ;;
    esac
}

# Validate chip type
_is_valid_chip_type() {
    case "$1" in
        wh|bh) return 0 ;;
        *)     return 1 ;;
    esac
}

# =============================================================================
# Helper Functions
# =============================================================================

# Print configuration summary
config_summary() {
    echo "=== tt-lang VM Configuration ==="
    echo "TT_ROOT:              $TT_ROOT"
    echo "TT_MLIR_DIR:          $TT_MLIR_DIR"
    echo "TT_LANG_DIR:          $TT_LANG_DIR"
    echo "TTMLIR_TOOLCHAIN_DIR: $TTMLIR_TOOLCHAIN_DIR"
    echo "TTSIM_BIN_DIR:        $TTSIM_BIN_DIR"
    echo "VM_NAME:              $VM_NAME"
    echo "VM_CPUS:              $VM_CPUS"
    echo "VM_MEMORY:            $VM_MEMORY"
    echo "VM_DISK:              $VM_DISK"
    echo "UBUNTU_VERSION:       $UBUNTU_VERSION"
    echo "CHIP_TYPE:            $CHIP_TYPE"
    echo "SIM_DIR:              $SIM_DIR"
    echo "VM_MOUNT_POINT:       $VM_MOUNT_POINT"
    echo "================================"
}

# Validate that required paths exist
validate_paths() {
    local errors=0

    if [[ ! -d "$TT_ROOT" ]]; then
        echo "ERROR: TT_ROOT does not exist: $TT_ROOT" >&2
        errors=$((errors + 1))
    fi

    if [[ ! -d "$TT_MLIR_DIR" ]]; then
        echo "ERROR: TT_MLIR_DIR does not exist: $TT_MLIR_DIR" >&2
        errors=$((errors + 1))
    fi

    if [[ ! -d "$TT_LANG_DIR" ]]; then
        echo "ERROR: TT_LANG_DIR does not exist: $TT_LANG_DIR" >&2
        errors=$((errors + 1))
    fi

    if ! _is_valid_chip_type "$CHIP_TYPE"; then
        echo "ERROR: Invalid CHIP_TYPE: $CHIP_TYPE (must be 'wh' or 'bh')" >&2
        errors=$((errors + 1))
    fi

    # Check for bundled ttsim binary
    local bundled_libttsim
    bundled_libttsim="$(_get_bundled_libttsim_path "$CHIP_TYPE")"
    if [[ ! -f "$bundled_libttsim" ]]; then
        echo "ERROR: Bundled libttsim.so not found: $bundled_libttsim" >&2
        errors=$((errors + 1))
    fi

    return $errors
}

# Get the SOC descriptor filename for current chip type
get_soc_descriptor() {
    _get_soc_descriptor_name "$CHIP_TYPE"
}

# Get the bundled libttsim.so path for current chip type
get_bundled_libttsim_path() {
    _get_bundled_libttsim_path "$CHIP_TYPE"
}

# Get the VM mount point (already expanded with $USER at source time)
get_vm_mount_point() {
    echo "$VM_MOUNT_POINT"
}

# =============================================================================
# Load Local Overrides
# =============================================================================

# Source local configuration if it exists (gitignored for personal settings)
if [[ -f "${VM_SCRIPTS_DIR}/config.local.sh" ]]; then
    # shellcheck source=/dev/null
    source "${VM_SCRIPTS_DIR}/config.local.sh"
fi
