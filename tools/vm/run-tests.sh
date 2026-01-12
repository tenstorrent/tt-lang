#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run tt-lang tests inside the ttsim VM.
# This script executes from macOS and SSHs into the VM to run tests.
#
# Usage:
#   ./run-tests.sh                        # Run check-ttlang-all
#   ./run-tests.sh check-ttlang-python-lit  # Run specific target
#   ./run-tests.sh test/python/simple_add.py  # Run single test
#   CHIP_TYPE=bh ./run-tests.sh           # Use Blackhole chip

set -euo pipefail

# Get script directory and source configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# =============================================================================
# Command Line Parsing
# =============================================================================

TEST_TARGET="${1:-check-ttlang-all}"
VERBOSE="${VERBOSE:-false}"

if [[ "$TEST_TARGET" == "--help" ]] || [[ "$TEST_TARGET" == "-h" ]]; then
    cat << EOF
Usage: $0 [TEST_TARGET]

Run tt-lang tests inside the ttsim VM.

Arguments:
  TEST_TARGET    CMake target or test file path (default: check-ttlang-all)

Examples:
  $0                              # Run all tests
  $0 check-ttlang-python-lit      # Run Python lit tests only
  $0 check-ttlang-mlir            # Run MLIR dialect tests
  $0 check-ttlang-pytest          # Run pytest tests
  $0 test/python/simple_add.py    # Run single test file

Environment Variables:
  CHIP_TYPE     Chip to simulate: wh (default) or bh
  VERBOSE       Set to 'true' for verbose output
  VM_NAME       VM instance name (default: ttlang-ttsim)

Configuration:
  Edit tools/vm/config.local.sh for persistent customization.
EOF
    exit 0
fi

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

check_lima_installed() {
    if ! command -v limactl &> /dev/null; then
        log_error "Lima is not installed."
        echo "Install with: brew install lima"
        exit 1
    fi
}

check_vm_running() {
    local status
    status=$(limactl list --json 2>/dev/null | \
        python3 -c "import sys,json; vms=json.load(sys.stdin); print(next((v['status'] for v in vms if v['name']=='${VM_NAME}'), 'NotFound'))" 2>/dev/null || echo "NotFound")

    if [[ "$status" != "Running" ]]; then
        log_error "VM '${VM_NAME}' is not running (status: $status)"
        echo ""
        echo "Start the VM with:"
        echo "  limactl start ${VM_NAME}"
        echo ""
        echo "Or create it with:"
        echo "  ./tools/vm/setup-vm.sh"
        exit 1
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    check_lima_installed
    check_vm_running

    local vm_mount_point
    vm_mount_point="$(get_vm_mount_point)"
    local tt_lang_vm="${vm_mount_point}/tt-lang"
    local tt_mlir_vm="${vm_mount_point}/tt-mlir"

    log_info "Running tests in VM '${VM_NAME}'..."
    log_info "  Test target: ${TEST_TARGET}"
    log_info "  Chip type: ${CHIP_TYPE}"

    # Build the test command
    local test_cmd

    if [[ "$TEST_TARGET" == *.py ]]; then
        # Single test file - use llvm-lit directly
        test_cmd="llvm-lit -sv '${tt_lang_vm}/${TEST_TARGET}'"
    elif [[ "$TEST_TARGET" == check-* ]]; then
        # CMake target
        test_cmd="cmake --build \${BUILD_DIR} --target ${TEST_TARGET}"
    else
        # Assume it's a directory or pattern for llvm-lit
        test_cmd="llvm-lit -sv '${tt_lang_vm}/${TEST_TARGET}'"
    fi

    # SIM_DIR inside VM (not the macOS path)
    local vm_sim_dir="\$HOME/sim"

    # Execute in VM
    # Filter out Lima's cd warnings from stderr
    limactl shell "${VM_NAME}" -- bash -c "
        # Linux VM artifacts directory
        export LINUX_VM_DIR='${vm_mount_point}/linux-vm'

        # Toolchain location (needed by env/activate)
        export TTMLIR_TOOLCHAIN_DIR='\${LINUX_VM_DIR}/ttmlir-toolchain'

        # Use build-linux directories to avoid conflicts with macOS builds
        export BUILD_DIR='build-linux'

        # Set up simulator environment
        export TT_METAL_SIMULATOR='${vm_sim_dir}/libttsim.so'
        export TT_METAL_SLOW_DISPATCH_MODE=1
        export TTLANG_HAS_DEVICE=1

        # Set system descriptor if available
        if [[ -f '${tt_mlir_vm}/build-linux/ttrt-artifacts/system_desc.ttsys' ]]; then
            export SYSTEM_DESC_PATH='${tt_mlir_vm}/build-linux/ttrt-artifacts/system_desc.ttsys'
        fi

        # tt-metal paths
        export TT_METAL_HOME='${tt_mlir_vm}/third_party/tt-metal/src/tt-metal'
        export TT_METAL_BUILD_HOME=\"\${TT_METAL_HOME}/build\"

        # Activate tt-lang environment
        cd '${tt_lang_vm}'
        if [[ -f 'env/activate' ]]; then
            source env/activate
        fi

        # Run the test command
        echo '=== Running: ${test_cmd} ==='
        ${test_cmd}
    " 2>&1 | grep -v "^bash: line 1: cd:"

    local exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -eq 0 ]]; then
        log_info "Tests passed!"
    else
        log_error "Tests failed with exit code: $exit_code"
    fi

    exit $exit_code
}

main
