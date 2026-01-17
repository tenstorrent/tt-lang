#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for tt-lang ttsim VM testing environment.
# Creates and configures a Lima VM for running hardware tests with ttsim simulator.
#
# Usage:
#   ./setup-vm.sh              # Create/start VM and run provisioning
#   ./setup-vm.sh --dry-run    # Show what would be done without executing
#   ./setup-vm.sh --skip-provision  # Create VM without running provisioner
#   ./setup-vm.sh --utm        # Show UTM setup instructions instead of using Lima
#   ./setup-vm.sh --delete     # Delete the VM
#   ./setup-vm.sh --status     # Show VM status

set -euo pipefail

# Get script directory and source configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# =============================================================================
# Command Line Parsing
# =============================================================================

DRY_RUN=false
SKIP_PROVISION=false
USE_UTM=false
DELETE_VM=false
SHOW_STATUS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-provision)
            SKIP_PROVISION=true
            shift
            ;;
        --utm)
            USE_UTM=true
            shift
            ;;
        --delete)
            DELETE_VM=true
            shift
            ;;
        --status)
            SHOW_STATUS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Show what would be done without executing"
            echo "  --skip-provision  Create VM without running provisioner"
            echo "  --utm             Show UTM setup instructions (for manual setup)"
            echo "  --delete          Delete the VM"
            echo "  --status          Show VM status"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Configuration is read from config.sh (edit config.local.sh for customization)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

run_cmd() {
    if $DRY_RUN; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

check_lima_installed() {
    if ! command -v limactl &> /dev/null; then
        log_error "Lima is not installed."
        echo ""
        echo "Install Lima using Homebrew:"
        echo "  brew install lima"
        echo ""
        echo "Or use --utm flag for UTM-based setup."
        exit 1
    fi
}

# =============================================================================
# Lima Configuration Generation
# =============================================================================

generate_lima_yaml() {
    local template="${SCRIPT_DIR}/lima.yaml.template"
    local output="${SCRIPT_DIR}/lima.yaml"
    local vm_mount_point
    vm_mount_point="$(get_vm_mount_point)"

    log_info "Generating Lima configuration..."

    if [[ ! -f "$template" ]]; then
        log_error "Template not found: $template"
        exit 1
    fi

    # Substitute placeholders
    sed -e "s|@TT_ROOT@|${TT_ROOT}|g" \
        -e "s|@VM_MOUNT_POINT@|${vm_mount_point}|g" \
        -e "s|@VM_CPUS@|${VM_CPUS}|g" \
        -e "s|@VM_MEMORY@|${VM_MEMORY}|g" \
        -e "s|@VM_DISK@|${VM_DISK}|g" \
        -e "s|@UBUNTU_VERSION@|${UBUNTU_VERSION}|g" \
        -e "s|@CHIP_TYPE@|${CHIP_TYPE}|g" \
        "$template" > "$output"

    log_info "Generated: $output"
}

# =============================================================================
# VM Management
# =============================================================================

vm_exists() {
    limactl list --quiet 2>/dev/null | grep -q "^${VM_NAME}$"
}

vm_running() {
    local status
    status=$(limactl list --json 2>/dev/null | \
        python3 -c "import sys,json; vms=json.load(sys.stdin); print(next((v['status'] for v in vms if v['name']=='${VM_NAME}'), 'NotFound'))" 2>/dev/null || echo "NotFound")
    [[ "$status" == "Running" ]]
}

show_vm_status() {
    if ! vm_exists; then
        echo "VM '${VM_NAME}' does not exist."
        echo ""
        echo "Run '$0' to create it."
        return
    fi

    echo "=== VM Status: ${VM_NAME} ==="
    limactl list | grep -E "^NAME|^${VM_NAME}"
    echo ""

    if vm_running; then
        echo "SSH command: limactl shell ${VM_NAME}"
        echo ""
        echo "VM Mount point: $(get_vm_mount_point)"
    fi
}

delete_vm() {
    if ! vm_exists; then
        log_info "VM '${VM_NAME}' does not exist."
        return
    fi

    log_warn "This will delete VM '${VM_NAME}' and all its data."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_cmd limactl delete --force "${VM_NAME}"
        log_info "VM deleted."
    else
        log_info "Cancelled."
    fi
}

create_vm() {
    if vm_exists; then
        log_info "VM '${VM_NAME}' already exists."
        if ! vm_running; then
            log_info "Starting VM..."
            run_cmd limactl start "${VM_NAME}"
        fi
        return
    fi

    log_info "Creating VM '${VM_NAME}'..."
    run_cmd limactl create --name="${VM_NAME}" "${SCRIPT_DIR}/lima.yaml"

    log_info "Starting VM..."
    run_cmd limactl start "${VM_NAME}"
}

wait_for_vm() {
    # limactl start blocks until VM is ready, so we just need to verify SSH works
    log_info "Verifying VM is accessible..."
    local max_attempts=30
    local attempt=0

    # Wait for SSH to be available
    while ! limactl shell "${VM_NAME}" -- true 2>/dev/null && [[ $attempt -lt $max_attempts ]]; do
        sleep 2
        attempt=$((attempt + 1))
    done

    if ! limactl shell "${VM_NAME}" -- true 2>/dev/null; then
        log_error "SSH not available within timeout."
        log_error "Check VM status with: limactl list"
        exit 1
    fi

    log_info "VM is ready."
}

run_provisioning() {
    log_info "Running provisioning script inside VM..."

    local vm_mount_point
    vm_mount_point="$(get_vm_mount_point)"

    # SIM_DIR should be a VM-local path, not the macOS path
    local vm_sim_dir="\$HOME/sim"

    # Linux VM artifacts directory - keeps Linux binaries separate from macOS
    local vm_linux_dir="${vm_mount_point}/linux-vm"
    local vm_toolchain_dir="${vm_linux_dir}/ttmlir-toolchain"

    # Execute provisioning script inside VM
    # Note: 2>&1 | grep -v "^bash: line 1: cd:" filters Lima's cd warnings
    run_cmd limactl shell "${VM_NAME}" -- bash -c "
        export TT_ROOT_VM='${vm_mount_point}'
        export LINUX_VM_DIR='${vm_linux_dir}'
        export TTMLIR_TOOLCHAIN_DIR='${vm_toolchain_dir}'
        export CHIP_TYPE='${CHIP_TYPE}'
        export SIM_DIR='${vm_sim_dir}'
        export CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'
        export USE_CCACHE='${USE_CCACHE}'
        bash '${vm_mount_point}/tt-lang/tools/vm/provision-vm.sh'
    " 2>&1 | grep -v "^bash: line 1: cd:"
}

# =============================================================================
# UTM Instructions
# =============================================================================

show_utm_instructions() {
    cat << 'EOF'
=== UTM Setup Instructions ===

UTM doesn't support CLI automation, so follow these manual steps:

1. Download and install UTM from: https://mac.getutm.app/

2. Download Ubuntu 22.04 ARM64 image:
   https://cdimage.ubuntu.com/releases/22.04/release/ubuntu-22.04.5-live-server-arm64.iso

3. Create a new VM in UTM:
   - Click '+' to create new VM
   - Choose 'Virtualize' (not Emulate)
   - Select 'Linux'
   - Check 'Use Apple Virtualization'
   - Select the Ubuntu ISO you downloaded
   - Optionally enable 'Rosetta'

4. Configure resources:
EOF
    echo "   - CPUs: ${VM_CPUS}"
    echo "   - Memory: ${VM_MEMORY}"
    echo "   - Disk: ${VM_DISK}"
    cat << 'EOF'

5. Set up shared directory:
   - In VM settings, go to 'Sharing'
   - Add a directory share pointing to your TT_ROOT
EOF
    echo "   - Host path: ${TT_ROOT}"
    echo "   - Guest mount: $(get_vm_mount_point)"
    cat << 'EOF'

6. Install Ubuntu:
   - Boot the VM and complete Ubuntu installation
   - Enable OpenSSH server during installation
   - Note the VM's IP address after install (run 'ip a')

7. After installation, SSH into the VM and run:
EOF
    echo "   ssh <vm-user>@<vm-ip>"
    echo "   cd $(get_vm_mount_point)/tt-lang/tools/vm"
    echo "   ./provision-vm.sh"
    cat << 'EOF'

For detailed instructions, see: tools/vm/utm-setup.md
EOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Show configuration
    config_summary

    # Validate paths
    if ! validate_paths; then
        log_error "Path validation failed. Check your configuration."
        exit 1
    fi

    # Handle special modes
    if $SHOW_STATUS; then
        check_lima_installed
        show_vm_status
        exit 0
    fi

    if $DELETE_VM; then
        check_lima_installed
        delete_vm
        exit 0
    fi

    if $USE_UTM; then
        show_utm_instructions
        exit 0
    fi

    # Main setup flow with Lima
    check_lima_installed

    # Generate Lima configuration
    generate_lima_yaml

    # Create/start VM
    create_vm

    # Wait for VM to be ready
    if ! $DRY_RUN; then
        wait_for_vm
    fi

    # Run provisioning unless skipped
    if ! $SKIP_PROVISION; then
        run_provisioning
    else
        log_info "Skipping provisioning (use --skip-provision was specified)"
        log_info "Run provisioning manually with: limactl shell ${VM_NAME} -- bash ~/tt/tt-lang/tools/vm/provision-vm.sh"
    fi

    log_info "Setup complete!"
    echo ""
    echo "To access the VM:"
    echo "  limactl shell ${VM_NAME}"
    echo ""
    echo "To run tests:"
    echo "  ./tools/vm/run-tests.sh check-ttlang-all"
}

main
