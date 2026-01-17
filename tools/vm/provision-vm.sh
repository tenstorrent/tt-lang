#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Provisioning script for tt-lang ttsim VM.
# This script runs INSIDE the VM to build ttsim, tt-mlir, and tt-lang.
#
# Prerequisites:
#   - System dependencies installed (done by Lima provisioning or manually)
#   - TT repositories mounted/accessible
#
# Environment variables (set by setup-vm.sh or manually):
#   TT_ROOT_VM      - Path to tt repositories in VM (default: ~/tt)
#   CHIP_TYPE       - Chip to simulate: wh or bh (default: wh)
#   SIM_DIR         - Directory for simulator files (default: ~/sim)
#   CMAKE_BUILD_TYPE - Build type (default: RelWithDebInfo)
#   USE_CCACHE      - Enable ccache (default: ON)

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

TT_ROOT_VM="${TT_ROOT_VM:-$HOME/tt}"
CHIP_TYPE="${CHIP_TYPE:-wh}"
SIM_DIR="${SIM_DIR:-$HOME/sim}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
USE_CCACHE="${USE_CCACHE:-ON}"

# Linux VM build artifacts directory - keeps Linux binaries separate from macOS
# This persists across VM recreations since it's in the shared mount
LINUX_VM_DIR="${LINUX_VM_DIR:-${TT_ROOT_VM}/linux-vm}"

# Toolchain location - stored under LINUX_VM_DIR for persistence across VM recreations
# This builds LLVM/MLIR from source (takes 1-2 hours on first run)
TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-${LINUX_VM_DIR}/ttmlir-toolchain}"
export TTMLIR_TOOLCHAIN_DIR

# Build directory name - use 'build-linux' to avoid conflicts with macOS builds
BUILD_DIR="${BUILD_DIR:-build-linux}"
export BUILD_DIR

# Derived paths
TT_MLIR_DIR="${TT_ROOT_VM}/tt-mlir"
TT_LANG_DIR="${TT_ROOT_VM}/tt-lang"
TT_METAL_DIR="${TT_MLIR_DIR}/third_party/tt-metal/src/tt-metal"
PATCHES_DIR="${TT_LANG_DIR}/tools/vm/patches"

# Bundled ttsim binaries (included in tt-lang repo)
TTSIM_BIN_DIR="${TT_LANG_DIR}/tools/vm/bin"

# SOC descriptor mapping (function-based for Bash 3.x compatibility)
get_soc_descriptor_name() {
    case "$1" in
        wh) echo "wormhole_b0_80_arch.yaml" ;;
        bh) echo "blackhole_140_arch.yaml" ;;
        *)  echo "" ;;
    esac
}

# Bundled libttsim.so path mapping (function-based for Bash 3.x compatibility)
get_bundled_libttsim_path() {
    case "$1" in
        wh) echo "${TTSIM_BIN_DIR}/wh/libttsim.so" ;;
        bh) echo "${TTSIM_BIN_DIR}/bh/libttsim.so" ;;
        *)  echo "" ;;
    esac
}

# Validate chip type
is_valid_chip_type() {
    case "$1" in
        wh|bh) return 0 ;;
        *)     return 1 ;;
    esac
}

# Marker files for idempotent provisioning
MARKER_DIR="$HOME/.tt-provision"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo "[$(date '+%H:%M:%S')] [INFO] $*"
}

log_warn() {
    echo "[$(date '+%H:%M:%S')] [WARN] $*" >&2
}

log_error() {
    echo "[$(date '+%H:%M:%S')] [ERROR] $*" >&2
}

log_step() {
    echo ""
    echo "=============================================="
    echo " $*"
    echo "=============================================="
}

is_step_done() {
    [[ -f "${MARKER_DIR}/$1" ]]
}

mark_step_done() {
    mkdir -p "${MARKER_DIR}"
    touch "${MARKER_DIR}/$1"
}

# =============================================================================
# Validation
# =============================================================================

validate_environment() {
    log_step "Validating environment"

    local errors=0

    if [[ ! -d "$TT_ROOT_VM" ]]; then
        log_error "TT_ROOT_VM does not exist: $TT_ROOT_VM"
        errors=$((errors + 1))
    fi

    if [[ ! -d "$TT_MLIR_DIR" ]]; then
        log_error "tt-mlir not found: $TT_MLIR_DIR"
        errors=$((errors + 1))
    fi

    if [[ ! -d "$TT_LANG_DIR" ]]; then
        log_error "tt-lang not found: $TT_LANG_DIR"
        errors=$((errors + 1))
    fi

    if ! is_valid_chip_type "$CHIP_TYPE"; then
        log_error "Invalid CHIP_TYPE: $CHIP_TYPE (must be 'wh' or 'bh')"
        errors=$((errors + 1))
    fi

    # Check for bundled ttsim binary
    local bundled_libttsim
    bundled_libttsim="$(get_bundled_libttsim_path "$CHIP_TYPE")"
    if [[ ! -f "$bundled_libttsim" ]]; then
        log_error "Bundled libttsim.so not found: $bundled_libttsim"
        errors=$((errors + 1))
    fi

    if [[ $errors -gt 0 ]]; then
        log_error "Validation failed with $errors error(s)."
        exit 1
    fi

    log_info "Environment validated successfully."
    log_info "  TT_ROOT_VM:           $TT_ROOT_VM"
    log_info "  TTMLIR_TOOLCHAIN_DIR: $TTMLIR_TOOLCHAIN_DIR"
    log_info "  CHIP_TYPE:            $CHIP_TYPE"
    log_info "  SIM_DIR:              $SIM_DIR"
    log_info "  BUILD_TYPE:           $CMAKE_BUILD_TYPE"
}

# =============================================================================
# Step 1: Install tt-metal dependencies
# =============================================================================

install_tt_metal_deps() {
    log_step "Step 2a: Installing tt-metal dependencies"

    if is_step_done "tt_metal_deps"; then
        log_info "tt-metal dependencies already installed, skipping."
        return
    fi

    # Check if we have the install script
    local install_script="${TT_MLIR_DIR}/third_party/tt-metal/src/tt-metal/install_dependencies.sh"

    if [[ ! -f "$install_script" ]]; then
        # tt-metal hasn't been fetched yet, we'll install deps after configure
        log_info "tt-metal not yet fetched, will install deps after configure."
        return
    fi

    log_info "Running tt-metal install_dependencies.sh..."
    cd "$(dirname "$install_script")"
    sudo bash install_dependencies.sh --docker --no-distributed

    mark_step_done "tt_metal_deps"
    log_info "tt-metal dependencies installed."
}

# =============================================================================
# Step 2: Build ttmlir-toolchain (LLVM/MLIR)
# =============================================================================

build_toolchain() {
    log_step "Step 1: Building ttmlir-toolchain"

    if is_step_done "toolchain_built"; then
        log_info "ttmlir-toolchain already built, skipping."
        return
    fi

    # Create toolchain directory if needed
    if [[ ! -d "$TTMLIR_TOOLCHAIN_DIR" ]]; then
        log_info "Creating $TTMLIR_TOOLCHAIN_DIR..."
        mkdir -p "$TTMLIR_TOOLCHAIN_DIR"
    fi

    # Check if toolchain already has LLVM built
    if [[ -f "$TTMLIR_TOOLCHAIN_DIR/bin/mlir-opt" ]]; then
        log_info "ttmlir-toolchain already exists at $TTMLIR_TOOLCHAIN_DIR"
        mark_step_done "toolchain_built"
        return
    fi

    cd "$TT_MLIR_DIR"

    log_info "Building ttmlir-toolchain at $TTMLIR_TOOLCHAIN_DIR..."
    log_info "This will take 1-2 hours on first run."

    # Temporarily filter out clang-tidy from build-requirements.txt
    # clang-tidy builds from source and takes 30+ minutes - not needed for runtime
    local build_reqs="env/build-requirements.txt"
    local build_reqs_backup="env/build-requirements.txt.backup"
    if grep -q "^clang-tidy" "$build_reqs" 2>/dev/null; then
        log_info "Temporarily removing clang-tidy from requirements (builds from source, not needed for runtime)..."
        cp "$build_reqs" "$build_reqs_backup"
        grep -v "^clang-tidy" "$build_reqs_backup" > "$build_reqs"
    fi

    # Source environment (disable -u for activate script)
    if [[ -f "env/activate" ]]; then
        set +u
        source env/activate
        set -u
    fi

    # Configure the toolchain build
    cmake -B env/build env

    # Build it (this compiles LLVM/MLIR)
    cmake --build env/build

    # Restore original build-requirements.txt if we backed it up
    if [[ -f "$build_reqs_backup" ]]; then
        mv "$build_reqs_backup" "$build_reqs"
    fi

    # Verify
    if [[ ! -f "$TTMLIR_TOOLCHAIN_DIR/bin/mlir-opt" ]]; then
        log_error "Toolchain build failed: mlir-opt not found"
        exit 1
    fi

    mark_step_done "toolchain_built"
    log_info "ttmlir-toolchain built successfully."
}

# =============================================================================
# Step 4: Configure tt-mlir (downloads tt-metal)
# =============================================================================

configure_tt_mlir() {
    log_step "Step 2: Configuring tt-mlir"

    if is_step_done "tt_mlir_configured"; then
        log_info "tt-mlir already configured, skipping."
        return
    fi

    cd "$TT_MLIR_DIR"

    # Source tt-mlir environment
    # Note: Temporarily disable -u since env/activate checks unset variables
    if [[ -f "env/activate" ]]; then
        log_info "Activating tt-mlir environment..."
        set +u
        source env/activate
        set -u
    fi

    # Configure with runtime enabled, disable unused features
    # Use clang-17 to match the toolchain's LLVM version (avoids LTO ABI issues)
    local cmake_args=(
        -G Ninja
        -B "${BUILD_DIR}"
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        -DCMAKE_PREFIX_PATH="${TTMLIR_TOOLCHAIN_DIR}"
        -DCMAKE_C_COMPILER=/usr/bin/clang-17
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++-17
        -DTTMLIR_ENABLE_RUNTIME=ON
        -DTTMLIR_ENABLE_RUNTIME_TESTS=OFF
        -DTTMLIR_ENABLE_STABLEHLO=OFF
        -DTTMLIR_ENABLE_OPMODEL=OFF
        -DTTMLIR_ENABLE_ALCHEMIST=OFF
        -DTTMLIR_ENABLE_EXPLORER=OFF
    )

    if [[ "$USE_CCACHE" == "ON" ]] && command -v ccache &> /dev/null; then
        cmake_args+=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
    fi

    log_info "Running cmake configure..."
    cmake "${cmake_args[@]}"

    mark_step_done "tt_mlir_configured"
    log_info "tt-mlir configured."
}

# =============================================================================
# Step 4: Install tt-metal dependencies (if needed after configure)
# =============================================================================

install_tt_metal_deps_post_configure() {
    log_step "Step 2b: Installing tt-metal dependencies (post-configure)"

    if is_step_done "tt_metal_deps"; then
        log_info "tt-metal dependencies already installed, skipping."
        return
    fi

    local install_script="${TT_METAL_DIR}/install_dependencies.sh"

    if [[ ! -f "$install_script" ]]; then
        log_warn "tt-metal install script not found: $install_script"
        log_warn "tt-metal may not have been downloaded yet. Try building tt-mlir first."
        return
    fi

    log_info "Running tt-metal install_dependencies.sh..."
    cd "$TT_METAL_DIR"
    sudo bash install_dependencies.sh --docker --no-distributed

    mark_step_done "tt_metal_deps"
    log_info "tt-metal dependencies installed."
}

# =============================================================================
# Step 5: Apply ttsim patches to tt-metal (if needed)
# =============================================================================

apply_ttsim_patches() {
    log_step "Step 3: Applying ttsim patches (if needed)"

    if is_step_done "ttsim_patches"; then
        log_info "Patches already applied, skipping."
        return
    fi

    if [[ ! -d "$TT_METAL_DIR" ]]; then
        log_warn "tt-metal directory not found, skipping patches."
        return
    fi

    # Check for patch files in tools/vm/patches/
    if [[ ! -d "$PATCHES_DIR" ]] || [[ -z "$(ls -A "$PATCHES_DIR" 2>/dev/null)" ]]; then
        log_info "No patches to apply."
        mark_step_done "ttsim_patches"
        return
    fi

    log_info "Checking for applicable patches..."
    cd "$TT_METAL_DIR"

    local applied=0
    for patch_file in "$PATCHES_DIR"/*.patch; do
        if [[ -f "$patch_file" ]]; then
            local patch_name
            patch_name=$(basename "$patch_file")

            # Check if patch can be applied (--dry-run)
            if git apply --check "$patch_file" 2>/dev/null; then
                log_info "Applying patch: $patch_name"
                git apply "$patch_file"
                applied=$((applied + 1))
            else
                log_warn "Patch already applied or not applicable: $patch_name"
            fi
        fi
    done

    log_info "Applied $applied patch(es)."
    mark_step_done "ttsim_patches"
}

# =============================================================================
# Step 6: Build tt-mlir
# =============================================================================

build_tt_mlir() {
    log_step "Step 4: Building tt-mlir"

    if is_step_done "tt_mlir_built"; then
        log_info "tt-mlir already built, skipping."
        return
    fi

    cd "$TT_MLIR_DIR"

    # Source environment (disable -u for activate script)
    if [[ -f "env/activate" ]]; then
        set +u
        source env/activate
        set -u
    fi

    log_info "Building tt-mlir (this may take 30-60 minutes)..."
    cmake --build "${BUILD_DIR}" -t all -t ttrt

    # Verify build succeeded
    if [[ ! -f "${BUILD_DIR}/bin/ttrt" ]]; then
        log_error "tt-mlir build failed: ttrt not found"
        exit 1
    fi

    mark_step_done "tt_mlir_built"
    log_info "tt-mlir built successfully."
}

# =============================================================================
# Step 6: Set up simulator directory
# =============================================================================

setup_simulator() {
    log_step "Step 5: Setting up simulator directory"

    mkdir -p "$SIM_DIR"

    # Use bundled libttsim.so from tt-lang repo
    local libttsim_src
    libttsim_src="$(get_bundled_libttsim_path "$CHIP_TYPE")"
    local libttsim_dst="${SIM_DIR}/libttsim.so"
    local soc_desc_name
    soc_desc_name="$(get_soc_descriptor_name "$CHIP_TYPE")"
    local soc_desc_src="${TT_METAL_DIR}/tt_metal/soc_descriptors/${soc_desc_name}"
    local soc_desc_dst="${SIM_DIR}/soc_descriptor.yaml"

    # Copy bundled libttsim.so
    if [[ ! -f "$libttsim_src" ]]; then
        log_error "Bundled libttsim.so not found: $libttsim_src"
        exit 1
    fi
    log_info "Copying bundled libttsim.so for $CHIP_TYPE..."
    cp "$libttsim_src" "$libttsim_dst"

    # Copy SOC descriptor
    if [[ ! -f "$soc_desc_src" ]]; then
        log_error "SOC descriptor not found: $soc_desc_src"
        exit 1
    fi
    log_info "Copying SOC descriptor: $soc_desc_name"
    cp "$soc_desc_src" "$soc_desc_dst"

    log_info "Simulator directory set up at: $SIM_DIR"
    ls -la "$SIM_DIR"
}

# =============================================================================
# Step 8: Build tt-lang
# =============================================================================

build_tt_lang() {
    log_step "Step 6: Building tt-lang"

    if is_step_done "tt_lang_built"; then
        log_info "tt-lang already built, skipping."
        return
    fi

    cd "$TT_LANG_DIR"

    # Source environment (disable -u for activate script)
    if [[ -f "env/activate" ]]; then
        set +u
        source env/activate
        set -u
    fi

    # Configure
    local cmake_args=(
        -G Ninja
        -B "${BUILD_DIR}"
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        -DTTMLIR_ENABLE_RUNTIME=ON
    )

    if [[ "$USE_CCACHE" == "ON" ]] && command -v ccache &> /dev/null; then
        cmake_args+=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
    fi

    log_info "Configuring tt-lang..."
    cmake "${cmake_args[@]}"

    log_info "Building tt-lang..."
    cmake --build "${BUILD_DIR}"

    mark_step_done "tt_lang_built"
    log_info "tt-lang built successfully."
}

# =============================================================================
# Step 9: Generate system descriptor
# =============================================================================

generate_system_desc() {
    log_step "Step 7: Generating system descriptor"

    cd "$TT_MLIR_DIR"

    # Source environment (disable -u for activate script)
    if [[ -f "env/activate" ]]; then
        set +u
        source env/activate
        set -u
    fi

    # Set simulator environment for ttrt
    export TT_METAL_SIMULATOR="${SIM_DIR}/libttsim.so"
    export TT_METAL_SLOW_DISPATCH_MODE=1

    log_info "Running ttrt query to generate system descriptor..."
    "./${BUILD_DIR}/bin/ttrt" query --save-artifacts || {
        log_warn "ttrt query failed. System descriptor may not be available."
        log_warn "Tests requiring system descriptor may fail."
    }

    if [[ -f "${BUILD_DIR}/ttrt-artifacts/system_desc.ttsys" ]]; then
        log_info "System descriptor generated: ${BUILD_DIR}/ttrt-artifacts/system_desc.ttsys"
    fi
}

# =============================================================================
# Step 10: Create activation script
# =============================================================================

create_activation_script() {
    log_step "Step 8: Creating activation script"

    local activation_script="$HOME/activate-ttsim.sh"

    cat > "$activation_script" << EOF
#!/bin/bash
# Auto-generated by provision-vm.sh
# Source this to set up ttsim test environment

# Build directory - use 'build-linux' to avoid conflicts with macOS builds
export BUILD_DIR="${BUILD_DIR}"

# Toolchain location (needed by tt-mlir/tt-lang env/activate)
export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR}"

# Simulator environment
export TT_METAL_SIMULATOR="${SIM_DIR}/libttsim.so"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TTLANG_HAS_DEVICE=1

# System descriptor (if generated)
if [[ -f "${TT_MLIR_DIR}/${BUILD_DIR}/ttrt-artifacts/system_desc.ttsys" ]]; then
    export SYSTEM_DESC_PATH="${TT_MLIR_DIR}/${BUILD_DIR}/ttrt-artifacts/system_desc.ttsys"
fi

# tt-metal paths
export TT_METAL_HOME="${TT_METAL_DIR}"
export TT_METAL_BUILD_HOME="\${TT_METAL_HOME}/build"

# Activate tt-lang environment
cd "${TT_LANG_DIR}"
if [[ -f "env/activate" ]]; then
    source env/activate
fi

echo "ttsim environment activated (chip: ${CHIP_TYPE})"
echo "  TT_METAL_SIMULATOR: \$TT_METAL_SIMULATOR"
echo "  BUILD_DIR: ${BUILD_DIR}"
echo "  TTMLIR_TOOLCHAIN_DIR: ${TTMLIR_TOOLCHAIN_DIR}"
EOF

    chmod +x "$activation_script"
    log_info "Activation script created: $activation_script"
    log_info "Source it with: source ~/activate-ttsim.sh"
}

# =============================================================================
# Main
# =============================================================================

main() {
    log_step "tt-lang ttsim VM Provisioning"
    echo "Started at: $(date)"

    validate_environment

    # Step 1: Build ttmlir-toolchain (LLVM/MLIR) - needed before configuring tt-mlir
    build_toolchain

    # Step 2: Try to install tt-metal deps if already fetched
    install_tt_metal_deps

    # Step 3: Configure tt-mlir (fetches tt-metal)
    configure_tt_mlir

    # Step 4: Install tt-metal deps after configure if needed
    install_tt_metal_deps_post_configure

    # Step 5: Apply any ttsim patches
    apply_ttsim_patches

    # Step 6: Build tt-mlir
    build_tt_mlir

    # Step 7: Set up simulator directory (uses bundled libttsim.so)
    setup_simulator

    # Step 8: Build tt-lang
    build_tt_lang

    # Step 9: Generate system descriptor
    generate_system_desc

    # Step 10: Create activation script
    create_activation_script

    log_step "Provisioning Complete"
    echo "Finished at: $(date)"
    echo ""
    echo "To activate the ttsim environment:"
    echo "  source ~/activate-ttsim.sh"
    echo ""
    echo "To run tests:"
    echo "  cd ${TT_LANG_DIR}"
    echo "  cmake --build ${BUILD_DIR} --target check-ttlang-all"
}

main "$@"
