#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Copy tt-metal runtime artifacts needed for JIT device compilation.
#
# Usage:
#   copy-ttmetal-runtime-artifacts.sh <src-dir> <dest-dir>
#   copy-ttmetal-runtime-artifacts.sh --restore <toolchain-dir> <source-dir>
#
# Default mode (save): copy from tt-metal source tree into toolchain.
# Restore mode: copy from toolchain back into source tree, skipping
#   artifacts that already exist in the source tree.
#
# Artifacts:
#   runtime/hw/                  - linker scripts and object files (build-generated)
#   runtime/sfpi/                - SFPI compiler intrinsics (JIT kernel compilation)
#   tt_metal/tt-llk/             - LLK headers (ckernel_structs.h, etc.)
#   tt_metal/soc_descriptors/    - SoC architecture descriptors (device open)
#   tt_metal/core_descriptors/   - Core architecture descriptors (device open)

set -euo pipefail

RESTORE=false
if [ "${1:-}" = "--restore" ]; then
    RESTORE=true
    shift
fi

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--restore] <src-dir> <dest-dir>"
    exit 1
fi

SRC="$1"
DEST="$2"

SOC_DESCRIPTOR_CHECKS="tt_metal/soc_descriptors/blackhole_140_arch.yaml"
SOC_DESCRIPTOR_CHECKS+="|tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"

CORE_DESCRIPTOR_CHECKS="tt_metal/core_descriptors/blackhole_140_arch.yaml"
CORE_DESCRIPTOR_CHECKS+="|tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml"
CORE_DESCRIPTOR_CHECKS+="|tt_metal/core_descriptors/blackhole_140_arch_fabric_mux.yaml"
CORE_DESCRIPTOR_CHECKS+="|tt_metal/core_descriptors/wormhole_b0_80_arch.yaml"
CORE_DESCRIPTOR_CHECKS+="|tt_metal/core_descriptors/wormhole_b0_80_arch_eth_dispatch.yaml"
CORE_DESCRIPTOR_CHECKS+="|tt_metal/core_descriptors/wormhole_b0_80_arch_fabric_mux.yaml"

# Each entry: <check-files> <artifact-dir>
# check-files is a `|`-separated list used in restore mode to skip only when
# all required files are already present.
ARTIFACTS=(
    "runtime/hw/toolchain"                            "runtime/hw"
    "runtime/sfpi/include"                            "runtime/sfpi"
    "tt_metal/tt-llk/README.md"                       "tt_metal/tt-llk"
    "$SOC_DESCRIPTOR_CHECKS"                           "tt_metal/soc_descriptors"
    "$CORE_DESCRIPTOR_CHECKS"                          "tt_metal/core_descriptors"
)

ERRORS=0

for ((i=0; i<${#ARTIFACTS[@]}; i+=2)); do
    check_files="${ARTIFACTS[i]}"
    artifact="${ARTIFACTS[i+1]}"
    parent_dir="$(dirname "$artifact")"

    if $RESTORE; then
        all_present=true
        IFS='|' read -r -a required_files <<< "$check_files"
        for check_file in "${required_files[@]}"; do
            if [ ! -e "$DEST/$check_file" ]; then
                all_present=false
                break
            fi
        done
        if $all_present; then
            continue
        fi
    fi

    if [ -d "$SRC/$artifact" ]; then
        mkdir -p "$DEST/$parent_dir"
        cp -a "$SRC/$artifact" "$DEST/$parent_dir/"
        echo "Copied $artifact"
    else
        echo "WARNING: $artifact not found at $SRC/$artifact"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ "$ERRORS" -gt 0 ]; then
    echo "WARNING: $ERRORS artifact(s) missing. Device runtime (JIT firmware builds) may fail."
fi
