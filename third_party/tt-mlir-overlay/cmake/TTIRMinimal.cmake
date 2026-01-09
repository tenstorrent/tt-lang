# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Minimal TTIR dialect that excludes TTIROps.cpp (which has TTNN dependencies)
# This is sufficient for TTKernelTargetCpp which only needs dialect registration
#
# We name this MLIRTTIRDialect (not MLIRTTIRDialectMinimal) because upstream
# TTKernelTargetCpp expects MLIRTTIRDialect as a dependency.

add_mlir_dialect_library(MLIRTTIRDialect
    ${ttmlir_SOURCE_DIR}/lib/Dialect/TTIR/IR/TTIRDialect.cpp
    ${ttmlir_SOURCE_DIR}/lib/Dialect/TTIR/IR/TTIROpsTypes.cpp
    ${ttmlir_SOURCE_DIR}/lib/Dialect/TTIR/IR/TTIRTraits.cpp

    ADDITIONAL_HEADER_DIRS
    ${ttmlir_SOURCE_DIR}/include/ttmlir

    DEPENDS
    MLIRTTIROpsIncGen
    TTIROpsInterfacesIncGen
    MLIRTTIRPassesIncGen
    MLIRTTCoreOpsIncGen

    LINK_LIBS PUBLIC
    MLIRTTCoreDialect
    MLIRBufferizationTransforms
    TTMLIRTTIRUtils
)
