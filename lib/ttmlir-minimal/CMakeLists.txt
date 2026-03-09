# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Minimal tt-mlir C++ libraries needed by tt-lang.

set(TT_MLIR_SRC "${TT_MLIR_SOURCE_DIR}")

# LLVM enforces that all source files in a directory belong to exactly one
# target. Declare local source files as optional so targets that don't use
# them won't error.
set(LLVM_OPTIONAL_SOURCES
  TTKernelToCppRegistration.cpp
)

# ============================================================================
# TTCore Dialect
# ============================================================================
add_mlir_dialect_library(MLIRTTCoreDialect
  "${TT_MLIR_SRC}/lib/Dialect/TTCore/IR/TTCoreDialect.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTCore/IR/TTCoreOps.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTCore/IR/Utils.cpp"

  DEPENDS
  MLIRTTCoreOpsIncGen
  TTCoreAttrInterfacesIncGen
  MLIRTTCoreOpsEnumsIncGen
  MLIRTTCoreOpsAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRQuantDialect
)

# ============================================================================
# TTCore Transforms
# ============================================================================
add_mlir_dialect_library(MLIRTTTransforms
  "${TT_MLIR_SRC}/lib/Dialect/TTCore/Transforms/OptimizationBarrierFold.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTCore/Transforms/TTCoreModuleWrap.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTCore/Transforms/TTCoreOneShotBufferize.cpp"

  DEPENDS
  MLIRTTCoreOpsIncGen
  MLIRTTCorePassesIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationTransforms
  MLIRTTCoreDialect
)

# ============================================================================
# TTMetal Dialect
# ============================================================================
add_mlir_dialect_library(MLIRTTMetalDialect
  "${TT_MLIR_SRC}/lib/Dialect/TTMetal/IR/TTMetalDialect.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTMetal/IR/TTMetalOps.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp"

  DEPENDS
  MLIRTTMetalOpsIncGen
  TTMetalAttrInterfacesIncGen
  MLIRTTMetalOpsAttributesIncGen
  MLIRTTMetalOpsEnumsIncGen
  MLIRTTCoreOpsIncGen

  LINK_LIBS PUBLIC
  MLIRTTCoreDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRSupport
)

# ============================================================================
# TTKernel Dialect
# ============================================================================
add_mlir_dialect_library(MLIRTTKernelDialect
  "${TT_MLIR_SRC}/lib/Dialect/TTKernel/IR/TTKernelDialect.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTKernel/IR/TTKernelOps.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTKernel/IR/TTKernelOpsTypes.cpp"

  DEPENDS
  MLIRTTKernelOpsIncGen
  MLIRTTKernelOpsAttributesIncGen
  MLIRTTKernelOpsEnumsIncGen
  MLIRTTCoreOpsIncGen

  LINK_LIBS PUBLIC
  MLIRTTMetalDialect
  MLIRTTCoreDialect
  MLIRSCFDialect
)

# ============================================================================
# TTKernel Transforms
# ============================================================================
add_mlir_dialect_library(MLIRTTKernelTransforms
  "${TT_MLIR_SRC}/lib/Dialect/TTKernel/Transforms/ControlDstSection.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTKernel/Transforms/HoistInits.cpp"
  "${TT_MLIR_SRC}/lib/Dialect/TTKernel/Transforms/InsertDeviceZoneScopes.cpp"

  DEPENDS
  MLIRTTKernelOpsIncGen
  MLIRTTKernelPassesIncGen
  MLIRTTCoreOpsIncGen

  LINK_LIBS PUBLIC
  MLIRTTKernelDialect
)

# ============================================================================
# TTKernelToEmitC Conversion
# ============================================================================
add_mlir_conversion_library(TTMLIRTTKernelToEmitC
  "${TT_MLIR_SRC}/lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp"

  DEPENDS
  TTMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithToEmitC
  MLIRMemRefToEmitC
  MLIRSCFToEmitC
  MLIREmitCDialect
  MLIRTargetCpp
  MLIRTransformUtils
  MLIRTTKernelDialect
)

# ============================================================================
# TTKernelToCpp Target Translation
#
# Uses our own TTKernelToCppRegistration.cpp instead of the upstream one
# because upstream registers the TTIR dialect (which we don't build).
# ============================================================================
add_mlir_translation_library(TTKernelTargetCpp
  "${TT_MLIR_SRC}/lib/Target/TTKernel/TTKernelToCpp.cpp"
  TTKernelToCppRegistration.cpp

  DEPENDS
  TTKernelGeneratedLLKHeaders

  LINK_LIBS PUBLIC
  MLIRTTKernelDialect
  MLIRTTMetalDialect
  MLIRTTCoreDialect
  MLIRTargetCpp
)

# ============================================================================
# CAPI Layer (for Python bindings)
# ============================================================================
add_subdirectory(CAPI)

# ============================================================================
# Aggregate target list for convenience
# ============================================================================
set(TTMLIR_MINIMAL_LIBS
  MLIRTTCoreDialect
  MLIRTTTransforms
  MLIRTTMetalDialect
  MLIRTTKernelDialect
  MLIRTTKernelTransforms
  TTMLIRTTKernelToEmitC
  TTKernelTargetCpp
  PARENT_SCOPE
)
