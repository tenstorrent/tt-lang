# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# CAPI layer for tt-mlir dialects used by tt-lang Python bindings.
#
# Compiles both our own Dialects.cpp (with minimal registration) and
# the tt-mlir CAPI sources for TTCore attrs and TTKernel types.

add_mlir_public_c_api_library(TTMLIRMinimalCAPI
  # Our registration wrapper
  Dialects.cpp

  # tt-mlir CAPI sources
  "${TT_MLIR_SOURCE_DIR}/lib/CAPI/TTCoreAttrs.cpp"
  "${TT_MLIR_SOURCE_DIR}/lib/CAPI/TTKernelTypes.cpp"

  LINK_LIBS PUBLIC
  MLIRTTCoreDialect
  MLIRTTTransforms
  MLIRTTKernelDialect
  MLIRTTKernelTransforms
  MLIRTTMetalDialect
  TTMLIRTTKernelToEmitC
  TTKernelTargetCpp
  MLIRIR
  MLIRCAPITransforms
  MLIRSupport
)

target_include_directories(TTMLIRMinimalCAPI PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
