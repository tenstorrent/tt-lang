add_compile_options(-Wno-deprecated-declarations -Wno-switch -Wno-covered-switch-default)
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# CAPI layer for the tt-lang-owned ttcore/ttkernel dialects (Python bindings).

add_mlir_public_c_api_library(TTMLIRMinimalCAPI
  Dialects.cpp
  TTCoreAttrs.cpp
  TTKernelTypes.cpp

  LINK_LIBS PUBLIC
  MLIRTTCoreDialect
  MLIRTTKernelDialect
  TTMLIRTTKernelToEmitC
  TTKernelTargetCpp
  MLIRIR
  MLIRCAPITransforms
  MLIRSupport
)

target_include_directories(TTMLIRMinimalCAPI PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
