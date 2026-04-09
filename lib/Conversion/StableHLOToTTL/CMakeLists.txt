# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(TTLangStableHLOToTTL
  ConvertStableHLOToTTL.cpp
  ShardingUtils.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  StablehloOps
  MLIRSdyDialect
  MLIRTTLDialect
  MLIRTTCoreDialect
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
)
