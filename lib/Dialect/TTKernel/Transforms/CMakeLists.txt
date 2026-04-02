add_mlir_dialect_library(TTLangTTKernelTransforms
  TTKernelCleanupPatterns.cpp
  TTKernelCombinePackTiles.cpp
  TTKernelInsertInits.cpp
  TTKernelInsertL1Accumulation.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTTKernelDialect
  MLIRTTLDialect
)
