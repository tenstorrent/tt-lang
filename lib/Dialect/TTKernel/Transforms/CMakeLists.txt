add_mlir_dialect_library(TTLangTTKernelTransforms
  TTKernelCleanupPatterns.cpp
  TTKernelCombinePackTiles.cpp
  TTKernelInsertInits.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRTTKernelDialect
  MLIRTTLDialect
)
