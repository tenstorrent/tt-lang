add_mlir_dialect_library(TTLangTTKernelTransforms
  TTKernelCleanupPatterns.cpp
  TTKernelInsertInits.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRTTKernelDialect
  MLIRTTLDialect
)
