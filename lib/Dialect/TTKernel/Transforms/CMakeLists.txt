add_mlir_dialect_library(TTLangTTKernelTransforms
  TTKernelCleanupPatterns.cpp
  TTKernelConsolidateInits.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRTTKernelDialect
  MLIRTTLDialect
)
