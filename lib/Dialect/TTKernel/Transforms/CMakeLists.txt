add_mlir_dialect_library(TTLangTTKernelTransforms
  TTKernelCleanupPatterns.cpp

  LINK_LIBS PUBLIC
  MLIRTTKernelDialect
)
