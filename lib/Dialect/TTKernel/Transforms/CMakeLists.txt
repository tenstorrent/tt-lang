add_mlir_dialect_library(TTLangTTKernelTransforms
  TTKernelCleanupPatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang/Dialect/TTKernel

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTTKernelDialect
)
