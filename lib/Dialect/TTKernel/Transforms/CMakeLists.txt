add_mlir_dialect_library(TTLangTTKernelTransforms
  FuseSiblingTileLoops.cpp
  TTKernelCleanupPatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang/Dialect/TTKernel

  DEPENDS
  TTLangTTKernelTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRTTKernelDialect
)
