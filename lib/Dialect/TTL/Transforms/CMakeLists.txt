add_mlir_dialect_library(TTLangTTLTransforms
  ConvertTTLToTTKernel.cpp
  FuseSiblingTileLoops.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRTTKernelDialect
  MLIRTTLDialect
)
