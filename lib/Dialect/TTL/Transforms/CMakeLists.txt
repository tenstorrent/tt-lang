add_mlir_dialect_library(TTLangTTLTransforms
  ConvertTTLToTTKernel.cpp
  TTLEraseDeadOps.cpp
  TTLKernelRegionsToFuncs.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTTKernelDialect
  MLIRTTLDialect
)
