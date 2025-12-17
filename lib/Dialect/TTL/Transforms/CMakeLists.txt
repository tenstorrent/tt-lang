add_mlir_dialect_library(TTLangTTLTransforms
  AssignCBIndices.cpp
  ConvertTTLToTTKernel.cpp
  ConvertTTLToCompute.cpp
  ConvertTTLTileOpsToTTKernel.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTTKernelDialect
  MLIRTTLDialect
  TTLangTTKernelTransforms
)
