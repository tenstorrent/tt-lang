add_mlir_dialect_library(TTLangTTLTransforms
  ConvertTTLToTTKernel.cpp
  ConvertTTLToCompute.cpp
  ConvertTTLComputeToSCF.cpp
  ConvertTTLTileOpsToTTKernel.cpp
  TTLAnnotateCBAssociations.cpp
  TTLInsertTileRegsSync.cpp
  TTLTileAndAssignDST.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
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
