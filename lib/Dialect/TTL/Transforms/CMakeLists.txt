add_mlir_dialect_library(TTLangTTLTransforms
  ConvertTTLToTTKernel.cpp
  ConvertTTLToCompute.cpp
  ConvertTTLComputeToSCF.cpp
  ConvertTTLTileOpsToTTKernel.cpp
  LowerSignpostToEmitC.cpp
  TTLAnnotateCBAssociations.cpp
  TTLDumpCBFlowGraph.cpp
  TTLInsertTileRegsSync.cpp
  TTLAssignDST.cpp
  TTLBatchDSTSync.cpp
  TTLSetComputeKernelConfig.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRPass
  MLIRTensorDialect
  MLIRTTKernelDialect
  MLIRTTLDialect
  TTLangTTKernelTransforms
)
