add_mlir_dialect_library(TTLangTTLTransforms
  ConvertTTLToTTKernel.cpp
  ConvertTTLToCompute.cpp
  ConvertTTLComputeToSCF.cpp
  ConvertTTLTileOpsToTTKernel.cpp
  LowerDPrintToEmitC.cpp
  LowerSignpostToEmitC.cpp
  TTLAnnotateCBAssociations.cpp
  TTLDumpCBFlowGraph.cpp
  TTLInsertTileRegsSync.cpp
  TTLAssignDST.cpp
  TTLScheduleOperations.cpp
  TTLSetComputeKernelConfig.cpp
  TTLSubblockComputeForDST.cpp

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
  MLIRTilingInterface
  MLIRTTKernelDialect
  MLIRTTLDialect
  TTLangTTKernelTransforms
)
