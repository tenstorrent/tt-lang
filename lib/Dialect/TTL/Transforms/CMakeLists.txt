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
  MLIRPass
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRTTKernelDialect
  MLIRTTLDialect
  TTLangTTKernelTransforms
)
