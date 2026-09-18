add_mlir_dialect_library(TTLangTTKernelTransforms
  TTKernelCleanup.cpp
  TTKernelBatchStaticPipeNetReceives.cpp
  TTKernelCleanupPatterns.cpp
  TTKernelCombinePackTiles.cpp
  TTKernelInsertInits.cpp
  TTKernelInsertL1Accumulation.cpp
  TTKernelLowerScalarFpTypes.cpp
  TTKernelAnnotateDFBUse.cpp
  TTKernelSpecializeCores.cpp
  TTKernelUnrollStaticPipeNetRecordLoops.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
  MLIRTTKernelDialect
  MLIRTTLDialect
)
