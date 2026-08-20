add_mlir_dialect_library(TTLangTTKernelTransforms
  TTKernelCleanupPatterns.cpp
  TTKernelCombinePackTiles.cpp
  TTKernelInsertInits.cpp
  TTKernelInsertL1Accumulation.cpp
  TTKernelLowerScalarFpTypes.cpp
  TTKernelAnnotateDFBUse.cpp
  TTKernelSpecializeCores.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
  MLIRTTKernelDialect
  MLIRTTLDialect
)
