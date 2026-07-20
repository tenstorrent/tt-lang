add_mlir_dialect_library(TTLangTTKernelTransforms
  TTKernelCleanupPatterns.cpp
  TTKernelCombinePackTiles.cpp
  TTKernelInsertInits.cpp
  TTKernelInsertL1Accumulation.cpp
  TTKernelLowerScalarFpTypes.cpp
  TTKernelSpecializeCores.cpp

  DEPENDS
  TTLangTTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
  MLIRTTKernelDialect
  MLIRTTLDialect
)
