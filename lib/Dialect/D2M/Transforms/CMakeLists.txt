add_mlir_dialect_library(TTLangD2MTransforms
  SimpleAllocate.cpp

  DEPENDS
  TTLangD2MTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRD2MDialect
  MLIRD2MAllocation
  MLIRD2MAnalysis
  MLIRD2MUtils
  MLIRTTCoreDialect
)
