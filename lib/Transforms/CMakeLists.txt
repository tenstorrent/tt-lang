add_mlir_dialect_library(TTLangTransforms
  SimpleAllocate.cpp

  DEPENDS
  TTLangTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRD2MDialect
  MLIRD2MAllocation
  MLIRD2MAnalysis
  MLIRD2MUtils
  MLIRTTCoreDialect
)
