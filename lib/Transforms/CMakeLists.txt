add_tt_library(TTLangTransforms
  SimpleAllocate.cpp

  DEPENDS
  TTLangTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRD2MDialect
  MLIRD2MAnalysisAllocation
  MLIRD2MUtils
  MLIRTTCoreDialect
)
