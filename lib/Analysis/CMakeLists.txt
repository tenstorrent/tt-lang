add_mlir_library(TTLangAnalysis
  ExecutionCountAnalysis.cpp
  IntegerExpressionEvaluator.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAnalysis
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRLoopLikeInterface
  MLIRPresburger
  MLIRSCFDialect
)
