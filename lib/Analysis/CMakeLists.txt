add_mlir_library(TTLangAnalysis
  ExecutionCountAnalysis.cpp
  IntegerExpressionEvaluator.cpp
  LoopIterationUtils.cpp
  ValueOriginAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSCFDialect
  MLIRTensorDialect
)
