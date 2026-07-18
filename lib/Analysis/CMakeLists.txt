add_mlir_library(TTLangAnalysis
  ExecutionCountAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSCFDialect
)
