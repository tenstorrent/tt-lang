add_mlir_library(TTLangOpCost
  PARTIAL_SOURCES_INTENDED
  OpCost.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang

  LINK_LIBS PUBLIC
  LLVMSupport
)

# CostTableBlackhole.inc sits beside OpCost.cpp and is included by it alone.
target_include_directories(TTLangOpCost PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# LLVM's build system expects one target per directory and checks that it lists
# every source in it. Two targets here is the intent, not an omission, so both say
# so.

add_mlir_library(TTLangAnalysis
  PARTIAL_SOURCES_INTENDED
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
