# Depends on LLVM Support alone -- deliberately no MLIR and no dialect, so that
# anything can query operation costs and no edge back from a dialect to here can
# ever be created. Recovering a KernelConfig from IR is the caller's job.
add_mlir_library(TTLangOpCost
  OpCost.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttlang

  LINK_LIBS PUBLIC
  LLVMSupport
)

target_include_directories(TTLangOpCost PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
