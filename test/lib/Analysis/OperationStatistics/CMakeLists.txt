add_llvm_executable(ttlang-op-stats
  OperationStatisticsTest.cpp
)

llvm_update_compile_flags(ttlang-op-stats)
set_target_properties(ttlang-op-stats PROPERTIES
  EXCLUDE_FROM_ALL ON
  FOLDER "Tests"
)

target_link_libraries(ttlang-op-stats PRIVATE
  ${TTLANG_DIALECT_LIBS}
  MLIRFuncDialect
  MLIRParser
  MLIRRegisterAllDialects
  MLIRTTLDialect
  TTLangAnalysis
)

mlir_check_link_libraries(ttlang-op-stats)
