add_llvm_executable(ttlang-execution-count-test
  ExecutionCountTest.cpp
)

llvm_update_compile_flags(ttlang-execution-count-test)
set_target_properties(ttlang-execution-count-test PROPERTIES
  EXCLUDE_FROM_ALL ON
  FOLDER "Tests"
)

target_link_libraries(ttlang-execution-count-test PRIVATE
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRParser
  MLIRSCFDialect
  TTLangAnalysis
)

mlir_check_link_libraries(ttlang-execution-count-test)

add_subdirectory(ValueOrigin)

# This toy client demonstrates aggregate operation statistics without making
# the output format a supported command-line interface.
add_subdirectory(OperationStatistics)
