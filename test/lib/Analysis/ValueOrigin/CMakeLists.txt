add_llvm_executable(ttlang-value-origin-test
  ValueOriginTest.cpp
)

llvm_update_compile_flags(ttlang-value-origin-test)
set_target_properties(ttlang-value-origin-test PROPERTIES
  EXCLUDE_FROM_ALL ON
  FOLDER "Tests"
)

target_link_libraries(ttlang-value-origin-test PRIVATE
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRParser
  MLIRSCFDialect
  MLIRTensorDialect
  TTLangAnalysis
)

mlir_check_link_libraries(ttlang-value-origin-test)
