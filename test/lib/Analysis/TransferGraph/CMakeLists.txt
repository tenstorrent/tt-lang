add_llvm_executable(ttlang-transfer-graph-test TransferGraphTest.cpp)
llvm_update_compile_flags(ttlang-transfer-graph-test)
set_target_properties(ttlang-transfer-graph-test PROPERTIES
  EXCLUDE_FROM_ALL ON
  FOLDER "Tests"
)
target_link_libraries(ttlang-transfer-graph-test PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRParser
  MLIRTTLDialect
  MLIRTTKernelDialect
  MLIRTransformUtils
)
mlir_check_link_libraries(ttlang-transfer-graph-test)
