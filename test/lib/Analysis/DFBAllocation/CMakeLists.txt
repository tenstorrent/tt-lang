add_llvm_executable(ttlang-dfb-allocation-oracle-test
  DFBAllocationOracleTest.cpp
)

llvm_update_compile_flags(ttlang-dfb-allocation-oracle-test)
set_target_properties(ttlang-dfb-allocation-oracle-test PROPERTIES
  EXCLUDE_FROM_ALL ON
  FOLDER "Tests"
)

target_include_directories(ttlang-dfb-allocation-oracle-test PRIVATE
  ${PROJECT_SOURCE_DIR}/lib/Dialect/TTL/Transforms
)

target_link_libraries(ttlang-dfb-allocation-oracle-test PRIVATE
  TTLangTTLTransforms
)

mlir_check_link_libraries(ttlang-dfb-allocation-oracle-test)
