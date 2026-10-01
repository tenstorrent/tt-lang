add_llvm_executable(ttlang-sram-allocator-contract-test
  SRAMAllocatorContractTest.cpp
)

llvm_update_compile_flags(ttlang-sram-allocator-contract-test)
set_target_properties(ttlang-sram-allocator-contract-test PROPERTIES
  EXCLUDE_FROM_ALL ON
  FOLDER "Tests"
)
target_include_directories(ttlang-sram-allocator-contract-test PRIVATE
  ${TTLANG_SOURCE_DIR}/lib/Dialect/TTL/Transforms
)
target_link_libraries(ttlang-sram-allocator-contract-test PRIVATE
  TTLangTTLTransforms
)
mlir_check_link_libraries(ttlang-sram-allocator-contract-test)
