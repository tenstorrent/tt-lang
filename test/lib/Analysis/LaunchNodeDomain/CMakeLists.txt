add_llvm_executable(ttlang-launch-node-domain-test
  LaunchNodeDomainTest.cpp
)

llvm_update_compile_flags(ttlang-launch-node-domain-test)
set_target_properties(ttlang-launch-node-domain-test PROPERTIES
  EXCLUDE_FROM_ALL ON
  FOLDER "Tests"
)

target_link_libraries(ttlang-launch-node-domain-test PRIVATE
  MLIRParser
  MLIRRegisterAllDialects
  MLIRTTCoreDialect
  MLIRTTKernelDialect
  MLIRTTLDialect
  TTLangTTLTransforms
)

mlir_check_link_libraries(ttlang-launch-node-domain-test)
