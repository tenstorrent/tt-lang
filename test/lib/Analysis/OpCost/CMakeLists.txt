add_llvm_executable(ttlang-opcost-test
  OpCostTest.cpp
)

llvm_update_compile_flags(ttlang-opcost-test)
set_target_properties(ttlang-opcost-test PROPERTIES
  EXCLUDE_FROM_ALL ON
  FOLDER "Tests"
)

# Links the cost library alone -- no dialect, no MLIR. If this ever needs more,
# TTLangOpCost has grown a dependency it was built to avoid.
target_link_libraries(ttlang-opcost-test PRIVATE
  LLVMSupport
  TTLangOpCost
)
