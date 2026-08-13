add_llvm_executable(ttlang-opcost-report
  OpCostReportTest.cpp
)

llvm_update_compile_flags(ttlang-opcost-report)
set_target_properties(ttlang-opcost-report PROPERTIES
  EXCLUDE_FROM_ALL ON
  FOLDER "Tests"
)

# Links the cost library alone -- no dialect, no MLIR. If this ever needs more,
# TTLangOpCost has grown a dependency it was built to avoid.
target_link_libraries(ttlang-opcost-report PRIVATE
  LLVMSupport
  TTLangOpCost
)
