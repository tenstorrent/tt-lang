add_llvm_executable(ttlang-opt
  ttlang-opt.cpp
)

target_link_libraries(ttlang-opt PRIVATE
  TTMLIR::TTMLIRCompilerStatic
  MLIROptLib
  MLIRRegisterAllPasses
)

if(CODE_COVERAGE)
  target_link_libraries(ttlang-opt PUBLIC TTMLIR::coverage_config)
endif()
