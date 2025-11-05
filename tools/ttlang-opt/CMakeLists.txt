add_llvm_executable(ttlang-opt ttlang-opt.cpp)

llvm_update_compile_flags(ttlang-opt)
target_link_libraries(ttlang-opt PRIVATE TTMLIRCompilerStatic MLIROptLib)
target_link_libraries(ttlang-opt PUBLIC coverage_config)

mlir_check_all_link_libraries(ttlang-opt)

install(TARGETS ttlang-opt DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)
