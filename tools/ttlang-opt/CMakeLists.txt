add_llvm_executable(ttlang-opt
  ttlang-opt.cpp
)

llvm_update_compile_flags(ttlang-opt)

target_link_libraries(ttlang-opt PRIVATE
  ${TTMLIR_MINIMAL_LIBS}
  MLIROptLib
  MLIRRegisterAllPasses
  MLIRRegisterAllDialects
  TTLangTTLTransforms
  TTLangTTLPipelines
  MLIRTTLDialect
)

mlir_check_link_libraries(ttlang-opt)

install(TARGETS ttlang-opt
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT Tools
)
