add_llvm_executable(ttlang-opt
  ttlang-opt.cpp
)

target_link_libraries(ttlang-opt PRIVATE
  ${TTMLIR_LINK_LIBS}
  MLIROptLib
  MLIRRegisterAllPasses
)
