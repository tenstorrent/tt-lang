add_llvm_executable(ttlang-translate
  ttlang-translate.cpp
)

llvm_update_compile_flags(ttlang-translate)

target_link_libraries(ttlang-translate PRIVATE
  ${TTMLIR_LINK_LIBS}
  MLIRTranslateLib
)

mlir_check_link_libraries(ttlang-translate)

install(TARGETS ttlang-translate
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT Tools
)
