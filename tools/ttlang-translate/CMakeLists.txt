add_llvm_executable(ttlang-translate
  ttlang-translate.cpp
)

llvm_update_compile_flags(ttlang-translate)

if(UNIX AND NOT APPLE)
  # TTMLIR_LINK_LIBS is a large set of static archives; circular dependencies
  # between archives require a linker group for reliable resolution.
  target_link_libraries(ttlang-translate PRIVATE
    "-Wl,--start-group"
    ${TTMLIR_LINK_LIBS}
    "-Wl,--end-group"
    MLIRTranslateLib
  )
else()
  target_link_libraries(ttlang-translate PRIVATE
    ${TTMLIR_LINK_LIBS}
    MLIRTranslateLib
  )
endif()

mlir_check_link_libraries(ttlang-translate)

install(TARGETS ttlang-translate
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT Tools
)
