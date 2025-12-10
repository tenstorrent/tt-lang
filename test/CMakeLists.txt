# Test configuration

# Configure lit.site.cfg.py from template in the source directory.
# Lit will find this when searching parent directories from test/python/.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py
  @ONLY
)

add_custom_target(check-ttlang
  COMMAND ${LLVM_TOOLS_DIR}/llvm-lit -sv ${CMAKE_CURRENT_SOURCE_DIR}/python --xunit-xml-output ${CMAKE_CURRENT_BINARY_DIR}/lit-results.xml
  COMMENT "Running lit tests (xunit output: test/lit-results.xml)"
  USES_TERMINAL
)
