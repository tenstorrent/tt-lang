# Test configuration

add_custom_target(check-ttlang
  COMMAND ${LLVM_TOOLS_DIR}/llvm-lit -sv ${CMAKE_CURRENT_SOURCE_DIR}/python --xunit-xml-output ${CMAKE_CURRENT_BINARY_DIR}/lit-results.xml
  COMMENT "Running lit tests (xunit output: test/lit-results.xml)"
  USES_TERMINAL
)
