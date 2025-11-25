# Test configuration

# Lit test target - produces JSON output
add_custom_target(check-ttlang
  COMMAND ${LLVM_TOOLS_DIR}/llvm-lit -sv ${CMAKE_CURRENT_SOURCE_DIR}/python --output ${CMAKE_CURRENT_BINARY_DIR}/lit-results.json
  COMMENT "Running lit tests (JSON output: test/lit-results.json)"
  USES_TERMINAL
)
