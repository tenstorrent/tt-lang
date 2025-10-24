# Code coverage configuration for tt-lang

# Add coverage config target
add_library(coverage_config INTERFACE)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(coverage_config INTERFACE
    -O0
    -g
    --coverage
    -fPIC
  )
  target_link_options(coverage_config INTERFACE --coverage)
endif()

