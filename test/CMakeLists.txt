# Lit Test configuration

# Check for Tenstorrent device at configure time (must be before configure_lit_site_cfg)
ttlang_check_device_available(TTLANG_HAS_DEVICE)

if(TTLANG_HAS_DEVICE)
  set(TTLANG_HAS_DEVICE_INT 1)
else()
  set(TTLANG_HAS_DEVICE_INT 0)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  CMAKE_BINARY_DIR
  CMAKE_SOURCE_DIR
  TTLANG_PYTHON_PACKAGES_DIR
  CMAKE_CURRENT_SOURCE_DIR
)



set(TTLANG_TEST_DEPENDS
  ttlang-opt
  ttlang-translate
)

# Common llvm-lit options for consistent output across tt-lang test suites.
set(TTLANG_LIT_COMMON_ARGS
  --verbose
)

if(DEFINED ENV{TTLANG_CMAKE_DEBUG})
  list(APPEND TTLANG_LIT_COMMON_ARGS --show-all --no-progress-bar)
endif()

# Python binding tests require the ttlang Python extension module. Ensure it is
# built before running the full suite (and python-only suites), so imports of
# `ttlang._mlir_libs._ttlang` do not spuriously fail.
set(TTLANG_PYTHON_TEST_DEPENDS ${TTLANG_TEST_DEPENDS})

if(TARGET TTLangPythonModules)
  list(APPEND TTLANG_PYTHON_TEST_DEPENDS TTLangPythonModules)
endif()

# MLIR tests only (no TT device required).
add_lit_testsuite(check-ttlang-mlir "Running ttlang MLIR tests"
  ${CMAKE_CURRENT_BINARY_DIR}/ttlang
  ${TTLANG_LIT_COMMON_ARGS} --xunit-xml-output ttlang-mlir-report.xml
  DEPENDS ${TTLANG_TEST_DEPENDS}
)
set_target_properties(check-ttlang-mlir PROPERTIES FOLDER "Tests")

# Python lit tests. Run sequentially to avoid TLB resource contention
# when multiple tests open Tenstorrent devices.
add_lit_testsuite(check-ttlang-python-lit "Running ttlang Python lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}/python
  -j1 ${TTLANG_LIT_COMMON_ARGS} --xunit-xml-output python-lit-report.xml
  DEPENDS ${TTLANG_PYTHON_TEST_DEPENDS}
)
set_target_properties(check-ttlang-python-lit PROPERTIES FOLDER "Tests")

# Python bindings tests only (no TT device required).
add_lit_testsuite(check-ttlang-python-bindings "Running ttlang Python binding tests"
  ${CMAKE_CURRENT_BINARY_DIR}/bindings/python
  ${TTLANG_LIT_COMMON_ARGS} --xunit-xml-output python-bindings-report.xml
  DEPENDS ${TTLANG_PYTHON_TEST_DEPENDS}
)
set_target_properties(check-ttlang-python-bindings PROPERTIES FOLDER "Tests")

# Lit tests that never require a TT device.
add_custom_target(check-ttlang
  DEPENDS check-ttlang-mlir check-ttlang-python-bindings
  COMMENT "Running ttlang lit tests that never require a TT device"
)

# Pytest tests (test/python/test_*.py files).
add_custom_target(check-ttlang-pytest
  COMMAND ${Python3_EXECUTABLE} -m pytest
    ${CMAKE_CURRENT_SOURCE_DIR}/python
    -v --tb=short
    --junitxml=${CMAKE_CURRENT_BINARY_DIR}/pytest-report.xml
  DEPENDS ${TTLANG_PYTHON_TEST_DEPENDS}
  COMMENT "Running ttlang pytest tests"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
  USES_TERMINAL
)
set_target_properties(check-ttlang-pytest PROPERTIES FOLDER "Tests")

# Middle end to end tests (test/me2e/).
# Run 'cmake --build build --target install-dev-deps' if tests fail due to missing pytest-order.
add_custom_target(check-ttlang-me2e
  COMMAND ${Python3_EXECUTABLE} -m pytest
    ${CMAKE_CURRENT_SOURCE_DIR}/me2e
    -v --tb=short
    --junitxml=${CMAKE_CURRENT_BINARY_DIR}/me2e-report.xml
  DEPENDS ${TTLANG_PYTHON_TEST_DEPENDS}
  COMMENT "Running ttlang me2e tests"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/me2e
  USES_TERMINAL
)
set_target_properties(check-ttlang-me2e PROPERTIES FOLDER "Tests")

# All tests including those requiring TT device.
# Uses sequential COMMAND invocations to ensure device-accessing tests
# (python-lit and pytest) don't run concurrently and cause TLB allocation failures.
# Uses USES_TERMINAL to show real-time progress output from each test suite.
add_custom_target(check-ttlang-all
  COMMAND ${CMAKE_COMMAND} -E echo "=== Running check-ttlang-mlir ==="
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-ttlang-mlir
  COMMAND ${CMAKE_COMMAND} -E echo "=== Running check-ttlang-python-bindings ==="
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-ttlang-python-bindings
  COMMAND ${CMAKE_COMMAND} -E echo "=== Running check-ttlang-python-lit ==="
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-ttlang-python-lit
  COMMAND ${CMAKE_COMMAND} -E echo "=== Running check-ttlang-me2e ==="
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-ttlang-me2e
  COMMAND ${CMAKE_COMMAND} -E echo "=== Running check-ttlang-pytest ==="
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-ttlang-pytest
  COMMAND ${CMAKE_COMMAND} -E echo "=== All ttlang tests completed ==="
  USES_TERMINAL
  COMMENT "Running all ttlang tests sequentially"
)
