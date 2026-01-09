# Test configuration

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Check for Tenstorrent device at configure time
ttlang_check_device_available(TTLANG_HAS_DEVICE)

if(TTLANG_HAS_DEVICE)
  set(TTLANG_HAS_DEVICE_INT 1)
else()
  set(TTLANG_HAS_DEVICE_INT 0)
endif()

# Generate test config file with device availability
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/python/test_config.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/python/test_config.py
  @ONLY
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

# Pytest tests (test_*.py files).
add_custom_target(check-ttlang-pytest
  COMMAND ${Python3_EXECUTABLE} -m pytest
    ${CMAKE_CURRENT_SOURCE_DIR}/python
    -v --tb=short
    --junitxml=${CMAKE_CURRENT_BINARY_DIR}/pytest-report.xml
  DEPENDS ${TTLANG_PYTHON_TEST_DEPENDS}
  COMMENT "Running ttlang pytest tests"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
)
set_target_properties(check-ttlang-pytest PROPERTIES FOLDER "Tests")

# All lit tests (including Python tests that require a TT device).
add_custom_target(check-ttlang-all
  DEPENDS check-ttlang-mlir check-ttlang-python-bindings check-ttlang-python-lit check-ttlang-pytest
  COMMENT "Running all ttlang tests (lit and pytest)"
)
