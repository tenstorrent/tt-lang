# Test configuration

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Check for Tenstorrent device at configure time by looking for /dev/tenstorrent*
# This avoids the slow ttnn.GetNumAvailableDevices() call at test runtime
file(GLOB TT_DEVICE_FILES "/dev/tenstorrent*")
if(TT_DEVICE_FILES)
  set(TTLANG_HAS_DEVICE_INT 1)
  message(STATUS "Tenstorrent device detected - tests will run on hardware")
else()
  set(TTLANG_HAS_DEVICE_INT 0)
  message(STATUS "No Tenstorrent device detected - tests will run in compile-only mode")
endif()

# Generate test config file with device availability
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/python/test_config.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/python/test_config.py
  @ONLY
)

set(TTLANG_TEST_DEPENDS
  ttlang-opt
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

# Python tests only (run sequentially to avoid TLB resource contention
# when multiple tests open Tenstorrent devices). TT device is required for
# semantics verification (when not available, the tests are run in compile-only mode).
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

# All lit tests (including Python tests that require a TT device).
add_custom_target(check-ttlang-all
  DEPENDS check-ttlang-mlir check-ttlang-python-bindings check-ttlang-python-lit
  COMMENT "Running all ttlang lit tests (including Python tests that require a TT device)"
)
