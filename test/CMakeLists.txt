# Test configuration

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TTLANG_TEST_DEPENDS
  ttlang-opt
)

# Common llvm-lit options for consistent output across tt-lang test suites.
set(TTLANG_LIT_COMMON_ARGS
  --verbose
  --show-pass
  --show-skipped
  --show-unsupported
  --no-progress-bar
)

# Python binding tests require the ttlang Python extension module. Ensure it is
# built before running the full suite (and python-only suites), so imports of
# `ttlang._mlir_libs._ttlang` do not spuriously fail.
set(TTLANG_PYTHON_TEST_DEPENDS ${TTLANG_TEST_DEPENDS})
if(TARGET TTLangPythonModules)
  list(APPEND TTLANG_PYTHON_TEST_DEPENDS TTLangPythonModules)
endif()

# All tests (run sequentially to avoid TLB resource contention when
# Python tests open Tenstorrent devices). Use check-ttlang-mlir for
# parallel MLIR-only test execution.
add_lit_testsuite(check-ttlang "Running all ttlang tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ${TTLANG_LIT_COMMON_ARGS}
  --threads=1
  --xunit-xml-output report.xml
  DEPENDS ${TTLANG_PYTHON_TEST_DEPENDS}
)
set_target_properties(check-ttlang PROPERTIES FOLDER "Tests")

# MLIR tests only
add_lit_testsuite(check-ttlang-mlir "Running ttlang MLIR tests"
  ${CMAKE_CURRENT_BINARY_DIR}/ttlang
  ${TTLANG_LIT_COMMON_ARGS}
  DEPENDS ${TTLANG_TEST_DEPENDS}
)
set_target_properties(check-ttlang-mlir PROPERTIES FOLDER "Tests")

# Python tests only (run sequentially to avoid TLB resource contention
# when multiple tests open Tenstorrent devices)
add_lit_testsuite(check-ttlang-python "Running ttlang Python tests"
  ${CMAKE_CURRENT_BINARY_DIR}/python
  ${TTLANG_LIT_COMMON_ARGS}
  --threads=1
  DEPENDS ${TTLANG_PYTHON_TEST_DEPENDS}
)
set_target_properties(check-ttlang-python PROPERTIES FOLDER "Tests")

# Python bindings tests only
add_lit_testsuite(check-ttlang-python-bindings "Running ttlang Python binding tests"
  ${CMAKE_CURRENT_BINARY_DIR}/bindings/python
  ${TTLANG_LIT_COMMON_ARGS}
  DEPENDS ${TTLANG_PYTHON_TEST_DEPENDS}
)
set_target_properties(check-ttlang-python-bindings PROPERTIES FOLDER "Tests")
