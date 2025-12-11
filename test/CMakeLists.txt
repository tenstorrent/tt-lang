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

# All tests
add_lit_testsuite(check-ttlang "Running all ttlang tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    --verbose
    DEPENDS ${TTLANG_TEST_DEPENDS}
)
set_target_properties(check-ttlang PROPERTIES FOLDER "Tests")

# MLIR tests only
add_lit_testsuite(check-ttlang-mlir "Running ttlang MLIR tests"
    ${CMAKE_CURRENT_BINARY_DIR}/ttlang
    --verbose
    DEPENDS ${TTLANG_TEST_DEPENDS}
)
set_target_properties(check-ttlang-mlir PROPERTIES FOLDER "Tests")

# Python tests only
add_lit_testsuite(check-ttlang-python "Running ttlang Python tests"
    ${CMAKE_CURRENT_BINARY_DIR}/python
    --verbose
    DEPENDS ${TTLANG_TEST_DEPENDS}
)
set_target_properties(check-ttlang-python PROPERTIES FOLDER "Tests")

# Python bindings tests only
add_lit_testsuite(check-ttlang-python-bindings "Running ttlang Python binding tests"
    ${CMAKE_CURRENT_BINARY_DIR}/bindings/python
    --verbose
    DEPENDS ${TTLANG_TEST_DEPENDS}
)
set_target_properties(check-ttlang-python-bindings PROPERTIES FOLDER "Tests")
