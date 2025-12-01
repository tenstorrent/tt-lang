set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(TTLANG_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttlang")

################################################################################
# Build _ttlang extension
#
# tt-lang uses ttmlir.* for MLIR functionality. The _ttlang extension registers
# tt-lang passes into ttmlir's MLIR registry.
################################################################################

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_ttlang
  TTLangModule.cpp
)

# Suppress warnings for nanobind sources
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(nanobind-static PRIVATE
    -Wno-cast-qual
    -Wno-zero-length-array
    -Wno-nested-anon-types
    -Wno-c++98-compat-extra-semi
    -Wno-covered-switch-default
  )
endif()

# Link against ttmlir's Python CAPI to share the same MLIR registry
find_library(TTMLIR_PYTHON_CAPI_LIB
  NAMES TTMLIRPythonCAPI
  PATHS "${TTMLIR_BUILD_DIR}/python_packages/ttmlir/_mlir_libs"
  NO_DEFAULT_PATH
  REQUIRED
)

target_link_libraries(_ttlang PRIVATE
  TTLangTransforms
  ${TTMLIR_PYTHON_CAPI_LIB}
)

target_include_directories(_ttlang PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(_ttlang PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TTLANG_PYTHON_PACKAGES_DIR}/ttlang/_mlir_libs"
)

if(NOT APPLE)
  target_link_options(_ttlang PRIVATE
    -Wl,--no-as-needed
  )
endif()

################################################################################
# Install Python sources
################################################################################

# Ensure TTLANG_PYTHON_PACKAGES_DIR is set (should come from parent CMakeLists.txt)
if(NOT TTLANG_PYTHON_PACKAGES_DIR)
  message(FATAL_ERROR "TTLANG_PYTHON_PACKAGES_DIR is not set")
endif()

set(TTLANG_PYTHON_SOURCES
  __init__.py
  circular_buffer.py
  constants.py
  d2m_api.py
  dtype_utils.py
  layouts.py
  operators.py
  semaphore.py
  _src/__init__.py
  _src/codegen.py
  _src/d2m_ast.py
  _src/tensor_accessor.py
)

# Use absolute path to ensure we write to the correct location
set(TTLANG_OUTPUT_DIR "${TTLANG_PYTHON_PACKAGES_DIR}/ttlang")
file(MAKE_DIRECTORY "${TTLANG_OUTPUT_DIR}/_mlir_libs")
file(WRITE "${TTLANG_OUTPUT_DIR}/_mlir_libs/__init__.py" "")

foreach(src ${TTLANG_PYTHON_SOURCES})
  get_filename_component(src_dir ${src} DIRECTORY)
  if(src_dir)
    file(MAKE_DIRECTORY "${TTLANG_OUTPUT_DIR}/${src_dir}")
  endif()
  configure_file(
    "${TTLANG_PYTHON_ROOT_DIR}/${src}"
    "${TTLANG_OUTPUT_DIR}/${src}"
    COPYONLY
  )
endforeach()

install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/ DESTINATION . COMPONENT TTLangPythonWheel EXCLUDE_FROM_ALL)
