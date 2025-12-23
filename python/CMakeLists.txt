include(AddMLIRPython)

# Disables generation of "version soname" (i.e. libFoo.so.<version>)
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins
set(TTLANG_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttlang")

# Align package prefix with ttmlir so bound types match upstream core.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttmlir.")

# ###############################################################################
# Python Sources
# ###############################################################################
declare_mlir_python_sources(TTLangPythonSources)

declare_mlir_python_sources(TTLangPythonSources.Dialects
  ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTLangPythonSources
  SOURCES
    dialects/__init__.py
    dialects/_ods_common.py
)

# Declare TTL dialect Python bindings
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTLangPythonSources.Dialects
  ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTLBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTLEnumBinding.td
  SOURCES dialects/ttl.py
  DIALECT_NAME ttl
)

# Site initialization for automatic dialect registration (use _1 since ttmlir uses _0)
declare_mlir_python_sources(TTLangPythonSiteInitialize
  ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTLangPythonSources
  SOURCES
    _mlir_libs/__init__.py
    _mlir_libs/_site_initialize_1.py
)

# ###############################################################################
# Python Extensions
# ###############################################################################
declare_mlir_python_sources(TTLangPythonExtensions)

declare_mlir_python_extension(TTLangPythonExtensions.Main
  MODULE_NAME _ttlang
  ADD_TO_PARENT TTLangPythonExtensions
  SOURCES
    TTLangModule.cpp
    TTLModule.cpp
  EMBED_CAPI_LINK_LIBS
    TTLangCAPI
  PRIVATE_LINK_LIBS
    TTLangCAPI
    TTLangTTLTransforms
    ${TTMLIR_PYTHON_CAPI_LIB}
  PYTHON_BINDINGS_LIBRARY nanobind
)

# Use the same domain as tt-mlir so we can share its _mlir extension types.
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "ttmlir")

# ###############################################################################
# Additional custom python code we want to bundle in the package
# ###############################################################################
declare_mlir_python_sources(TTLangPythonCommon)

declare_mlir_python_sources(TTLangPythonCommon.Main
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttlang"
  ADD_TO_PARENT TTLangPythonCommon
  SOURCES
    __init__.py
    circular_buffer.py
    constants.py
    ttl_api.py
    dtype_utils.py
    ir.py
    layouts.py
    operators.py
    semaphore.py
)

declare_mlir_python_sources(TTLangPythonCommon.Src
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttlang"
  ADD_TO_PARENT TTLangPythonCommon
  SOURCES
    _src/__init__.py
    _src/ttl_ast.py
    _src/tensor_accessor.py
    _src/tensor_registry.py
)

# ###############################################################################
# Generate packages and shared library
# ###############################################################################

# Link against ttmlir's Python CAPI to share the same MLIR registry.
# Try the build tree, install tree, and the toolchain venv site-packages.
set(_TTMLIR_PYTHON_CAPI_HINTS)
if(DEFINED TTMLIR_BUILD_DIR)
  list(APPEND _TTMLIR_PYTHON_CAPI_HINTS
    "${TTMLIR_BUILD_DIR}/python_packages/ttmlir/_mlir_libs")
endif()
if(DEFINED TTMLIR_PATH)
  list(APPEND _TTMLIR_PYTHON_CAPI_HINTS
    "${TTMLIR_PATH}/python_packages/ttmlir/_mlir_libs")
endif()
list(APPEND _TTMLIR_PYTHON_CAPI_HINTS
  "${TTMLIR_TOOLCHAIN_DIR}/python_packages/ttmlir/_mlir_libs"
  "${TTMLIR_TOOLCHAIN_DIR}/venv/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/ttmlir/_mlir_libs")

find_library(TTMLIR_PYTHON_CAPI_LIB
  NAMES TTMLIRPythonCAPI
  HINTS ${_TTMLIR_PYTHON_CAPI_HINTS}
  PATH_SUFFIXES _mlir_libs
  REQUIRED
)

# Use the upstream tt-mlir common CAPI directly and expose a lightweight
# interface target for tt-lang.
add_library(TTLangPythonCAPI INTERFACE)
target_link_libraries(TTLangPythonCAPI INTERFACE ${TTMLIR_PYTHON_CAPI_LIB} TTLangCAPI)

set(TTLANG_PYTHON_SOURCES
  TTLangPythonSources
  TTLangPythonExtensions
  TTLangPythonCommon
  TTLangPythonSiteInitialize
)

add_mlir_python_modules(TTLangPythonModules
  ROOT_PREFIX "${TTLANG_PYTHON_PACKAGES_DIR}/ttlang"
  INSTALL_PREFIX "python_packages/ttlang"
  DECLARED_SOURCES
    TTLangPythonSources
    TTLangPythonExtensions
    TTLangPythonCommon
    TTLangPythonSiteInitialize
  COMMON_CAPI_LINK_LIBS
    TTLangPythonCAPI
)

# Link ttmlir's CAPI library for registry sharing
target_link_libraries(TTLangPythonModules.extension._ttlang.dso PRIVATE
  ${TTMLIR_PYTHON_CAPI_LIB}
)

# ###############################################################################
# Install
# ###############################################################################
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/
  DESTINATION .
  COMPONENT TTLangPythonWheel
  EXCLUDE_FROM_ALL)
