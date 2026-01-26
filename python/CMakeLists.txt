include(AddMLIRPython)

# Disables generation of "version soname" (i.e. libFoo.so.<version>)
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins
set(TTLANG_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttl")

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
# Generate elementwise operations from TTLElementwiseOps.def
# ###############################################################################
set(TTL_ELEMENTWISE_DEF "${CMAKE_SOURCE_DIR}/include/ttlang/Dialect/TTL/TTLElementwiseOps.def")
set(GEN_ELEMENTWISE_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/gen_elementwise.py")
set(GENERATED_ELEMENTWISE_PY "${CMAKE_CURRENT_BINARY_DIR}/ttl/_generated_elementwise.py")

add_custom_command(
  OUTPUT ${GENERATED_ELEMENTWISE_PY}
  COMMAND ${Python3_EXECUTABLE} ${GEN_ELEMENTWISE_SCRIPT}
          ${TTL_ELEMENTWISE_DEF} -o ${GENERATED_ELEMENTWISE_PY}
  DEPENDS ${TTL_ELEMENTWISE_DEF} ${GEN_ELEMENTWISE_SCRIPT}
  COMMENT "Generating elementwise ops Python bindings from TTLElementwiseOps.def"
  VERBATIM
)

add_custom_target(TTLangGeneratedElementwise DEPENDS ${GENERATED_ELEMENTWISE_PY})

# ###############################################################################
# Additional custom python code we want to bundle in the package
# ###############################################################################
declare_mlir_python_sources(TTLangPythonCommon)

declare_mlir_python_sources(TTLangPythonCommon.Main
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttl"
  ADD_TO_PARENT TTLangPythonCommon
  SOURCES
    __init__.py
    circular_buffer.py
    constants.py
    diagnostics.py
    kernel_runner.py
    ttl.py
    ttl_api.py
    ttl_math.py
    ttl_utils.py
    dtype_utils.py
    ir.py
    layouts.py
    operators.py
)

# Add generated elementwise ops file
declare_mlir_python_sources(TTLangPythonCommon.Generated
  ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ttl"
  ADD_TO_PARENT TTLangPythonCommon
  SOURCES
    _generated_elementwise.py
)

declare_mlir_python_sources(TTLangPythonCommon.Src
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttl"
  ADD_TO_PARENT TTLangPythonCommon
  SOURCES
    _src/__init__.py
    _src/auto_profile.py
    _src/ttl_ast.py
    _src/tensor_registry.py
)

declare_mlir_python_sources(TTLangPythonCommon.Utils
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT TTLangPythonCommon
  SOURCES
    utils/__init__.py
    utils/block_allocation.py
    utils/correctness.py
)

declare_mlir_python_sources(PykernelPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pykernel"
  SOURCES
    __init__.py
    _src/__init__.py
    _src/base_ast.py
    _src/kernel_ast.py
    _src/kernel_types.py
    _src/utils.py
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
else()
  if(DEFINED TTMLIR_PATH)
    list(APPEND _TTMLIR_PYTHON_CAPI_HINTS
      "${TTMLIR_PATH}/python_packages/ttmlir/_mlir_libs")
  endif()
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

# Create pykernel as a separate top-level package
add_mlir_python_modules(PykernelPythonModules
  ROOT_PREFIX "${TTLANG_PYTHON_PACKAGES_DIR}/pykernel"
  INSTALL_PREFIX "python_packages/pykernel"
  DECLARED_SOURCES
    PykernelPythonSources
)

add_mlir_python_modules(TTLangPythonModules
  ROOT_PREFIX "${TTLANG_PYTHON_PACKAGES_DIR}/ttl"
  INSTALL_PREFIX "python_packages/ttl"
  DECLARED_SOURCES
    TTLangPythonSources
    TTLangPythonExtensions
    TTLangPythonCommon
    TTLangPythonSiteInitialize
  COMMON_CAPI_LINK_LIBS
    TTLangPythonCAPI
)

# Ensure generated elementwise ops are created before Python modules are built
add_dependencies(TTLangPythonModules TTLangGeneratedElementwise)

# Link ttmlir's CAPI library for registry sharing
target_link_libraries(TTLangPythonModules.extension._ttlang.dso PRIVATE
  ${TTMLIR_PYTHON_CAPI_LIB}
)

# Ensure we do not accidentally resolve libTTMLIRPythonCAPI.so from a different
# tt-mlir install tree at runtime. Mixed CAPI builds in one Python process can
# lead to MLIRContext registry mismatches.
#
# Constrain RUNPATH to the configured tt-mlir build tree.
if(DEFINED TTMLIR_BUILD_DIR)
  set(_TTMLIR_PYTHON_LIBDIR "${TTMLIR_BUILD_DIR}/python_packages/ttmlir/_mlir_libs")
  set(_TTMLIR_LIBDIR "${TTMLIR_BUILD_DIR}/lib")
  if(EXISTS "${_TTMLIR_PYTHON_LIBDIR}" AND EXISTS "${_TTMLIR_LIBDIR}")
    set_target_properties(TTLangPythonModules.extension._ttlang.dso PROPERTIES
      BUILD_RPATH "\$ORIGIN:${_TTMLIR_LIBDIR}:${_TTMLIR_PYTHON_LIBDIR}"
      INSTALL_RPATH "\$ORIGIN:${_TTMLIR_LIBDIR}:${_TTMLIR_PYTHON_LIBDIR}"
    )
  endif()
endif()

# ###############################################################################
# Install
# ###############################################################################
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/
  DESTINATION .
  COMPONENT TTLangPythonWheel
  EXCLUDE_FROM_ALL)
