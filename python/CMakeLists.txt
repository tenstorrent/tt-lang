include(AddMLIRPython)

# Disables generation of "version soname" (i.e. libFoo.so.<version>)
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# Package prefix: all MLIR Python modules (upstream + tt-mlir + ttlang) are
# installed under the single `ttl` top-level package, following the
# industry convention (IREE uses iree.compiler., torch-mlir uses torch_mlir.).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttl.")

# Nanobind domain shared by _ttmlir and _ttlang extensions so bound C types
# are compatible across modules.
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "ttl")

# Root directories for Python source declarations
set(TTLANG_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttl")
set(TTMLIR_PYTHON_ROOT_DIR "${CMAKE_SOURCE_DIR}/third-party/tt-mlir/python/ttmlir")

# ###############################################################################
# ttmlir Python Sources (minimal tt-mlir dialects)
# ###############################################################################
declare_mlir_python_sources(TTMLIRMinPythonSources)

declare_mlir_python_sources(TTMLIRMinPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRMinPythonSources
)

# TTCore dialect Python bindings (ODS-generated from tt-mlir submodule)
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRMinPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTCoreBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTCoreEnumBinding.td
  SOURCES dialects/ttcore.py
  DIALECT_NAME ttcore
)

# TTKernel dialect Python bindings (ODS-generated from tt-mlir submodule)
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRMinPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTKernelBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTKernelEnumBinding.td
  SOURCES dialects/ttkernel.py
  DIALECT_NAME ttkernel
)

# passes.py re-export
declare_mlir_python_sources(TTMLIRMinPythonSources.TopLevel
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttmlir"
  ADD_TO_PARENT TTMLIRMinPythonSources
  SOURCES
    passes.py
)

# Site initialization for tt-mlir dialects (suffix _0, loaded before TTL's _1).
# Must be under _mlir_libs/ so the MLIR site-init chain finds it.
declare_mlir_python_sources(TTMLIRMinPythonSources.SiteInit
  ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTMLIRMinPythonSources
  SOURCES
    _mlir_libs/_site_initialize_0.py
)

# ###############################################################################
# ttmlir Python Extensions (_ttmlir nanobind module)
# ###############################################################################
declare_mlir_python_sources(TTMLIRMinPythonExtensions)

declare_mlir_python_extension(TTMLIRMinPythonExtensions.Main
  MODULE_NAME _ttmlir
  ADD_TO_PARENT TTMLIRMinPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    ttmlir/TT_MLIRMinimalExtension.cpp
    ttmlir/TT_MLIRMinimalPasses.cpp
    ttmlir/TT_MLIRMinimalTTModule.cpp
    ttmlir/TT_MLIRMinimalTTKernelModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPITransforms
    TTMLIRMinimalCAPI
)


# ###############################################################################
# ttlang Python Sources (TTL dialect)
# ###############################################################################
declare_mlir_python_sources(TTLangPythonSources)

# Note: dialects/__init__.py and dialects/_ods_common.py are provided by
# MLIRPythonSources — we only declare our TTL-specific dialect sources here.
declare_mlir_python_sources(TTLangPythonSources.Dialects
  ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTLangPythonSources
)

# TTL dialect Python bindings (ODS-generated)
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTLangPythonSources.Dialects
  ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTLBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTLEnumBinding.td
  SOURCES dialects/ttl.py
  DIALECT_NAME ttl
)

# Site initialization for TTL dialect (suffix _1, loaded after ttmlir's _0).
# Note: _mlir_libs/__init__.py is provided by MLIRPythonSources.
declare_mlir_python_sources(TTLangPythonSiteInitialize
  ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTLangPythonSources
  SOURCES
    _mlir_libs/_site_initialize_1.py
)

# ###############################################################################
# ttlang Python Extensions (_ttlang nanobind module)
# ###############################################################################
declare_mlir_python_sources(TTLangPythonExtensions)

declare_mlir_python_extension(TTLangPythonExtensions.Main
  MODULE_NAME _ttlang
  ADD_TO_PARENT TTLangPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    ttlang/TTLangModule.cpp
    ttlang/TTLModule.cpp
  EMBED_CAPI_LINK_LIBS
    TTLangCAPI
)

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

# Python config file for various CMake-computed settings, e.g., device availability
# (TTLANG_HAS_DEVICE and TTLANG_HAS_DEVICE_INT are set by top-level CMakeLists.txt)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ttl/config.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/ttl/config.py
  @ONLY
)

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
    compiler_options.py
    constants.py
    diagnostics.py
    kernel_runner.py
    pipe.py
    ttl.py
    ttl_api.py
    ttl_math.py
    ttl_utils.py
    dtype_utils.py
    layouts.py
    operators.py
    version.py
)

# Add generated elementwise ops file and config file
declare_mlir_python_sources(TTLangPythonCommon.Generated
  ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ttl"
  ADD_TO_PARENT TTLangPythonCommon
  SOURCES
    _generated_elementwise.py
    config.py
)

declare_mlir_python_sources(TTLangPythonCommon.Src
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttl"
  ADD_TO_PARENT TTLangPythonCommon
  SOURCES
    _src/__init__.py
    _src/auto_profile.py
    _src/perf_summary.py
    _src/perf_trace_server.py
    _src/signpost_profile.py
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
# Common CAPI library and module aggregation
# ###############################################################################

# Build a single common CAPI shared library that includes upstream MLIR,
# our minimal tt-mlir dialects, and ttlang dialects.
add_mlir_python_common_capi_library(TTLangPythonCAPI
  INSTALL_DESTINATION python_packages/ttl/_mlir_libs
  OUTPUT_DIRECTORY "${TTLANG_PYTHON_PACKAGES_DIR}/ttl/_mlir_libs"
  DECLARED_SOURCES
    # Upstream MLIR (MLIRPythonSources includes .Core and all sub-packages)
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    # Minimal tt-mlir
    TTMLIRMinPythonSources
    TTMLIRMinPythonExtensions
    # ttlang
    TTLangPythonSources
    TTLangPythonExtensions
)

# Create pykernel as a separate top-level package
add_mlir_python_modules(PykernelPythonModules
  ROOT_PREFIX "${TTLANG_PYTHON_PACKAGES_DIR}/pykernel"
  INSTALL_PREFIX "python_packages/pykernel"
  DECLARED_SOURCES
    PykernelPythonSources
)

# Main ttl Python package — includes upstream MLIR, tt-mlir subset, ttlang
# dialect bindings, and ttl runtime code, all under a single `ttl` package.
add_mlir_python_modules(TTLangPythonModules
  ROOT_PREFIX "${TTLANG_PYTHON_PACKAGES_DIR}/ttl"
  INSTALL_PREFIX "python_packages/ttl"
  DECLARED_SOURCES
    # Upstream MLIR (MLIRPythonSources includes .Core and all sub-packages)
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    # Minimal tt-mlir
    TTMLIRMinPythonSources
    TTMLIRMinPythonExtensions
    # ttlang dialect bindings
    TTLangPythonSources
    TTLangPythonExtensions
    # ttl runtime Python files (ttl_api.py, layouts.py, etc.)
    TTLangPythonCommon
  COMMON_CAPI_LINK_LIBS
    TTLangPythonCAPI
)

# Ensure generated elementwise ops are created before Python modules are built
add_dependencies(TTLangPythonModules TTLangGeneratedElementwise)

# Add include paths for the _ttmlir extension so wrapper .cpp files can find
# TTMLIRMinimalModule.h and the upstream tt-mlir Python sources (TTModule.cpp,
# TTKernelModule.cpp) that get #included from our wrappers.
target_include_directories(TTLangPythonModules.extension._ttmlir.dso PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/ttmlir"
  "${CMAKE_SOURCE_DIR}/third-party/tt-mlir/python"
  "${CMAKE_SOURCE_DIR}/lib/ttmlir-minimal/CAPI"
)

# ###############################################################################
# Install
# ###############################################################################
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/
  DESTINATION .
  COMPONENT TTLangPythonWheel
  EXCLUDE_FROM_ALL)
