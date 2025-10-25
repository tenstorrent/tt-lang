include(AddMLIRPython)

# Disables generation of "version soname" (i.e. libFoo.so.<version>)
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins
set(TTLANG_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ttlang")

# We vendor our own MLIR instance in the `ttlang` namespace
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttlang.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(TTLangPythonSources)

declare_mlir_python_sources(TTLangPythonSources.Dialects
  ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TTLangPythonSources
)

# Example dialect bindings (to be added later):
# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT TTLangPythonSources.Dialects
#   ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
#   TD_FILE dialects/TTLangBinding.td
#   SOURCES dialects/ttlang.py
#   DIALECT_NAME ttlang
# )

################################################################################
# Extensions
################################################################################

declare_mlir_python_sources(TTLangPythonExtensions)

# Example Python extension (to be added later):
# declare_mlir_python_extension(TTLangPythonExtensions.Main
#   MODULE_NAME _ttlang
#   ADD_TO_PARENT TTLangPythonExtensions
#   SOURCES
#     TTLangModule.cpp
#   EMBED_CAPI_LINK_LIBS
#     MLIRCAPITransforms
#   PRIVATE_LINK_LIBS
#     # Add tt-lang libraries here
#   PYTHON_BINDINGS_LIBRARY nanobind
# )

set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "ttlang")

################################################################################
# Generate packages and shared library
################################################################################

# NOTE: Python bindings are DISABLED by default (TTLANG_ENABLE_BINDINGS_PYTHON=OFF)
set(TTLANG_PYTHON_SOURCES
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.affine
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.linalg
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.complex
  MLIRPythonSources.Dialects.ml_program
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.tosa
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.emitc
  MLIRPythonSources.Dialects.quant
  TTLangPythonSources
  TTLangPythonExtensions
)

add_mlir_python_common_capi_library(TTLangPythonCAPI
  INSTALL_COMPONENT TTLangPythonModules
  INSTALL_DESTINATION python_packages/ttlang/_mlir_libs
  OUTPUT_DIRECTORY "${TTLANG_PYTHON_PACKAGES_DIR}/ttlang/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${TTLANG_PYTHON_SOURCES}
)

add_mlir_python_modules(TTLangPythonModules
  ROOT_PREFIX "${TTLANG_PYTHON_PACKAGES_DIR}/ttlang"
  INSTALL_PREFIX "python_packages/ttlang"
  DECLARED_SOURCES ${TTLANG_PYTHON_SOURCES}
  COMMON_CAPI_LINK_LIBS
    TTLangPythonCAPI
)

# PyWheel Component to create install directives for the packaged ttlang bindings wheel
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/ DESTINATION . COMPONENT TTLangPythonWheel EXCLUDE_FROM_ALL)
