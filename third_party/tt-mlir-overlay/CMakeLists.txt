# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# tt-mlir Minimal Overlay
# This CMakeLists.txt builds only the minimum required tt-mlir targets for tt-lang:
# - TTCore, TTMetal, TTKernel dialects
# - TTIR dialect (dependency of TTKernelTargetCpp)
# - TTKernelToEmitC conversion
# - TTKernelTargetCpp translation
# - ttmlir-translate tool
#
# Part of "Plan: Minimal tt-metal and tt-mlir Integration for tt-lang"

cmake_minimum_required(VERSION 3.24.0)

# These variables must be set by the parent CMake before including this overlay:
# - ttmlir_SOURCE_DIR: Path to tt-mlir source checkout
# - ttmlir_BINARY_DIR: Path to tt-mlir build directory
# - TTMLIR_TOOLCHAIN_DIR: Path to LLVM/MLIR toolchain

if(NOT DEFINED ttmlir_SOURCE_DIR)
  message(FATAL_ERROR "ttmlir_SOURCE_DIR must be defined")
endif()

if(NOT DEFINED ttmlir_BINARY_DIR)
  message(FATAL_ERROR "ttmlir_BINARY_DIR must be defined")
endif()

if(NOT DEFINED TTMLIR_TOOLCHAIN_DIR)
  message(FATAL_ERROR "TTMLIR_TOOLCHAIN_DIR must be defined")
endif()

# Set up required variables that tt-mlir CMake files expect
set(TTMLIR_SOURCE_DIR ${ttmlir_SOURCE_DIR})
set(TTMLIR_BINARY_DIR ${ttmlir_BINARY_DIR})
set(PROJECT_SOURCE_DIR ${ttmlir_SOURCE_DIR})
set(CMAKE_SOURCE_DIR ${ttmlir_SOURCE_DIR})

# Include paths
# Add our overlay include directory FIRST to shadow upstream headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${ttmlir_SOURCE_DIR}/include)
include_directories(${ttmlir_BINARY_DIR}/include)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/include)

# Add tt-mlir's cmake modules
list(APPEND CMAKE_MODULE_PATH ${ttmlir_SOURCE_DIR}/cmake/modules)

# Required CMake modules from tt-mlir
include(AddMLIR)
include(GenerateRawStringHeader)
include(BuildFlatbuffers)

# ==============================================================================
# Tablegen targets (must be built first)
# ==============================================================================

# TTCore tablegen
add_subdirectory(${ttmlir_SOURCE_DIR}/include/ttmlir/Dialect/TTCore/IR
                 ${ttmlir_BINARY_DIR}/include/ttmlir/Dialect/TTCore/IR)

# TTMetal tablegen
add_subdirectory(${ttmlir_SOURCE_DIR}/include/ttmlir/Dialect/TTMetal/IR
                 ${ttmlir_BINARY_DIR}/include/ttmlir/Dialect/TTMetal/IR)

# TTKernel tablegen
add_subdirectory(${ttmlir_SOURCE_DIR}/include/ttmlir/Dialect/TTKernel/IR
                 ${ttmlir_BINARY_DIR}/include/ttmlir/Dialect/TTKernel/IR)

# TTIR tablegen (needed by TTKernelTargetCpp)
add_subdirectory(${ttmlir_SOURCE_DIR}/include/ttmlir/Dialect/TTIR/IR
                 ${ttmlir_BINARY_DIR}/include/ttmlir/Dialect/TTIR/IR)

# TTIR Transforms tablegen (needed by TTIR dialect)
add_subdirectory(${ttmlir_SOURCE_DIR}/include/ttmlir/Dialect/TTIR/Transforms
                 ${ttmlir_BINARY_DIR}/include/ttmlir/Dialect/TTIR/Transforms)

# Conversion passes tablegen
add_subdirectory(${ttmlir_SOURCE_DIR}/include/ttmlir/Conversion
                 ${ttmlir_BINARY_DIR}/include/ttmlir/Conversion)

# Common FlatBuffers targets (needed by TTCore for Target.h)
add_subdirectory(${ttmlir_SOURCE_DIR}/include/ttmlir/Target/Common
                 ${ttmlir_BINARY_DIR}/include/ttmlir/Target/Common)

# LLK generated headers (custom build with correct paths for overlay)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TTKernelLLKs.cmake)

# ==============================================================================
# Dialect libraries
# ==============================================================================

# TTCore dialect (custom build with FlatBuffers dependency)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TTCoreDialect.cmake)

# TTMetal dialect
add_subdirectory(${ttmlir_SOURCE_DIR}/lib/Dialect/TTMetal/IR
                 ${ttmlir_BINARY_DIR}/lib/Dialect/TTMetal/IR)

# TTKernel dialect
add_subdirectory(${ttmlir_SOURCE_DIR}/lib/Dialect/TTKernel/IR
                 ${ttmlir_BINARY_DIR}/lib/Dialect/TTKernel/IR)

# TTIR Utils (needed by TTIR dialect)
add_subdirectory(${ttmlir_SOURCE_DIR}/lib/Dialect/TTIR/Utils
                 ${ttmlir_BINARY_DIR}/lib/Dialect/TTIR/Utils)

# TTIR dialect (minimal version that excludes TTIROps.cpp which has TTNN dependencies)
# TTKernelTargetCpp only needs dialect registration, not the full ops implementation
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TTIRMinimal.cmake)

# ==============================================================================
# Conversion libraries
# ==============================================================================

# TTKernel to EmitC conversion
add_subdirectory(${ttmlir_SOURCE_DIR}/lib/Conversion/TTKernelToEmitC
                 ${ttmlir_BINARY_DIR}/lib/Conversion/TTKernelToEmitC)

# ==============================================================================
# Translation libraries
# ==============================================================================

# TTKernel to C++ translation target
add_subdirectory(${ttmlir_SOURCE_DIR}/lib/Target/TTKernel
                 ${ttmlir_BINARY_DIR}/lib/Target/TTKernel)

# ==============================================================================
# Aggregation target
# ==============================================================================

# Create a minimal aggregation target that links only what we need
add_library(TTMLIRMinimal INTERFACE)
target_link_libraries(TTMLIRMinimal INTERFACE
  MLIRTTCoreDialect
  MLIRTTMetalDialect
  MLIRTTKernelDialect
  MLIRTTIRDialect
  TTMLIRTTKernelToEmitC
  TTKernelTargetCpp
)

# ==============================================================================
# Python Bindings (optional)
# ==============================================================================

option(TTMLIR_MINIMAL_ENABLE_PYTHON "Build minimal Python bindings" ON)

if(TTMLIR_MINIMAL_ENABLE_PYTHON)
  message(STATUS "Building minimal Python bindings")

  # Disable soname versioning for Python packages
  set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

  # Python output directory
  set(TTMLIR_PYTHON_PACKAGES_DIR "${ttmlir_BINARY_DIR}/python_packages")

  # Package prefix for vendored MLIR
  add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttmlir.")

  # ==============================================================================
  # C API library for Python bindings
  # ==============================================================================
  # Build a minimal CAPI that links to TTMLIRMinimal instead of TTMLIRCompilerStatic

  add_library(TTMLIRMinimalCAPI STATIC
    ${ttmlir_SOURCE_DIR}/lib/CAPI/Dialects.cpp
    ${ttmlir_SOURCE_DIR}/lib/CAPI/TTCoreAttrs.cpp
    ${ttmlir_SOURCE_DIR}/lib/CAPI/TTKernelTypes.cpp
    ${ttmlir_SOURCE_DIR}/lib/CAPI/TTTypes.cpp
  )

  target_include_directories(TTMLIRMinimalCAPI PRIVATE
    ${ttmlir_SOURCE_DIR}/include
    ${ttmlir_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
  )

  target_link_libraries(TTMLIRMinimalCAPI PUBLIC
    TTMLIRMinimal
    MLIRIR
    MLIRCAPITransforms
    MLIRSupport
    MLIRTargetCpp
  )

  # ==============================================================================
  # Python Extensions (using nanobind directly, not MLIR's macros)
  # ==============================================================================
  # We build the extension manually because MLIR's declare_mlir_python_extension
  # expects relative paths from ROOT_DIR, but we have sources from multiple dirs.

  # Build the extension module using nanobind
  nanobind_add_module(_ttmlir_minimal
    NB_SHARED
    STABLE_ABI
    LTO
    # Custom minimal module sources
    ${CMAKE_CURRENT_SOURCE_DIR}/python/TTMLIRMinimalModule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/PassesMinimal.cpp
    # Upstream dialect-specific modules
    ${ttmlir_SOURCE_DIR}/python/TTModule.cpp
    ${ttmlir_SOURCE_DIR}/python/TTIRModule.cpp
    ${ttmlir_SOURCE_DIR}/python/TTKernelModule.cpp
    ${ttmlir_SOURCE_DIR}/python/Util.cpp
  )

  target_link_libraries(_ttmlir_minimal PRIVATE
    TTMLIRMinimal
    TTMLIRMinimalCAPI
    MLIRCAPITransforms
    MLIRTargetCpp
  )

  target_include_directories(_ttmlir_minimal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ttmlir_SOURCE_DIR}/include
    ${ttmlir_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
  )

  # Install to python_packages directory
  set_target_properties(_ttmlir_minimal PROPERTIES
    OUTPUT_NAME "_ttmlir"
    LIBRARY_OUTPUT_DIRECTORY "${TTMLIR_PYTHON_PACKAGES_DIR}/ttmlir/_mlir_libs"
  )

  # Note: We're not using MLIR's add_mlir_python_modules here because we build
  # the extension directly with nanobind. The Python package structure (dialect
  # bindings, etc.) should be provided by tt-mlir's installed wheel.
  #
  # For a full minimal build, we'd also need to copy/generate the Python source
  # files for the dialects, but for now we just build the native extension.

  message(STATUS "  Python extension: _ttmlir_minimal -> ${TTMLIR_PYTHON_PACKAGES_DIR}/ttmlir/_mlir_libs/")
endif()

message(STATUS "tt-mlir minimal overlay configured")
message(STATUS "  Source: ${ttmlir_SOURCE_DIR}")
message(STATUS "  Binary: ${ttmlir_BINARY_DIR}")
message(STATUS "  Targets: TTMLIRMinimal")
