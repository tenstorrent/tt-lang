# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# External tt-mlir dependency management.
# Reads the required tt-mlir commit from third-party/tt-mlir.commit.
#
# Search priority:
# 1. Pre-built tt-mlir: User specifies path to build tree via TTMLIR_BUILD_DIR.
# 2. Pre-installed tt-mlir: TTMLIR_DIR pointing to TTMLIRConfig.cmake, or TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir.
# 3. FetchContent fallback: Build locally when neither is found.

set(TTMLIR_HINTS)

# Scenario 1: Pre-built tt-mlir (build tree)
if(DEFINED TTMLIR_BUILD_DIR)
  set(_TTMLIR_CONFIG_PATH "${TTMLIR_BUILD_DIR}/lib/cmake/ttmlir")
  if(EXISTS "${_TTMLIR_CONFIG_PATH}/TTMLIRConfig.cmake")
    list(APPEND TTMLIR_HINTS "${_TTMLIR_CONFIG_PATH}")
    set(_TTMLIR_BUILD_DIR "${TTMLIR_BUILD_DIR}")
  endif()
endif()

# Scenario 2: Pre-installed tt-mlir
if(DEFINED TTMLIR_DIR)
  list(APPEND TTMLIR_HINTS "${TTMLIR_DIR}")
endif()

if(DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
  set(TTMLIR_TOOLCHAIN_DIR "$ENV{TTMLIR_TOOLCHAIN_DIR}" CACHE PATH "tt-mlir toolchain installation directory")
elseif(NOT DEFINED TTMLIR_TOOLCHAIN_DIR)
  set(TTMLIR_TOOLCHAIN_DIR "/opt/ttmlir-toolchain" CACHE PATH "tt-mlir toolchain installation directory")
endif()

list(APPEND TTMLIR_HINTS "${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/ttmlir")

find_package(TTMLIR QUIET CONFIG HINTS ${TTMLIR_HINTS})

if(TTMLIR_FOUND)
  get_filename_component(_TTMLIR_ACTUAL_TOOLCHAIN_DIR "${TTMLIR_CMAKE_DIR}/../.." ABSOLUTE)
  set(TTMLIR_TOOLCHAIN_DIR "${_TTMLIR_ACTUAL_TOOLCHAIN_DIR}" CACHE PATH "tt-mlir toolchain installation directory" FORCE)

  if(DEFINED _TTMLIR_BUILD_DIR)
    message(STATUS "Using pre-built tt-mlir from build tree: ${_TTMLIR_BUILD_DIR}")
    load_cache("${_TTMLIR_BUILD_DIR}" READ_WITH_PREFIX _TTMLIR_
      CMAKE_HOME_DIRECTORY
      _Python3_EXECUTABLE
    )
    if(DEFINED _TTMLIR_TTMLIR_INSTALL_PREFIX)
      set(_TTMLIR_CACHE_INSTALL_PREFIX "${_TTMLIR_TTMLIR_INSTALL_PREFIX}")
      if(DEFINED TTMLIR_TOOLCHAIN_DIR AND NOT "${TTMLIR_TOOLCHAIN_DIR}" STREQUAL "${_TTMLIR_CACHE_INSTALL_PREFIX}")
        message(WARNING "TTMLIR_TOOLCHAIN_DIR differs from tt-mlir's configured installation prefix. Using tt-mlir's value: ${_TTMLIR_CACHE_INSTALL_PREFIX}")
      endif()
      set(TTMLIR_TOOLCHAIN_DIR "${_TTMLIR_CACHE_INSTALL_PREFIX}" CACHE PATH "tt-mlir toolchain installation directory" FORCE)
    endif()
    if(DEFINED _TTMLIR__Python3_EXECUTABLE)
      set(Python3_EXECUTABLE "${_TTMLIR__Python3_EXECUTABLE}" CACHE FILEPATH "Python 3 executable from tt-mlir build" FORCE)
      message(STATUS "Using Python from tt-mlir build: ${Python3_EXECUTABLE}")
      ttlang_get_parent_dir("${Python3_EXECUTABLE}" 2 _TTMLIR_EXTRACTED_TOOLCHAIN_DIR)
      set(TTMLIR_TOOLCHAIN_DIR "${_TTMLIR_EXTRACTED_TOOLCHAIN_DIR}" CACHE PATH "tt-mlir toolchain installation directory" FORCE)
    endif()
    if(DEFINED _TTMLIR_CMAKE_HOME_DIRECTORY)
      set(_TTMLIR_SOURCE_DIR "${_TTMLIR_CMAKE_HOME_DIRECTORY}")
      if(EXISTS "${_TTMLIR_SOURCE_DIR}/cmake/modules")
        message(STATUS "Found tt-mlir source directory: ${_TTMLIR_SOURCE_DIR}")
        list(APPEND CMAKE_MODULE_PATH "${_TTMLIR_SOURCE_DIR}/cmake/modules")
        include(TTMLIRBuildTypes OPTIONAL)
      endif()
    endif()
  else()
    set(_TTMLIR_VENV_DIR "${TTMLIR_TOOLCHAIN_DIR}/venv")
    if(EXISTS "${_TTMLIR_VENV_DIR}/bin/python3")
      set(Python3_EXECUTABLE "${_TTMLIR_VENV_DIR}/bin/python3" CACHE FILEPATH "Python 3 executable from tt-mlir" FORCE)
      message(STATUS "Using Python from tt-mlir installation: ${Python3_EXECUTABLE}")
    endif()
  endif()
else()
  # Scenario 3: FetchContent fallback - build locally
  set(TTMLIR_COMMIT_FILE "${CMAKE_SOURCE_DIR}/third-party/tt-mlir.commit")
  file(READ "${TTMLIR_COMMIT_FILE}" TTMLIR_GIT_TAG)
  string(STRIP "${TTMLIR_GIT_TAG}" TTMLIR_GIT_TAG)

  if("${TTMLIR_GIT_TAG}" STREQUAL "")
    message(FATAL_ERROR "tt-mlir.commit file does not contain a valid commit hash")
  endif()

  message(STATUS "tt-mlir not found. Building private copy version: ${TTMLIR_GIT_TAG}")
  set(_TTMLIR_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/ttmlir-toolchain")
  message(STATUS "tt-mlir will be installed to: ${_TTMLIR_INSTALL_PREFIX}")

  include(FetchContent)
  FetchContent_Declare(
      tt-mlir
      GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
      GIT_TAG ${TTMLIR_GIT_TAG}
      GIT_SUBMODULES_RECURSE TRUE
      SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/tt-mlir-src"
      BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/tt-mlir-build"
  )

  if(APPLE)
    set(TTMLIR_ENABLE_RUNTIME OFF CACHE BOOL "Enable tt-mlir runtime" FORCE)
    set(TTMLIR_ENABLE_RUNTIME_TESTS OFF CACHE BOOL "Enable tt-mlir runtime tests" FORCE)
  else()
    set(TTMLIR_ENABLE_RUNTIME ON CACHE BOOL "Enable tt-mlir runtime" FORCE)
    set(TTMLIR_ENABLE_RUNTIME_TESTS ON CACHE BOOL "Enable tt-mlir runtime tests" FORCE)
  endif()

  if(DEFINED CMAKE_INSTALL_PREFIX)
    set(_TTLANG_ORIGINAL_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  endif()

  set(CMAKE_INSTALL_PREFIX "${_TTMLIR_INSTALL_PREFIX}" CACHE PATH "Installation prefix for tt-mlir" FORCE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type for tt-mlir" FORCE)
  set(CMAKE_C_COMPILER "clang" CACHE STRING "C compiler for tt-mlir" FORCE)
  set(CMAKE_CXX_COMPILER "clang++" CACHE STRING "C++ compiler for tt-mlir" FORCE)
  set(TTMLIR_ENABLE_STABLEHLO OFF CACHE BOOL "Enable StableHLO in tt-mlir" FORCE)
  set(TT_RUNTIME_ENABLE_PERF_TRACE OFF CACHE BOOL "Enable performance tracing in tt-mlir runtime" FORCE)
  set(TTMLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "Enable Python bindings in tt-mlir" FORCE)
  set(TTMLIR_ENABLE_DEBUG_STRINGS ON CACHE BOOL "Enable debug strings in tt-mlir" FORCE)
  set(TTMLIR_ENABLE_EXPLORER OFF CACHE BOOL "Enable Explorer in tt-mlir" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "Build tests for tt-mlir" FORCE)

  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "ccache" CACHE STRING "C++ compiler launcher for tt-mlir" FORCE)
    message(STATUS "Using ccache for tt-mlir build")
  endif()

  FetchContent_MakeAvailable(tt-mlir)

  if(DEFINED _TTLANG_ORIGINAL_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${_TTLANG_ORIGINAL_INSTALL_PREFIX}" CACHE PATH "Installation prefix" FORCE)
  endif()

  find_package(TTMLIR REQUIRED CONFIG HINTS "${_TTMLIR_INSTALL_PREFIX}/lib/cmake/ttmlir")
  message(STATUS "Built and using private tt-mlir installation from: ${TTMLIR_CMAKE_DIR}")
  set(TTMLIR_TOOLCHAIN_DIR "${_TTMLIR_INSTALL_PREFIX}" CACHE PATH "tt-mlir toolchain installation directory" FORCE)

  set(_TTMLIR_VENV_DIR "${_TTMLIR_INSTALL_PREFIX}/venv")
  if(EXISTS "${_TTMLIR_VENV_DIR}/bin/python3")
    set(Python3_EXECUTABLE "${_TTMLIR_VENV_DIR}/bin/python3" CACHE FILEPATH "Python 3 executable from tt-mlir installation" FORCE)
    message(STATUS "Using Python from tt-mlir installation: ${Python3_EXECUTABLE}")
  endif()

  FetchContent_GetProperties(tt-mlir)
  if(tt-mlir_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${tt-mlir_SOURCE_DIR}/cmake/modules")
    include(TTMLIRBuildTypes OPTIONAL)
  endif()
endif()
