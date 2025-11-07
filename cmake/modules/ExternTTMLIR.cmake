# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# External tt-mlir dependency management.
# Reads the required tt-mlir commit from third-party/tt-mlir.commit.
#
# Search priority:
# 1. Pre-built tt-mlir: User specifies path to build tree via TTMLIR_BUILD_DIR.
# 2. Pre-installed tt-mlir: TTMLIR_DIR pointing to TTMLIRConfig.cmake, or TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir.
# 3. FetchContent fallback: Build locally when neither is found.


# Set up Python from the toolchain venv (used by all scenarios).
set(Python3_ROOT_DIR "${TTMLIR_TOOLCHAIN_DIR}/venv")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python from toolchain: ${Python3_EXECUTABLE}")
set(_TOOLCHAIN_Python3_ROOT_DIR "${Python3_ROOT_DIR}")
set(_TOOLCHAIN_Python3_EXECUTABLE "${Python3_EXECUTABLE}")

# Scenario 1: Pre-built tt-mlir (build tree)
if(DEFINED TTMLIR_BUILD_DIR)
  set(_TTMLIR_CONFIG_PATH "${TTMLIR_BUILD_DIR}/lib/cmake/ttmlir")
  if(EXISTS "${_TTMLIR_CONFIG_PATH}/TTMLIRConfig.cmake")
    list(APPEND TTMLIR_HINTS "${_TTMLIR_CONFIG_PATH}")
    set(_TTMLIR_BUILD_DIR "${TTMLIR_BUILD_DIR}")
  endif()
endif()

# Scenario 2: Pre-installed tt-mlir (no source or build trees)
if(DEFINED TTMLIR_DIR)
  list(APPEND TTMLIR_HINTS "${TTMLIR_DIR}")
endif()

if(DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
  set(TTMLIR_TOOLCHAIN_DIR "$ENV{TTMLIR_TOOLCHAIN_DIR}" CACHE PATH "tt-mlir toolchain installation directory")
elseif(NOT DEFINED TTMLIR_TOOLCHAIN_DIR)
  set(TTMLIR_TOOLCHAIN_DIR "/opt/ttmlir-toolchain" CACHE PATH "tt-mlir toolchain installation directory")
endif()
message(STATUS "TTMLIR_TOOLCHAIN_DIR: ${TTMLIR_TOOLCHAIN_DIR}")

find_package(TTMLIR QUIET CONFIG HINTS ${TTMLIR_HINTS})

if(TTMLIR_FOUND)
  if(DEFINED _TTMLIR_BUILD_DIR)
    if(NOT EXISTS "${_TTMLIR_BUILD_DIR}/CMakeCache.txt")
      message(FATAL_ERROR "TTMLIR_BUILD_DIR points to an install directory, not a build directory. Please set TTMLIR_BUILD_DIR to the tt-mlir build directory (e.g., /path/to/tt-mlir/build), not the install directory.")
    endif()
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
    set(TTMLIR_PATH "${_TTMLIR_BUILD_DIR}")
  else()
    message(STATUS "Using pre-installed tt-mlir from: ${TTMLIR_CMAKE_DIR}")
    set(TTMLIR_PATH "${TTMLIR_TOOLCHAIN_DIR}")
  endif()
  find_package(MLIR REQUIRED CONFIG HINTS "${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir")
  find_package(LLVM REQUIRED CONFIG HINTS "${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/llvm")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  if(MLIR_ENABLE_BINDINGS_PYTHON AND TTLANG_ENABLE_BINDINGS_PYTHON)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
  endif()

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
else()
  # Scenario 3: FetchContent fallback - build locally
  set(TTMLIR_COMMIT_FILE "${CMAKE_SOURCE_DIR}/third-party/tt-mlir.commit")
  file(READ "${TTMLIR_COMMIT_FILE}" TTMLIR_GIT_TAG)
  string(STRIP "${TTMLIR_GIT_TAG}" TTMLIR_GIT_TAG)

  if("${TTMLIR_GIT_TAG}" STREQUAL "")
    message(FATAL_ERROR "tt-mlir.commit file does not contain a valid commit hash")
  endif()

  message(STATUS "tt-mlir not found. Building private copy version: ${TTMLIR_GIT_TAG}")

  if(APPLE)
    set(_TTMLIR_ENABLE_RUNTIME OFF)
    set(_TTMLIR_ENABLE_RUNTIME_TESTS OFF)
  else()
    set(_TTMLIR_ENABLE_RUNTIME ON)
    set(_TTMLIR_ENABLE_RUNTIME_TESTS ON)
  endif()

  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(_TTMLIR_CXX_LAUNCHER ccache)
  else()
    set(_TTMLIR_CXX_LAUNCHER "")
  endif()

  set(_TTMLIR_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/tt-mlir-install")

  message(STATUS "tt-mlir will be installed to: ${_TTMLIR_INSTALL_PREFIX}")

  include(FetchContent)
  FetchContent_Populate(
    tt-mlir
    GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
    GIT_TAG ${TTMLIR_GIT_TAG}
    GIT_SUBMODULES_RECURSE TRUE
    SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/tt-mlir-src"
  )

  set(_TTMLIR_SOURCE_DIR "${tt-mlir_SOURCE_DIR}")
  set(_TTMLIR_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/tt-mlir-build")
  file(MAKE_DIRECTORY "${_TTMLIR_BUILD_DIR}")

  set(_TTMLIR_CMAKE_ARGS
    -G Ninja
    -DCMAKE_INSTALL_PREFIX=${_TTMLIR_INSTALL_PREFIX}
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_CXX_COMPILER_LAUNCHER=${_TTMLIR_CXX_LAUNCHER}
    -DPython3_EXECUTABLE=${_TOOLCHAIN_Python3_EXECUTABLE}
    -DPython3_ROOT_DIR=${_TOOLCHAIN_Python3_ROOT_DIR}
    -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir
    -DLLVM_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/llvm
    -DTTMLIR_ENABLE_RUNTIME=${_TTMLIR_ENABLE_RUNTIME}
    -DTTMLIR_ENABLE_RUNTIME_TESTS=${_TTMLIR_ENABLE_RUNTIME_TESTS}
    -DTTMLIR_ENABLE_STABLEHLO=OFF
    -DTTMLIR_ENABLE_OPMODEL=ON
    -DTT_RUNTIME_ENABLE_PERF_TRACE=OFF
    -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DTTMLIR_ENABLE_DEBUG_STRINGS=ON
    -DTTMLIR_ENABLE_EXPLORER=OFF
    -DBUILD_TESTING=OFF
  )

  message(STATUS "Configuring tt-mlir...")
  set(ENV{TTMLIR_TOOLCHAIN_DIR} "${TTMLIR_TOOLCHAIN_DIR}")
  string(REPLACE ";" " " _TTMLIR_CMAKE_ARGS_STRING "${_TTMLIR_CMAKE_ARGS}")
  ttlang_execute_with_env(
    COMMAND "${CMAKE_COMMAND} ${_TTMLIR_CMAKE_ARGS_STRING} -S ${_TTMLIR_SOURCE_DIR} -B ${_TTMLIR_BUILD_DIR}"
    ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
    WORKING_DIRECTORY "${_TTMLIR_BUILD_DIR}"
  )

  message(STATUS "Building tt-mlir in ${_TTMLIR_BUILD_DIR}...")
  ttlang_execute_with_env(
    COMMAND "${CMAKE_COMMAND} --build ${_TTMLIR_BUILD_DIR} --target all --target TTMLIRPythonModules"
    ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
    WORKING_DIRECTORY "${_TTMLIR_BUILD_DIR}"
  )

  message(STATUS "Installing tt-mlir...")
  ttlang_execute_with_env(
    COMMAND "${CMAKE_COMMAND} --build ${_TTMLIR_BUILD_DIR} --target install"
    ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
    WORKING_DIRECTORY "${_TTMLIR_BUILD_DIR}"
  )

  message(STATUS "Installing tt-mlir Python wheel...")
  ttlang_execute_with_env(
    COMMAND "${CMAKE_COMMAND} --install ${_TTMLIR_BUILD_DIR} --component TTMLIRPythonWheel"
    ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
    WORKING_DIRECTORY "${_TTMLIR_BUILD_DIR}"
  )

  # Save original toolchain dir before finding the locally built tt-mlir.
  set(_ORIGINAL_TTMLIR_TOOLCHAIN_DIR "${TTMLIR_TOOLCHAIN_DIR}")

  message(STATUS "Searching for TTMLIRConfig.cmake in: ${_TTMLIR_INSTALL_PREFIX}/lib/cmake/ttmlir")
  find_package(TTMLIR REQUIRED CONFIG PATHS "${_TTMLIR_INSTALL_PREFIX}/lib/cmake/ttmlir")
  message(STATUS "Built and using private tt-mlir installation from: ${TTMLIR_CMAKE_DIR}")

  if(EXISTS "${_TTMLIR_SOURCE_DIR}/cmake/modules")
    list(APPEND CMAKE_MODULE_PATH "${_TTMLIR_SOURCE_DIR}/cmake/modules")
    include(TTMLIRBuildTypes OPTIONAL)
  endif()

  set(TTMLIR_PATH "${_TTMLIR_INSTALL_PREFIX}")

  # Set up MLIR and LLVM environment.
  find_package(MLIR REQUIRED CONFIG HINTS "${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir")
  find_package(LLVM REQUIRED CONFIG HINTS "${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/llvm")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  if(MLIR_ENABLE_BINDINGS_PYTHON AND TTLANG_ENABLE_BINDINGS_PYTHON)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
  endif()
endif()
