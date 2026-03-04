# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BuildLLVM.cmake - Dual-mode LLVM/MLIR dependency management.
#
# Option A (pre-built): User provides MLIR_PREFIX or MLIR_DIR pointing to an
#   LLVM/MLIR install. find_package(MLIR) transitively provides all LLVM settings.
#
# Option B (from submodule): Configure, build and install LLVM/MLIR from
#   third-party/llvm-project at CMake configure time using execute_process.
#   Then find_package(MLIR) against the fresh install, making all MLIR macros
#   available for the rest of the build.
#
# Control the LLVM build type independently via LLVM_BUILD_TYPE (default: Release).

set(LLVM_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/third-party/llvm-project")

# ---------------------------------------------------------------------------
# Option A: Pre-built LLVM/MLIR
# ---------------------------------------------------------------------------
# Accept MLIR_PREFIX (friendly) or raw MLIR_DIR.
if(DEFINED MLIR_PREFIX)
  set(MLIR_DIR "${MLIR_PREFIX}/lib/cmake/mlir" CACHE PATH "MLIR CMake dir" FORCE)
  message(STATUS "Using pre-built MLIR from prefix: ${MLIR_PREFIX}")
endif()

if(DEFINED MLIR_DIR)
  find_package(MLIR REQUIRED CONFIG)

  # Derive the install prefix from MLIR_DIR (strip lib/cmake/mlir).
  get_filename_component(LLVM_INSTALL_DIR "${MLIR_DIR}/../../.." ABSOLUTE)
  set(TTLANG_LLVM_FROM_SUBMODULE OFF)

# ---------------------------------------------------------------------------
# Option B: Build from submodule (configure-time)
# ---------------------------------------------------------------------------
else()
  if(NOT EXISTS "${LLVM_SUBMODULE_DIR}/llvm/CMakeLists.txt")
    message(FATAL_ERROR
      "LLVM submodule not initialized. Run:\n"
      "  git submodule update --init --depth 1 third-party/llvm-project\n"
      "Or provide a pre-built MLIR install via -DMLIR_PREFIX=/path/to/install")
  endif()

  set(TTLANG_LLVM_FROM_SUBMODULE ON)
  set(LLVM_INSTALL_DIR "${CMAKE_BINARY_DIR}/llvm-install")
  set(LLVM_BUILD_DIR "${CMAKE_BINARY_DIR}/llvm-build")

  # LLVM build type — independent of the parent project's CMAKE_BUILD_TYPE.
  if(NOT DEFINED LLVM_BUILD_TYPE)
    set(LLVM_BUILD_TYPE "Release")
  endif()

  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(_LLVM_CCACHE_BUILD ON)
  else()
    set(_LLVM_CCACHE_BUILD OFF)
  endif()

  message(STATUS "Building LLVM/MLIR from submodule: ${LLVM_SUBMODULE_DIR}")
  message(STATUS "  Build type:    ${LLVM_BUILD_TYPE}")
  message(STATUS "  Build dir:     ${LLVM_BUILD_DIR}")
  message(STATUS "  Install dir:   ${LLVM_INSTALL_DIR}")
  message(STATUS "  ccache:        ${_LLVM_CCACHE_BUILD}")

  # --- Python venv setup ---
  # MLIR Python bindings need pybind11, nanobind, numpy, etc.
  # Create a venv (or reuse existing) with these dependencies.
  if(NOT DEFINED TTLANG_PYTHON_VENV)
    set(TTLANG_PYTHON_VENV "${CMAKE_BINARY_DIR}/venv")
  endif()
  set(_VENV_PYTHON "${TTLANG_PYTHON_VENV}/bin/python3")

  if(NOT EXISTS "${_VENV_PYTHON}")
    message(STATUS "Creating Python venv at ${TTLANG_PYTHON_VENV}...")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -m venv "${TTLANG_PYTHON_VENV}"
      RESULT_VARIABLE _VENV_RESULT
    )
    if(NOT _VENV_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to create Python venv")
    endif()

    message(STATUS "Installing MLIR Python requirements...")
    set(_MLIR_REQUIREMENTS "${LLVM_SUBMODULE_DIR}/mlir/python/requirements.txt")
    execute_process(
      COMMAND "${_VENV_PYTHON}" -m pip install --quiet -r "${_MLIR_REQUIREMENTS}"
      RESULT_VARIABLE _PIP_RESULT
    )
    if(NOT _PIP_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to install MLIR Python requirements")
    endif()

  else()
    message(STATUS "Reusing existing Python venv at ${TTLANG_PYTHON_VENV}")
  endif()

  set(Python3_EXECUTABLE "${_VENV_PYTHON}")
  message(STATUS "  Python:        ${Python3_EXECUTABLE}")

  # Check if LLVM is already built (skip rebuild if install exists).
  if(EXISTS "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
    message(STATUS "LLVM/MLIR already built at ${LLVM_INSTALL_DIR}, skipping rebuild")
  else()
    set(_LLVM_CMAKE_ARGS
      -G Ninja
      -S "${LLVM_SUBMODULE_DIR}/llvm"
      -B "${LLVM_BUILD_DIR}"
      -DCMAKE_BUILD_TYPE=${LLVM_BUILD_TYPE}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}

      # Only build MLIR
      -DLLVM_ENABLE_PROJECTS=mlir

      # Minimal target: host architecture only
      -DLLVM_TARGETS_TO_BUILD=host

      # Install utilities (llvm-lit, FileCheck, etc.)
      -DLLVM_INSTALL_UTILS=ON

      # Assertions for catching bugs
      -DLLVM_ENABLE_ASSERTIONS=ON

      # Disable everything we don't need
      -DLLVM_INCLUDE_TESTS=OFF
      -DLLVM_INCLUDE_EXAMPLES=OFF
      -DLLVM_INCLUDE_BENCHMARKS=OFF
      -DLLVM_INCLUDE_DOCS=OFF
      -DLLVM_ENABLE_OCAMLDOC=OFF
      -DLLVM_ENABLE_LIBEDIT=OFF
      -DMLIR_INCLUDE_TESTS=OFF
      -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF

      # ccache
      -DLLVM_CCACHE_BUILD=${_LLVM_CCACHE_BUILD}

      # Python bindings
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    )

    # --- Configure ---
    message(STATUS "Configuring LLVM/MLIR...")
    execute_process(
      COMMAND ${CMAKE_COMMAND} ${_LLVM_CMAKE_ARGS}
      RESULT_VARIABLE _LLVM_CONFIG_RESULT
    )
    if(NOT _LLVM_CONFIG_RESULT EQUAL 0)
      message(FATAL_ERROR "LLVM configure failed (exit ${_LLVM_CONFIG_RESULT})")
    endif()

    # --- Build ---
    message(STATUS "Building LLVM/MLIR (this may take a while)...")
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build "${LLVM_BUILD_DIR}"
      RESULT_VARIABLE _LLVM_BUILD_RESULT
    )
    if(NOT _LLVM_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "LLVM build failed (exit ${_LLVM_BUILD_RESULT})")
    endif()

    # --- Install ---
    message(STATUS "Installing LLVM/MLIR to ${LLVM_INSTALL_DIR}...")
    execute_process(
      COMMAND ${CMAKE_COMMAND} --install "${LLVM_BUILD_DIR}"
      RESULT_VARIABLE _LLVM_INSTALL_RESULT
    )
    if(NOT _LLVM_INSTALL_RESULT EQUAL 0)
      message(FATAL_ERROR "LLVM install failed (exit ${_LLVM_INSTALL_RESULT})")
    endif()

    # llvm-lit is a Python script that cmake --install doesn't copy.
    # Install it manually from the build directory.
    file(COPY "${LLVM_BUILD_DIR}/bin/llvm-lit"
         DESTINATION "${LLVM_INSTALL_DIR}/bin"
         FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                          GROUP_READ GROUP_EXECUTE
                          WORLD_READ WORLD_EXECUTE)
  endif()

  # Now find the freshly installed MLIR.
  set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH "MLIR CMake dir" FORCE)
  find_package(MLIR REQUIRED CONFIG)
endif()

# ---------------------------------------------------------------------------
# Common setup — runs for both pre-built and submodule builds.
# At this point find_package(MLIR) has completed successfully.
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "LLVM install prefix: ${LLVM_INSTALL_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
