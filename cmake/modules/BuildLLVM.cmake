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
# Parse the expected LLVM commit SHA from tt-mlir's toolchain definition.
# Used to verify pre-built LLVM installations match the expected version.
# ---------------------------------------------------------------------------
set(_TTMLIR_ENV_CMAKELISTS "${CMAKE_SOURCE_DIR}/third-party/tt-mlir/env/CMakeLists.txt")
if(EXISTS "${_TTMLIR_ENV_CMAKELISTS}")
  file(STRINGS "${_TTMLIR_ENV_CMAKELISTS}" _llvm_version_line
       REGEX "set\\(LLVM_PROJECT_VERSION")
  if(_llvm_version_line)
    string(REGEX MATCH "\"([a-f0-9]+)\"" _match "${_llvm_version_line}")
    if(_match)
      set(_TTLANG_EXPECTED_LLVM_SHA "${CMAKE_MATCH_1}")
      ttlang_debug_message("Expected LLVM SHA (from tt-mlir): ${_TTLANG_EXPECTED_LLVM_SHA}")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# TTLANG_USE_TTMLIR_TOOLCHAIN: convenience option to use pre-built LLVM from
# the ttmlir toolchain directory ($TTMLIR_TOOLCHAIN_DIR or /opt/ttmlir-toolchain).
# ---------------------------------------------------------------------------
option(TTLANG_USE_TTMLIR_TOOLCHAIN "Use pre-built LLVM from ttmlir toolchain" OFF)

if(TTLANG_USE_TTMLIR_TOOLCHAIN AND NOT DEFINED MLIR_PREFIX)
  if(DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
    set(_toolchain_dir "$ENV{TTMLIR_TOOLCHAIN_DIR}")
  else()
    set(_toolchain_dir "/opt/ttmlir-toolchain")
  endif()

  if(NOT EXISTS "${_toolchain_dir}")
    message(FATAL_ERROR
      "TTLANG_USE_TTMLIR_TOOLCHAIN is ON but toolchain directory not found: ${_toolchain_dir}\n"
      "Set TTMLIR_TOOLCHAIN_DIR to the correct path, or disable this option.")
  endif()

  set(MLIR_PREFIX "${_toolchain_dir}")
  message(STATUS "Using ttmlir toolchain at: ${_toolchain_dir}")

  # Use the Python from the toolchain's venv so that MLIR Python bindings
  # (nanobind stubs, etc.) resolve against the same interpreter they were
  # built with.
  set(_toolchain_python "${_toolchain_dir}/venv/bin/python")
  if(EXISTS "${_toolchain_python}")
    set(Python3_EXECUTABLE "${_toolchain_python}" CACHE FILEPATH
      "Python interpreter from ttmlir toolchain" FORCE)
    message(STATUS "Using toolchain Python: ${_toolchain_python}")
  else()
    message(WARNING
      "Toolchain Python not found at ${_toolchain_python}.\n"
      "Falling back to system Python. Python binding compatibility is not guaranteed.")
  endif()
endif()

# ---------------------------------------------------------------------------
# Determine build mode: pre-built or submodule.
# ---------------------------------------------------------------------------
# Accept MLIR_PREFIX (friendly) or raw MLIR_DIR from user.
# Cache TTLANG_LLVM_FROM_SUBMODULE to remember the decision across reconfigures.
if(DEFINED MLIR_PREFIX)
  set(MLIR_DIR "${MLIR_PREFIX}/lib/cmake/mlir" CACHE PATH "MLIR CMake dir" FORCE)
  set(TTLANG_LLVM_FROM_SUBMODULE OFF CACHE BOOL "Whether LLVM is built from submodule" FORCE)
  message(STATUS "Using pre-built MLIR from prefix: ${MLIR_PREFIX}")
endif()

# Use cached TTLANG_LLVM_FROM_SUBMODULE to decide path on reconfigures.
# On first configure, if neither MLIR_PREFIX nor MLIR_DIR is user-provided,
# TTLANG_LLVM_FROM_SUBMODULE won't exist yet, so we fall through to else().
if(DEFINED TTLANG_LLVM_FROM_SUBMODULE AND NOT TTLANG_LLVM_FROM_SUBMODULE)
  # ---------------------------------------------------------------------------
  # Option A: Pre-built LLVM/MLIR
  # ---------------------------------------------------------------------------
  find_package(MLIR REQUIRED CONFIG)

  # Derive the install prefix from MLIR_DIR (strip lib/cmake/mlir).
  get_filename_component(LLVM_INSTALL_DIR "${MLIR_DIR}/../../.." ABSOLUTE)

  # Verify the pre-built LLVM matches tt-mlir's expected commit.
  if(DEFINED _TTLANG_EXPECTED_LLVM_SHA)
    ttlang_verify_llvm_sha("${LLVM_INSTALL_DIR}" "${_TTLANG_EXPECTED_LLVM_SHA}")
  endif()

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

  set(TTLANG_LLVM_FROM_SUBMODULE ON CACHE BOOL "Whether LLVM is built from submodule" FORCE)
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
    set(TTLANG_PYTHON_VENV "${CMAKE_BINARY_DIR}/venv" CACHE PATH "Python venv for MLIR" FORCE)
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

  else()
    message(STATUS "Reusing existing Python venv at ${TTLANG_PYTHON_VENV}")
  endif()

  # Install/update Python requirements on every configure (pip is a no-op when
  # packages are already satisfied, so this is cheap on subsequent runs).
  ttlang_pip_install_requirements("${_VENV_PYTHON}"
    "${LLVM_SUBMODULE_DIR}/mlir/python/requirements.txt" FATAL)
  ttlang_pip_install_requirements("${_VENV_PYTHON}"
    "${CMAKE_SOURCE_DIR}/requirements.txt" FATAL)

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
