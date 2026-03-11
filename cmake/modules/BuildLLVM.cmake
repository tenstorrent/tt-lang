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
# TTLANG_TOOLCHAIN_DIR: directory for the tt-lang toolchain (LLVM, tt-metal,
# Python venv).  Works in two modes:
#
#   Build mode (default): LLVM and tt-metal are built from submodules and
#     installed into TTLANG_TOOLCHAIN_DIR.  Use this to create a reusable
#     toolchain from a from-source build.
#
#   Use mode (TTLANG_USE_TOOLCHAIN=ON): a pre-built toolchain at
#     TTLANG_TOOLCHAIN_DIR is consumed directly (no LLVM/tt-metal build).
#
# Can be set via -DTTLANG_TOOLCHAIN_DIR=... or the environment variable.
# Defaults to /opt/ttlang-toolchain when TTLANG_USE_TOOLCHAIN is ON.
# ---------------------------------------------------------------------------
if(NOT DEFINED TTLANG_TOOLCHAIN_DIR)
  if(DEFINED ENV{TTLANG_TOOLCHAIN_DIR})
    set(TTLANG_TOOLCHAIN_DIR "$ENV{TTLANG_TOOLCHAIN_DIR}" CACHE PATH
      "tt-lang toolchain directory")
  endif()
endif()

option(TTLANG_USE_TOOLCHAIN "Use pre-built LLVM from ttlang toolchain" OFF)
option(TTLANG_FORCE_TOOLCHAIN_REBUILD
  "Force rebuild of LLVM and tt-metal into TTLANG_TOOLCHAIN_DIR" OFF)

# Force rebuild implies build mode — override any cached state from a
# previous TTLANG_USE_TOOLCHAIN configure.
if(TTLANG_FORCE_TOOLCHAIN_REBUILD)
  set(TTLANG_USE_TOOLCHAIN OFF CACHE BOOL
    "Use pre-built LLVM from ttlang toolchain" FORCE)
  unset(MLIR_DIR CACHE)
endif()

if(TTLANG_USE_TOOLCHAIN AND NOT DEFINED MLIR_PREFIX)
  if(NOT DEFINED TTLANG_TOOLCHAIN_DIR)
    set(TTLANG_TOOLCHAIN_DIR "/opt/ttlang-toolchain" CACHE PATH
      "tt-lang toolchain directory" FORCE)
  endif()

  if(NOT EXISTS "${TTLANG_TOOLCHAIN_DIR}")
    message(FATAL_ERROR
      "TTLANG_USE_TOOLCHAIN is ON but toolchain directory not found: ${TTLANG_TOOLCHAIN_DIR}\n"
      "Set TTLANG_TOOLCHAIN_DIR to the correct path, or disable this option.")
  endif()

  set(MLIR_PREFIX "${TTLANG_TOOLCHAIN_DIR}")
  set(TTMETAL_BUILD_DIR "${TTLANG_TOOLCHAIN_DIR}/tt-metal" CACHE PATH
    "tt-metal build directory (from toolchain)" FORCE)
  set(TTLANG_PYTHON_VENV "${TTLANG_TOOLCHAIN_DIR}/venv" CACHE PATH
    "Python venv (from toolchain)" FORCE)
  message(STATUS "Using ttlang toolchain at: ${TTLANG_TOOLCHAIN_DIR}")

  # Use the Python from the toolchain's venv so that MLIR Python bindings
  # (nanobind stubs, etc.) resolve against the same interpreter they were
  # built with.  Set VIRTUAL_ENV so that find_package(Python3) with
  # Python3_FIND_VIRTUALENV=ONLY stays within the toolchain venv (this
  # overrides Python3_ROOT_DIR that actions/setup-python may inject).
  set(_toolchain_venv "${TTLANG_TOOLCHAIN_DIR}/venv")
  set(_toolchain_python "${_toolchain_venv}/bin/python3.12")

  if(EXISTS "${_toolchain_python}")
    set(ENV{VIRTUAL_ENV} "${_toolchain_venv}")
    set(Python3_FIND_VIRTUALENV ONLY)
    set(Python_FIND_VIRTUALENV ONLY)
    message(STATUS "Using toolchain Python: ${_toolchain_python}")
  else()
    message(WARNING
      "Toolchain Python not found at ${_toolchain_python}.\n"
      "Falling back to system Python. Python binding compatibility is not guaranteed.")
  endif()

elseif(DEFINED TTLANG_TOOLCHAIN_DIR AND NOT DEFINED MLIR_PREFIX)
  # Build mode: install toolchain components into TTLANG_TOOLCHAIN_DIR.
  set(LLVM_INSTALL_DIR "${TTLANG_TOOLCHAIN_DIR}" CACHE PATH
    "Install prefix for the submodule LLVM/MLIR build" FORCE)
  set(TTMETAL_BUILD_DIR "${TTLANG_TOOLCHAIN_DIR}/tt-metal" CACHE PATH
    "tt-metal build directory" FORCE)
  set(TTLANG_PYTHON_VENV "${TTLANG_TOOLCHAIN_DIR}/venv" CACHE PATH
    "Python venv" FORCE)

  if(TTLANG_FORCE_TOOLCHAIN_REBUILD)
    file(REMOVE "${TTLANG_TOOLCHAIN_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
    file(REMOVE "${TTLANG_TOOLCHAIN_DIR}/tt-metal/ttnn/_ttnn.so")
    # Remove LLVM build dir so it reconfigures with the correct install prefix.
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/llvm-build")
    message(STATUS "Forcing toolchain rebuild into: ${TTLANG_TOOLCHAIN_DIR}")
  else()
    message(STATUS "Building toolchain into: ${TTLANG_TOOLCHAIN_DIR}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Determine build mode: pre-built or submodule.
# ---------------------------------------------------------------------------
# Accept MLIR_PREFIX or raw MLIR_DIR from user.
if(DEFINED MLIR_PREFIX)
  set(MLIR_DIR "${MLIR_PREFIX}/lib/cmake/mlir" CACHE PATH "MLIR CMake dir" FORCE)
  message(STATUS "Using pre-built MLIR from prefix: ${MLIR_PREFIX}")
endif()

# ---------------------------------------------------------------------------
# Choose between pre-built LLVM (Option A) or submodule build (Option B).
# MLIR_PREFIX is set either explicitly by the user or by TTLANG_USE_TOOLCHAIN.
# ---------------------------------------------------------------------------
if(DEFINED MLIR_PREFIX OR DEFINED MLIR_DIR)
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
  ttlang_ensure_submodules(third-party/llvm-project)

  set(LLVM_INSTALL_DIR "${CMAKE_BINARY_DIR}/llvm-install" CACHE PATH
    "Install prefix for the submodule LLVM/MLIR build")
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

  ttlang_get_submodule_sha("${LLVM_SUBMODULE_DIR}" _LLVM_SUBMODULE_SHA)
  string(SUBSTRING "${_LLVM_SUBMODULE_SHA}" 0 7 _LLVM_SHORT_SHA)

  message(STATUS "Building LLVM/MLIR from submodule: ${LLVM_SUBMODULE_DIR}")
  message(STATUS "  Commit SHA:    ${_LLVM_SHORT_SHA}")
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
      COMMAND ${Python3_EXECUTABLE} -m venv --prompt ttlang "${TTLANG_PYTHON_VENV}"
      RESULT_VARIABLE _VENV_RESULT
    )
    if(NOT _VENV_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to create Python venv")
    endif()

    # Ensure 'python' symlink exists (some venvs only create python3).
    if(NOT EXISTS "${TTLANG_PYTHON_VENV}/bin/python")
      file(CREATE_LINK "python3" "${TTLANG_PYTHON_VENV}/bin/python" SYMBOLIC)
    endif()

    execute_process(
      COMMAND "${_VENV_PYTHON}" -m pip install --upgrade pip --quiet
    )

  else()
    message(STATUS "Reusing existing Python venv at ${TTLANG_PYTHON_VENV}")
  endif()

  # Install/update Python requirements on every configure (pip is a no-op when
  # packages are already satisfied, so this is cheap on subsequent runs).
  ttlang_pip_install_requirements("${_VENV_PYTHON}"
    "${LLVM_SUBMODULE_DIR}/mlir/python/requirements.txt" FATAL)
  ttlang_pip_install_requirements("${_VENV_PYTHON}"
    "${CMAKE_SOURCE_DIR}/requirements.txt" FATAL)

  # Install lit from the LLVM source tree so that llvm-lit can import it
  # regardless of whether the toolchain was built locally or restored from cache.
  ttlang_pip_install_package("${_VENV_PYTHON}"
    "${LLVM_SUBMODULE_DIR}/llvm/utils/lit" FATAL)

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

# ---------------------------------------------------------------------------
# clean-llvm target: removes LLVM build and install dirs so the next cmake
# configure rebuilds from scratch.
# ---------------------------------------------------------------------------
add_custom_target(clean-llvm
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/llvm-build"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVM_INSTALL_DIR}"
  COMMENT "Removing LLVM build and install directories. Re-run cmake configure to rebuild."
)
