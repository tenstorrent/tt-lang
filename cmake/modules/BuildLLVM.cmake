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
# Parse the expected LLVM commit SHA from tt-lang's own LLVM submodule.
# Used to verify pre-built LLVM installations match the expected version.
# ---------------------------------------------------------------------------
if(EXISTS "${LLVM_SUBMODULE_DIR}/.git")
  ttlang_get_submodule_sha("${LLVM_SUBMODULE_DIR}" _TTLANG_EXPECTED_LLVM_SHA)
  ttlang_debug_message("Expected LLVM SHA (from submodule): ${_TTLANG_EXPECTED_LLVM_SHA}")
  # If the submodule .git exists but the SHA could not be read (e.g. Docker
  # container with a gitlink but no actual objects), unset to skip verification.
  if(TTLANG_USE_TOOLCHAIN AND _TTLANG_EXPECTED_LLVM_SHA STREQUAL "unknown")
    unset(_TTLANG_EXPECTED_LLVM_SHA)
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
# TTLANG_USE_TOOLCHAIN and TTLANG_TOOLCHAIN_DIR are declared in CMakeLists.txt.
option(TTLANG_FORCE_TOOLCHAIN_REBUILD
  "Force rebuild of LLVM and tt-metal into TTLANG_TOOLCHAIN_DIR" OFF)

# ---------------------------------------------------------------------------
# TTLANG_BUILD_TOOLCHAIN: shortcut for building a fresh toolchain from source.
# Equivalent to TTLANG_TOOLCHAIN_DIR + TTLANG_USE_TOOLCHAIN=OFF with forced
# cleanup of stale artifacts.
# ---------------------------------------------------------------------------
if(TTLANG_BUILD_TOOLCHAIN AND TTLANG_USE_TOOLCHAIN)
  message(FATAL_ERROR
    "TTLANG_BUILD_TOOLCHAIN and TTLANG_USE_TOOLCHAIN are mutually exclusive.\n"
    "Use TTLANG_BUILD_TOOLCHAIN to build a toolchain from source.\n"
    "Use TTLANG_USE_TOOLCHAIN to consume an existing toolchain.")
endif()

if(TTLANG_BUILD_TOOLCHAIN)
  if(NOT DEFINED TTLANG_TOOLCHAIN_DIR)
    set(TTLANG_TOOLCHAIN_DIR "${CMAKE_BINARY_DIR}/toolchain-install" CACHE PATH
      "tt-lang toolchain directory" FORCE)
  endif()

  set(TTLANG_USE_TOOLCHAIN OFF CACHE BOOL
    "Use pre-built LLVM from ttlang toolchain" FORCE)
  unset(MLIR_DIR CACHE)

  # Clean stale artifacts so the build is always fresh.
  file(REMOVE "${TTLANG_TOOLCHAIN_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
  file(REMOVE "${TTLANG_TOOLCHAIN_DIR}/tt-metal/ttnn/_ttnn.so")
  file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/llvm-build")
  unset(Python3_EXECUTABLE CACHE)

  message(STATUS
    "TTLANG_BUILD_TOOLCHAIN: building fresh toolchain into ${TTLANG_TOOLCHAIN_DIR}")
endif()

# Force rebuild implies build mode — override any cached state from a
# previous TTLANG_USE_TOOLCHAIN configure.
if(TTLANG_FORCE_TOOLCHAIN_REBUILD)
  set(TTLANG_USE_TOOLCHAIN OFF CACHE BOOL
    "Use pre-built LLVM from ttlang toolchain" FORCE)
  unset(MLIR_DIR CACHE)
endif()

if(TTLANG_USE_TOOLCHAIN AND NOT DEFINED MLIR_PREFIX)
  if(NOT EXISTS "${TTLANG_TOOLCHAIN_DIR}")
    message(FATAL_ERROR
      "TTLANG_USE_TOOLCHAIN is ON but toolchain directory not found: ${TTLANG_TOOLCHAIN_DIR}\n"
      "Set TTLANG_TOOLCHAIN_DIR to the correct path, or disable this option.")
  endif()

  set(MLIR_PREFIX "${TTLANG_TOOLCHAIN_DIR}")
  message(STATUS "Using ttlang toolchain at: ${TTLANG_TOOLCHAIN_DIR}")

elseif(DEFINED TTLANG_TOOLCHAIN_DIR AND NOT DEFINED MLIR_PREFIX)
  # Build mode: install toolchain components into TTLANG_TOOLCHAIN_DIR.
  set(LLVM_INSTALL_DIR "${TTLANG_TOOLCHAIN_DIR}" CACHE PATH
    "Install prefix for the submodule LLVM/MLIR build" FORCE)

  unset(MLIR_DIR CACHE)

  if(TTLANG_FORCE_TOOLCHAIN_REBUILD)
    file(REMOVE "${TTLANG_TOOLCHAIN_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
    file(REMOVE "${TTLANG_TOOLCHAIN_DIR}/tt-metal/ttnn/_ttnn.so")
    # Remove LLVM build dir so it reconfigures with the correct install prefix.
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/llvm-build")
    # Clear cached Python3_EXECUTABLE so find_package discovers the system
    # interpreter instead of a stale venv path from a previous configure.
    unset(Python3_EXECUTABLE CACHE)
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
# Python venv setup.
#
# Both pre-built toolchain and submodule-build modes need a working Python
# venv.  The venv may not exist yet (first configure), or it may have been
# stripped from the toolchain cache because venvs are not portable across
# machines.  In either case, create a fresh venv using the system Python and
# install tt-lang's runtime requirements.
# ---------------------------------------------------------------------------
_ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _VENV_PYTHON)

if(NOT _VENV_PYTHON)
  message(STATUS "Creating Python venv at ${TTLANG_PYTHON_VENV}...")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m venv --prompt ttlang "${TTLANG_PYTHON_VENV}"
    RESULT_VARIABLE _VENV_RESULT
  )
  if(NOT _VENV_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to create Python venv")
  endif()

  _ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _VENV_PYTHON)
  _ttlang_activate_venv("${TTLANG_PYTHON_VENV}")

  execute_process(
    COMMAND "${_VENV_PYTHON}" -m pip install --upgrade pip --quiet
  )
endif()

# Install/update tt-lang Python requirements on every configure (pip is a
# no-op when packages are already satisfied).  requirements.txt includes all
# runtime dependencies: MLIR bindings, tt-metal/ttnn, and tt-lang itself.
ttlang_pip_install_requirements("${_VENV_PYTHON}"
  "${CMAKE_SOURCE_DIR}/requirements.txt" FATAL)

set(Python3_EXECUTABLE "${_VENV_PYTHON}")
message(STATUS "Python venv: ${TTLANG_PYTHON_VENV}")
message(STATUS "  Python: ${Python3_EXECUTABLE}")

# ---------------------------------------------------------------------------
# Pre-built LLVM/MLIR (toolchain or user-supplied MLIR_PREFIX/MLIR_DIR).
# ---------------------------------------------------------------------------
if(DEFINED MLIR_PREFIX OR DEFINED MLIR_DIR)
  find_package(MLIR REQUIRED CONFIG)

  # Derive the install prefix from MLIR_DIR (strip lib/cmake/mlir).
  get_filename_component(LLVM_INSTALL_DIR "${MLIR_DIR}/../../.." ABSOLUTE)

  # Verify the pre-built LLVM matches tt-lang's expected commit.
  if(DEFINED _TTLANG_EXPECTED_LLVM_SHA)
    ttlang_verify_llvm_sha("${LLVM_INSTALL_DIR}" "${_TTLANG_EXPECTED_LLVM_SHA}")
  endif()

# ---------------------------------------------------------------------------
# Build LLVM/MLIR from submodule (configure-time).
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

  # Install LLVM-specific Python build dependencies (nanobind, PyYAML, etc.
  # for MLIR Python bindings) and lit for test execution.
  ttlang_pip_install_requirements("${_VENV_PYTHON}"
    "${LLVM_SUBMODULE_DIR}/mlir/python/requirements.txt" FATAL)

  # Check if LLVM is already built (skip rebuild if install exists).
  if(EXISTS "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
    message(STATUS "LLVM/MLIR already built at ${LLVM_INSTALL_DIR}, skipping rebuild")
    # Warn if the submodule moved to a different commit than what was built.
    if(DEFINED _TTLANG_EXPECTED_LLVM_SHA)
      ttlang_verify_llvm_sha("${LLVM_INSTALL_DIR}" "${_TTLANG_EXPECTED_LLVM_SHA}")
    endif()
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
#
# When consuming a pre-built toolchain (TTLANG_USE_TOOLCHAIN), LLVM_INSTALL_DIR
# points to the shared toolchain root — deleting it would destroy the entire
# toolchain. Only register the target in build-from-source mode.
# ---------------------------------------------------------------------------
if(NOT TTLANG_USE_TOOLCHAIN)
  add_custom_target(clean-llvm
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/llvm-build"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVM_INSTALL_DIR}"
    COMMENT "Removing LLVM build and install directories. Re-run cmake configure to rebuild."
  )
endif()
