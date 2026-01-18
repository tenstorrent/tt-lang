# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# External tt-mlir dependency management.
# Reads the required tt-mlir commit from third-party/tt-mlir.commit.
#
# Search priority:
# 1. Pre-built tt-mlir: User specifies path to build tree via TTMLIR_BUILD_DIR.
# 2. Pre-installed tt-mlir: TTMLIR_DIR pointing to TTMLIRConfig.cmake, or TTMLIR_TOOLCHAIN_DIR/lib/cmake/ttmlir.
# 3. FetchContent fallback: Build locally when neither is found.

# Create alias targets for installed tt-mlir exported targets.
# When tt-mlir is installed, targets are exported with the TTMLIR:: namespace prefix
# (e.g., TTMLIR::MLIRTTKernelDialect). tt-lang's CMakeLists files reference these
# targets without the prefix (e.g., MLIRTTKernelDialect) for consistency with
# tt-mlir build tree usage. This function creates non-namespaced aliases.
function(ttlang_create_ttmlir_aliases)
  if(NOT DEFINED TTMLIR_EXPORTED_TARGETS)
    message(WARNING "TTMLIR_EXPORTED_TARGETS not defined, skipping alias creation")
    return()
  endif()

  set(_alias_count 0)
  foreach(target IN LISTS TTMLIR_EXPORTED_TARGETS)
    if(TARGET TTMLIR::${target} AND NOT TARGET ${target})
      add_library(${target} ALIAS TTMLIR::${target})
      math(EXPR _alias_count "${_alias_count} + 1")
    endif()
  endforeach()
  message(STATUS "Created ${_alias_count} aliases for installed tt-mlir targets")
endfunction()

# Scenario 1: Pre-built tt-mlir (build tree)
if(DEFINED TTMLIR_BUILD_DIR)
  # Check if it's a valid build directory (has CMakeCache.txt)
  if(EXISTS "${TTMLIR_BUILD_DIR}/CMakeCache.txt")
    set(_TTMLIR_BUILD_DIR "${TTMLIR_BUILD_DIR}")
    set(_TTMLIR_CONFIG_PATH "${TTMLIR_BUILD_DIR}/lib/cmake/ttmlir")
  else()
    message(FATAL_ERROR "Could not find CMakeCache.txt in the provided build directory: ${TTMLIR_BUILD_DIR}")
  endif()
endif()

# Scenario 2: Pre-installed tt-mlir (no source or build trees)
if(DEFINED TTMLIR_DIR)
  list(APPEND TTMLIR_HINTS "${TTMLIR_DIR}")
endif()

if(DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
  set(TTMLIR_TOOLCHAIN_DIR "$ENV{TTMLIR_TOOLCHAIN_DIR}" CACHE PATH "tt-mlir toolchain installation directory")
  set(_TTMLIR_TOOLCHAIN_DIR_FROM_ENV TRUE)
elseif(NOT DEFINED TTMLIR_TOOLCHAIN_DIR)
  set(TTMLIR_TOOLCHAIN_DIR "/opt/ttmlir-toolchain" CACHE PATH "tt-mlir toolchain installation directory")
  set(_TTMLIR_TOOLCHAIN_DIR_FROM_ENV FALSE)
else()
  set(_TTMLIR_TOOLCHAIN_DIR_FROM_ENV FALSE)
endif()

message(STATUS "TTMLIR_TOOLCHAIN_DIR: ${TTMLIR_TOOLCHAIN_DIR}")

if(EXISTS "${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/ttmlir")
  list(APPEND TTMLIR_HINTS "${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/ttmlir")
endif()

# Ensure we use Python from the toolchain venv.
#
# Note: MLIR's Python bindings configuration runs both `find_package(Python3 ...)`
# and `find_package(Python ...)` (nanobind expects Python_ variables). If they
# resolve to different interpreters, the computed extension suffix (SOABI) can
# mismatch and produce an un-importable module name (e.g. `_ttlang.cpython-314-*.so`
# for a Python 3.11 interpreter).
if(EXISTS "${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python3")
  # Set as regular variables (not cache) to take precedence over find_package.
  set(Python3_FIND_VIRTUALENV ONLY)
  set(Python3_ROOT_DIR "${TTMLIR_TOOLCHAIN_DIR}/venv")
  set(Python3_EXECUTABLE "${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python3")

  set(Python_FIND_VIRTUALENV ONLY)
  set(Python_ROOT_DIR "${TTMLIR_TOOLCHAIN_DIR}/venv")
  set(Python_EXECUTABLE "${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python3")

  message(STATUS "Using Python from toolchain: ${Python3_EXECUTABLE}")
endif()

if(NOT DEFINED _TTMLIR_BUILD_DIR)
  find_package(TTMLIR QUIET CONFIG HINTS ${TTMLIR_HINTS})
endif()

if(TTMLIR_FOUND OR DEFINED _TTMLIR_BUILD_DIR)
  if(DEFINED _TTMLIR_BUILD_DIR)
    # Scenario 1: Using a pre-built tt-mlir build tree
    ttlang_setup_ttmlir_build_tree(${_TTMLIR_BUILD_DIR})
  else()
    message(STATUS "Using pre-installed tt-mlir from: ${TTMLIR_CMAKE_DIR}")

    # Set TTMLIR_PATH to the installation root (parent of lib/cmake/ttmlir)
    if(NOT EXISTS "${TTMLIR_CMAKE_DIR}")
      message(FATAL_ERROR "TTMLIR_CMAKE_DIR does not exist: ${TTMLIR_CMAKE_DIR}. Can't install tt-lang from the specified installed tt-mlir location.")
    endif()

    get_filename_component(TTMLIR_PATH "${TTMLIR_CMAKE_DIR}/../../.." ABSOLUTE)

    if(TARGET TTMLIR::TTMLIRCompilerStatic)
      set(TTMLIR_LINK_LIBS TTMLIR::TTMLIRCompilerStatic)
      message(STATUS "Using TTMLIR library: TTMLIR::TTMLIRCompilerStatic")
    else()
      # TTMLIRCompilerStatic is not properly exported in TTMLIRInstall.cmake.
      # Try to find the static library file directly.
      find_library(TTMLIR_COMPILER_STATIC_LIB
        NAMES libTTMLIRCompilerStatic.a TTMLIRCompilerStatic
        PATHS "${TTMLIR_PATH}/lib"
        NO_DEFAULT_PATH
      )

      if(TTMLIR_COMPILER_STATIC_LIB)
        message(STATUS "TTMLIRCompilerStatic target not exported, using library file directly: ${TTMLIR_COMPILER_STATIC_LIB}")
        set(TTMLIR_LINK_LIBS "${TTMLIR_COMPILER_STATIC_LIB}")
      else()
        message(FATAL_ERROR "Required TTMLIR::TTMLIRCompilerStatic target not found in installed tt-mlir at: ${TTMLIR_CMAKE_DIR}, and library file not found in ${TTMLIR_PATH}/lib")
      endif()
    endif()

    # Create aliases for namespaced targets so tt-lang can use unprefixed names.
    ttlang_create_ttmlir_aliases()
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

  # For build tree (scenario 1), collect all tt-mlir and MLIR libraries
  if(DEFINED _TTMLIR_BUILD_DIR)
    ttlang_collect_ttmlir_link_libs(TTMLIR_LINK_LIBS)
  endif()

  if(MLIR_ENABLE_BINDINGS_PYTHON AND TTLANG_ENABLE_BINDINGS_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    message(STATUS "Using Python: ${Python3_EXECUTABLE}")
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
  endif()

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
else()
  # Scenario 3: Use FetchContent to build tt-mlir locally
  # Check if commit was provided via CMake variable, otherwise read from file
  if(NOT DEFINED TTMLIR_GIT_TAG)
    set(TTMLIR_COMMIT_FILE "${CMAKE_SOURCE_DIR}/third-party/tt-mlir.commit")
    file(READ "${TTMLIR_COMMIT_FILE}" TTMLIR_GIT_TAG)
    string(STRIP "${TTMLIR_GIT_TAG}" TTMLIR_GIT_TAG)
  else()
    string(STRIP "${TTMLIR_GIT_TAG}" TTMLIR_GIT_TAG)
  endif()

  if("${TTMLIR_GIT_TAG}" STREQUAL "")
    message(FATAL_ERROR "tt-mlir commit hash not specified. Either set TTMLIR_GIT_TAG or ensure third-party/tt-mlir.commit contains a valid commit hash")
  endif()

  message(STATUS "tt-mlir not found. Building private copy version: ${TTMLIR_GIT_TAG}")

  set(_TOOLCHAIN_Python3_ROOT_DIR "${TTMLIR_TOOLCHAIN_DIR}/venv")
  set(_TOOLCHAIN_Python3_EXECUTABLE "${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python3")
  set(_TTMLIR_TARGETS_TO_BUILD "-t all")

  # Check if TTMLIR_ENABLE_RUNTIME is already set (e.g., from workflow), otherwise default based on platform
  if(NOT DEFINED TTMLIR_ENABLE_RUNTIME)
    if(APPLE)
      set(_TTMLIR_ENABLE_PERF_TRACE OFF)
    else()
      set(_TTMLIR_ENABLE_RUNTIME ON)
      set(_TTMLIR_ENABLE_RUNTIME_TESTS OFF)

      if(TTLANG_ENABLE_PERF_TRACE)
        set(_TTMLIR_ENABLE_PERF_TRACE ON)
      else()
        set(_TTMLIR_ENABLE_PERF_TRACE OFF)
      endif()
    endif()
  else()
    # Use the provided value and derive related settings
    set(_TTMLIR_ENABLE_RUNTIME ${TTMLIR_ENABLE_RUNTIME})

    if(NOT DEFINED TTMLIR_ENABLE_RUNTIME_TESTS)
      set(_TTMLIR_ENABLE_RUNTIME_TESTS ${TTMLIR_ENABLE_RUNTIME})
    else()
      set(_TTMLIR_ENABLE_RUNTIME_TESTS ${TTMLIR_ENABLE_RUNTIME_TESTS})
    endif()

    if(TTLANG_ENABLE_PERF_TRACE AND _TTMLIR_ENABLE_RUNTIME)
      set(_TTMLIR_ENABLE_PERF_TRACE ON)
    else()
      set(_TTMLIR_ENABLE_PERF_TRACE OFF)
    endif()
  endif()

  if(NOT DEFINED TTMLIR_CMAKE_BUILD_TYPE)
    set(_TTMLIR_CMAKE_BUILD_TYPE "RelWithDebInfo")
  else()
    set(_TTMLIR_CMAKE_BUILD_TYPE "${TTMLIR_CMAKE_BUILD_TYPE}")
  endif()

  find_program(CCACHE_PROGRAM ccache)

  if(CCACHE_PROGRAM)
    set(_TTMLIR_CXX_LAUNCHER ccache)
  else()
    set(_TTMLIR_CXX_LAUNCHER "")
  endif()

  if(NOT DEFINED TTMLIR_INSTALL_PREFIX)
    set(TTMLIR_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/tt-mlir-install" CACHE PATH "Installation prefix for automatically built tt-mlir")
  endif()

  set(_TTMLIR_INSTALL_PREFIX "${TTMLIR_INSTALL_PREFIX}")

  message(STATUS "tt-mlir will be installed to: ${_TTMLIR_INSTALL_PREFIX}")
  set(_TTMLIR_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/tt-mlir-src")
  set(_TTMLIR_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/tt-mlir-build")

  # Check if TTMLIR_SRC_DIR is provided and SHA matches. If not, fetch tt-mlir from GitHub.
  set(_USE_PROVIDED_SRC_DIR FALSE)
  if(DEFINED TTMLIR_SRC_DIR)
    set(_CHECK_SCRIPT "${CMAKE_SOURCE_DIR}/.github/scripts/check-ttmlir-src-dir.sh")
    execute_process(
      COMMAND bash "${_CHECK_SCRIPT}" "${TTMLIR_SRC_DIR}" "${TTMLIR_GIT_TAG}"
      RESULT_VARIABLE _CHECK_RESULT
    )
    if(_CHECK_RESULT EQUAL 0)
      message(STATUS "Using provided tt-mlir source directory: ${TTMLIR_SRC_DIR} (SHA matches: ${TTMLIR_GIT_TAG})")
      set(_TTMLIR_SOURCE_DIR "${TTMLIR_SRC_DIR}")
      set(_USE_PROVIDED_SRC_DIR TRUE)

    else()
      message(STATUS "Provided tt-mlir source directory does not match required SHA (${TTMLIR_GIT_TAG}), will fetch instead")
    endif()
  endif()

  # Only use FetchContent if we don't have a matching source directory
  if(NOT _USE_PROVIDED_SRC_DIR)
    include(FetchContent)
    FetchContent_Populate(
        tt-mlir
        GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
        GIT_TAG ${TTMLIR_GIT_TAG}
        GIT_SUBMODULES_RECURSE TRUE
        SOURCE_DIR "${_TTMLIR_SOURCE_DIR}"
        BINARY_DIR "${_TTMLIR_BUILD_DIR}"
    )
    set(_TTMLIR_SOURCE_DIR "${tt-mlir_SOURCE_DIR}")
    set(_TTMLIR_BUILD_DIR "${tt-mlir_BINARY_DIR}")
  endif()

  set(_TTMLIR_CMAKE_ARGS
      -G Ninja
      -B ${_TTMLIR_BUILD_DIR}
      -DCMAKE_INSTALL_PREFIX=${_TTMLIR_INSTALL_PREFIX}
      -DTTMLIR_ENABLE_SHARED_LIB=ON
      -DCMAKE_BUILD_TYPE=${_TTMLIR_CMAKE_BUILD_TYPE}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${_TTMLIR_CXX_LAUNCHER}
      -DPython3_EXECUTABLE=${_TOOLCHAIN_Python3_EXECUTABLE}
      -DPython3_ROOT_DIR=${_TOOLCHAIN_Python3_ROOT_DIR}
      -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir
      -DLLVM_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/llvm
      -DTTMLIR_ENABLE_RUNTIME=${_TTMLIR_ENABLE_RUNTIME}
      -DTTMLIR_ENABLE_RUNTIME_TESTS=OFF
      -DTTMLIR_ENABLE_STABLEHLO=OFF
      -DTTMLIR_ENABLE_OPMODEL=OFF
      -DTTMLIR_ENABLE_EXPLORER=OFF
      -DTT_RUNTIME_ENABLE_PERF_TRACE=${_TTMLIR_ENABLE_PERF_TRACE}
      -DTTMLIR_ENABLE_BINDINGS_PYTHON=${TTLANG_ENABLE_BINDINGS_PYTHON}
      -DTT_RUNTIME_ENABLE_TTNN=ON
      -DTTMLIR_ENABLE_TTNN_JIT=ON
      -DUSE_TTNN_JIT_WHEEL=OFF
  )

  message(STATUS "Configuring tt-mlir...")
  set(ENV{TTMLIR_TOOLCHAIN_DIR} "${TTMLIR_TOOLCHAIN_DIR}")
  string(REPLACE ";" " " _TTMLIR_CMAKE_ARGS_STRING "${_TTMLIR_CMAKE_ARGS}")
  ttlang_debug_message("Configuring tt-mlir with: ${_TTMLIR_CMAKE_ARGS_STRING}")
  ttlang_execute_with_env(
      COMMAND "${CMAKE_COMMAND} ${_TTMLIR_CMAKE_ARGS_STRING} -S ${_TTMLIR_SOURCE_DIR} -B ${_TTMLIR_BUILD_DIR}"
      ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
  )

  message(STATUS "Building tt-mlir in ${_TTMLIR_BUILD_DIR}...")
  ttlang_execute_with_env(
      COMMAND "${CMAKE_COMMAND} -E env TT_METAL_RUNTIME_ROOT=${_TTMLIR_SOURCE_DIR}/third_party/tt-metal/src/tt-metal -- ${CMAKE_COMMAND} --build ${_TTMLIR_BUILD_DIR} ${_TTMLIR_TARGETS_TO_BUILD}"
      ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
      WORKING_DIRECTORY "${_TTMLIR_BUILD_DIR}"
  )

  message(STATUS "Installing tt-mlir...")
  ttlang_execute_with_env(
      COMMAND "${CMAKE_COMMAND} -E env TT_METAL_RUNTIME_ROOT=${_TTMLIR_SOURCE_DIR}/third_party/tt-metal/src/tt-metal -- ${CMAKE_COMMAND} --build ${_TTMLIR_BUILD_DIR} --target install"
      ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
      WORKING_DIRECTORY "${_TTMLIR_BUILD_DIR}"
  )

  # Install the SharedLib component (includes CMake config files)
  message(STATUS "Installing tt-mlir SharedLib component...")
  ttlang_execute_with_env(
      COMMAND "${CMAKE_COMMAND} -E env TT_METAL_RUNTIME_ROOT=${_TTMLIR_SOURCE_DIR}/third_party/tt-metal/src/tt-metal -- ${CMAKE_COMMAND} --install ${_TTMLIR_BUILD_DIR} --component SharedLib"
      ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
      WORKING_DIRECTORY "${_TTMLIR_BUILD_DIR}"
  )

  # Install the Test component to get ttmlir-opt and other tools installed.
  message(STATUS "Installing tt-mlir Test component...")
  ttlang_execute_with_env(
      COMMAND "${CMAKE_COMMAND} -E env TT_METAL_RUNTIME_ROOT=${_TTMLIR_SOURCE_DIR}/third_party/tt-metal/src/tt-metal -- ${CMAKE_COMMAND} --install ${_TTMLIR_BUILD_DIR} --component Test"
      ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
      WORKING_DIRECTORY "${_TTMLIR_BUILD_DIR}"
  )

  message(STATUS "Installing tt-mlir Python wheel...")
  ttlang_execute_with_env(
      COMMAND "${CMAKE_COMMAND} --install ${_TTMLIR_BUILD_DIR} --component TTMLIRPythonWheel"
      ENV_SCRIPT "${_TTMLIR_SOURCE_DIR}/env/activate"
      WORKING_DIRECTORY "${_TTMLIR_BUILD_DIR}"
  )

  # Fix RISC-V compiler libexec permissions
  # The installed compiler's internal executables (cc1plus, etc.) need execute permissions
  set(_RISCV_LIBEXEC_DIR "${_TTMLIR_INSTALL_PREFIX}/ttrt/runtime/runtime/sfpi/compiler/libexec")
  if(EXISTS "${_RISCV_LIBEXEC_DIR}")
    message(STATUS "Fixing RISC-V compiler libexec permissions...")
    file(GLOB_RECURSE _RISCV_LIBEXEC_FILES "${_RISCV_LIBEXEC_DIR}/*")
    foreach(_file IN LISTS _RISCV_LIBEXEC_FILES)
      if(NOT IS_DIRECTORY "${_file}")
        file(CHMOD "${_file}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
      endif()
    endforeach()
  endif()

  # Copy tt-metal soc_descriptors to expected location for runtime
  # This is needed because the runtime looks for these files relative to the tt-lang source directory
  set(_TT_METAL_SOC_DESCRIPTORS_SRC "${_TTMLIR_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/soc_descriptors")
  set(_TT_METAL_SOC_DESCRIPTORS_DST "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/soc_descriptors")
  if(EXISTS "${_TT_METAL_SOC_DESCRIPTORS_SRC}")
    message(STATUS "Copying tt-metal soc_descriptors from tt-mlir source to ${_TT_METAL_SOC_DESCRIPTORS_DST}...")
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal")
    file(COPY "${_TT_METAL_SOC_DESCRIPTORS_SRC}" DESTINATION "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal")
  else()
    message(WARNING "tt-metal soc_descriptors not found at ${_TT_METAL_SOC_DESCRIPTORS_SRC}, runtime may fail to find device descriptor files")
  endif()

  # Save original toolchain dir before finding the locally built tt-mlir.
  set(_ORIGINAL_TTMLIR_TOOLCHAIN_DIR "${TTMLIR_TOOLCHAIN_DIR}")

  message(STATUS "Searching for TTMLIRConfig.cmake in: ${_TTMLIR_INSTALL_PREFIX}/lib/cmake/ttmlir")
  find_package(TTMLIR REQUIRED CONFIG PATHS "${_TTMLIR_INSTALL_PREFIX}/lib/cmake/ttmlir")
  message(STATUS "Built and using private tt-mlir installation from: ${TTMLIR_CMAKE_DIR}")

  # Create aliases for namespaced targets so tt-lang can use unprefixed names.
  ttlang_create_ttmlir_aliases()

  if(EXISTS "${_TTMLIR_SOURCE_DIR}/cmake/modules")
    list(APPEND CMAKE_MODULE_PATH "${_TTMLIR_SOURCE_DIR}/cmake/modules")
    include(TTMLIRBuildTypes OPTIONAL)
  endif()

  set(TTMLIR_PATH "${_TTMLIR_INSTALL_PREFIX}")

  # For scenario 3, use the installed tt-mlir compiler static library
  set(TTMLIR_LINK_LIBS TTMLIR::TTMLIRCompilerStatic)

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
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    message(STATUS "Using Python: ${Python3_EXECUTABLE}")
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
  endif()
endif()

# Set TT_METAL_RUNTIME_ROOT for env/activate script
# Search order:
# 1. tt-mlir install directory (FetchContent build with installed tt-mlir)
# 2. tt-mlir source directory (build tree scenario)
# 3. tt-lang third_party directory (legacy/fallback)
if(EXISTS "${TTMLIR_PATH}/tt-metal/tt_metal")
  # Install scenario: tt-metal installed alongside tt-mlir (most common for FetchContent)
  set(TT_METAL_RUNTIME_ROOT "${TTMLIR_PATH}/tt-metal")
  message(STATUS "Found tt-metal runtime at: ${TT_METAL_RUNTIME_ROOT}")
elseif(EXISTS "${TTMLIR_PATH}/../third_party/tt-metal/src/tt-metal")
  # Build tree scenario: tt-mlir source tree contains tt-metal
  get_filename_component(TT_METAL_RUNTIME_ROOT "${TTMLIR_PATH}/../third_party/tt-metal/src/tt-metal" ABSOLUTE)
  message(STATUS "Found tt-metal runtime at: ${TT_METAL_RUNTIME_ROOT}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal")
  # Fallback: tt-lang third_party directory
  set(TT_METAL_RUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/tt-metal/src/tt-metal")
  message(STATUS "Found tt-metal runtime at: ${TT_METAL_RUNTIME_ROOT}")
else()
  set(TT_METAL_RUNTIME_ROOT "")
  message(WARNING "Could not find tt-metal runtime. Hardware tests may fail. You can set TT_METAL_RUNTIME_ROOT environment variable manually.")
endif()
