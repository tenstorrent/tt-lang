# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# TTLangUtils.cmake - Utility macros and functions for tt-lang.

# ttlang_set_version(VERSION)
# Sets tt-lang version from a version triplet string (e.g., "0.2.0").
# Sets TTLANG_VERSION_MAJOR, TTLANG_VERSION_MINOR, TTLANG_VERSION_PATCH cache variables
# and TTLANG_VERSION as the full version string.
macro(ttlang_set_version VERSION)
  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" _match "${VERSION}")

  if(NOT _match)
    message(FATAL_ERROR "Invalid version format: ${VERSION}. Expected format: MAJOR.MINOR.PATCH (e.g., 0.2.0)")
  endif()

  set(TTLANG_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE STRING "tt-lang major version")
  set(TTLANG_VERSION_MINOR ${CMAKE_MATCH_2} CACHE STRING "tt-lang minor version")
  set(TTLANG_VERSION_PATCH ${CMAKE_MATCH_3} CACHE STRING "tt-lang patch version")
  set(TTLANG_VERSION "${TTLANG_VERSION_MAJOR}.${TTLANG_VERSION_MINOR}.${TTLANG_VERSION_PATCH}")
  message(STATUS "tt-lang version: ${TTLANG_VERSION}")
endmacro()

# ttlang_get_parent_dir(PATH LEVELS OUTPUT_VAR)
# Gets the parent directory N levels up from a given path.
# For example: PATH="/a/b/c/d", LEVELS=2 returns "/a/b".
function(ttlang_get_parent_dir PATH LEVELS OUTPUT_VAR)
  set(_current_path "${PATH}")

  foreach(_i RANGE 1 ${LEVELS})
    get_filename_component(_current_path "${_current_path}" DIRECTORY)
  endforeach()

  set(${OUTPUT_VAR} "${_current_path}" PARENT_SCOPE)
endfunction()

# ttlang_execute_with_env(
# COMMAND <command_string>
# ENV_SCRIPT <path_to_activate_script>
# [WORKING_DIRECTORY <directory>]
# )
# Executes a command with an environment activation script sourced.
# The command is run in a bash shell with the environment script sourced first.
# Output is echoed to stdout and errors are fatal.
function(ttlang_execute_with_env)
  set(options)
  set(oneValueArgs COMMAND ENV_SCRIPT WORKING_DIRECTORY)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_COMMAND)
    message(FATAL_ERROR "ttlang_execute_with_env: COMMAND is required")
  endif()

  if(NOT ARG_ENV_SCRIPT)
    message(FATAL_ERROR "ttlang_execute_with_env: ENV_SCRIPT is required")
  endif()

  set(_exec_args
    COMMAND_ECHO STDOUT
    COMMAND_ERROR_IS_FATAL ANY
  )

  if(ARG_WORKING_DIRECTORY)
    list(APPEND _exec_args WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}")
  endif()

  execute_process(
    COMMAND /bin/bash -c ". ${ARG_ENV_SCRIPT} && ${ARG_COMMAND}"
    ${_exec_args}
  )
endfunction()

# ttlang_collect_ttmlir_link_libs(OUTPUT_VAR)
# Collects all tt-mlir and MLIR libraries needed for linking when using a build tree.
# This includes:
# - TTMLIRCompilerStatic (contains RegisterAll.cpp)
# - All tt-mlir export targets from TTMLIRInstall.cmake
# - All MLIR dialect, conversion, extension, and translation libraries
# - Core MLIR libraries
# The collected libraries are stored in the variable named by OUTPUT_VAR.
macro(ttlang_collect_ttmlir_link_libs OUTPUT_VAR)
  # Get MLIR libraries from global properties
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
  get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

  # Use tt-mlir export targets variable (set by TTMLIRInstall.cmake)
  if(DEFINED ttmlir_export_targets_filtered)
    set(_ttmlir_targets ${ttmlir_export_targets_filtered})
  elseif(DEFINED ttmlir_export_targets)
    set(_ttmlir_targets ${ttmlir_export_targets})
  else()
    set(_ttmlir_targets "")
  endif()

  # Filter out targets that don't exist
  set(_ttmlir_targets_existing "")

  foreach(target ${_ttmlir_targets})
    if(TARGET ${target})
      list(APPEND _ttmlir_targets_existing ${target})
    endif()
  endforeach()


  # Some tt-mlir pipeline libraries (TTIR/TTNN) are not exported, so add them
  # explicitly when available.
  set(_ttmlir_pipeline_libs
    MLIRTTIRPipelines
    MLIRTTNNPipelines
  )
  set(_ttmlir_pipeline_libs_existing "")

  foreach(target ${_ttmlir_pipeline_libs})
    if(TARGET ${target})
      list(APPEND _ttmlir_pipeline_libs_existing ${target})
    endif()
  endforeach()

  set(${OUTPUT_VAR}
    TTMLIRCompilerStatic
    ${_ttmlir_targets_existing}
    ${_ttmlir_pipeline_libs_existing}
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ${translation_libs}
    MLIRToLLVMIRTranslationRegistration
    MLIRPass
    MLIRSupport
    MLIRRegisterAllPasses
  )
endmacro()

# ttlang_setup_ttmlir_build_tree(BUILD_DIR)
# Sets up tt-mlir from a build tree. This includes:
# - Loading build cache to get source directory and Python executable
# - Setting up include and link directories
# - Importing tt-mlir targets from the build tree
# - Including tt-mlir CMake modules (TTMLIRBuildTypes, TTMLIRInstall)
macro(ttlang_setup_ttmlir_build_tree BUILD_DIR)
  if(NOT EXISTS "${BUILD_DIR}/CMakeCache.txt")
    message(FATAL_ERROR "TTMLIR_BUILD_DIR points to an install directory, not a build directory. Please set TTMLIR_BUILD_DIR to the tt-mlir build directory (e.g., /path/to/tt-mlir/build), not the install directory.")
  endif()

  message(STATUS "Using pre-built tt-mlir from build tree: ${BUILD_DIR}")

  # Load build cache to get source directory and configuration
  load_cache("${BUILD_DIR}" READ_WITH_PREFIX _TTMLIR_
    CMAKE_HOME_DIRECTORY
    LLVM_DIR
    _Python3_EXECUTABLE
  )

  if(DEFINED _TTMLIR_TTMLIR_ENABLE_STABLEHLO AND _TTMLIR_TTMLIR_ENABLE_STABLEHLO)
    message(STATUS "tt-lang detected tt-mlir built with StableHLO support enabled.")
  endif()

  if(DEFINED _TTMLIR_TTMLIR_ENABLE_TNN_JIT AND NOT _TTMLIR_TTMLIR_ENABLE_TNN_JIT)
    message(WARNING "tt-lang recommends tt-mlir builds with -DTTMLIR_ENABLE_TNN_JIT=ON; continuing without TNN JIT support may disable certain pipelines.")
  endif()

  if(DEFINED _TTMLIR_LLVM_DIR)
    list(APPEND CMAKE_MODULE_PATH "${_TTMLIR_LLVM_DIR}/cmake")
  endif()

  if(DEFINED _TTMLIR__Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${_TTMLIR__Python3_EXECUTABLE}" CACHE FILEPATH "Python 3 executable from tt-mlir build" FORCE)
    message(STATUS "Using Python from tt-mlir build: ${Python3_EXECUTABLE}")

    # Only extract toolchain dir from Python if TTMLIR_TOOLCHAIN_DIR was not set from environment.
    if(NOT _TTMLIR_TOOLCHAIN_DIR_FROM_ENV)
      # Python is at /path/to/toolchain/venv/bin/python3, so go up 3 directories to get toolchain root
      ttlang_get_parent_dir("${Python3_EXECUTABLE}" 3 _TTMLIR_EXTRACTED_TOOLCHAIN_DIR)
      set(TTMLIR_TOOLCHAIN_DIR "${_TTMLIR_EXTRACTED_TOOLCHAIN_DIR}" CACHE PATH "tt-mlir toolchain installation directory" FORCE)
    endif()
  endif()

  if(DEFINED _TTMLIR_CMAKE_HOME_DIRECTORY)
    # Using a tt-mlir build tree
    set(_TTMLIR_SOURCE_DIR "${_TTMLIR_CMAKE_HOME_DIRECTORY}")

    if(EXISTS "${_TTMLIR_SOURCE_DIR}/cmake/modules")
      message(STATUS "Found tt-mlir source directory: ${_TTMLIR_SOURCE_DIR}")
      list(APPEND CMAKE_MODULE_PATH "${_TTMLIR_SOURCE_DIR}/cmake/modules")
      include(TTMLIRBuildTypes OPTIONAL)
      include(TTMLIRInstall OPTIONAL)
    endif()

    set(TTMLIR_PATH "${BUILD_DIR}")
    include_directories(${_TTMLIR_SOURCE_DIR}/include ${BUILD_DIR}/include)
    link_directories(${BUILD_DIR}/lib)

    # Import tt-mlir targets from build tree
    if(EXISTS "${BUILD_DIR}/lib/cmake/ttmlir/TTMLIRTargets.cmake")
      include("${BUILD_DIR}/lib/cmake/ttmlir/TTMLIRTargets.cmake")
    endif()
  endif()
endmacro()

# ttlang_check_ttnn_available(OUTPUT_VAR)
# Checks if the TTNN Python package is available at configure time.
# Sets the variable named by OUTPUT_VAR to TRUE if available, FALSE otherwise.
function(ttlang_check_ttnn_available OUTPUT_VAR)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import ttnn"
    RESULT_VARIABLE _ttnn_import_result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(_ttnn_import_result EQUAL 0)
    set(${OUTPUT_VAR} TRUE PARENT_SCOPE)
    message(STATUS "TTNN Python package available")
  else()
    set(${OUTPUT_VAR} FALSE PARENT_SCOPE)
    message(STATUS "TTNN Python package not available")
  endif()
endfunction()

# ttlang_check_device_available(OUTPUT_VAR)
# Checks if a Tenstorrent device is available at configure time by looking for
# /dev/tenstorrent* files. This is faster than calling ttnn.GetNumAvailableDevices().
# Sets the variable named by OUTPUT_VAR to TRUE if available, FALSE otherwise.
function(ttlang_check_device_available OUTPUT_VAR)
  file(GLOB _tt_device_files "/dev/tenstorrent*")
  if(_tt_device_files)
    set(${OUTPUT_VAR} TRUE PARENT_SCOPE)
    message(STATUS "Tenstorrent device detected")
  else()
    set(${OUTPUT_VAR} FALSE PARENT_SCOPE)
    message(STATUS "No Tenstorrent device detected")
  endif()
endfunction()
