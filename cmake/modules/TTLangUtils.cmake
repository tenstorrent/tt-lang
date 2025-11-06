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
#   COMMAND <command_string>
#   ENV_SCRIPT <path_to_activate_script>
#   [WORKING_DIRECTORY <directory>]
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
