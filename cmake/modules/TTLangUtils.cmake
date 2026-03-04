# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# TTLangUtils.cmake - Utility macros and functions for tt-lang.

# ttlang_set_version(VERSION)
# Sets tt-lang version from a version string (e.g., "0.2.0" or "0.2.0.dev5").
# Sets TTLANG_VERSION_MAJOR, TTLANG_VERSION_MINOR, TTLANG_VERSION_PATCH cache variables
# and TTLANG_VERSION as the full version string (including dev suffix if present).
macro(ttlang_set_version VERSION)
  # Extract base version (MAJOR.MINOR.PATCH) from version string, handling .devX suffix
  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _match "${VERSION}")

  if(NOT _match)
    message(FATAL_ERROR "Invalid version format: ${VERSION}. Expected format: MAJOR.MINOR.PATCH[.devX] (e.g., 0.2.0 or 0.2.0.dev5)")
  endif()

  set(TTLANG_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE STRING "tt-lang major version")
  set(TTLANG_VERSION_MINOR ${CMAKE_MATCH_2} CACHE STRING "tt-lang minor version")
  set(TTLANG_VERSION_PATCH ${CMAKE_MATCH_3} CACHE STRING "tt-lang patch version")
  # Preserve the full version string including .devX suffix if present
  set(TTLANG_VERSION "${VERSION}")
  message(STATUS "tt-lang version: ${TTLANG_VERSION}")
endmacro()

# ttlang_check_device_available(OUTPUT_VAR)
# Checks if a Tenstorrent device is available at configure time by looking for
# /dev/tenstorrent* files. This is faster than calling ttnn.GetNumAvailableDevices().
# Sets the variable named by OUTPUT_VAR to TRUE if available, FALSE otherwise.
function(ttlang_check_device_available OUTPUT_VAR)
  if(DEFINED ENV{TT_METAL_SIMULATOR})
    set(${OUTPUT_VAR} TRUE PARENT_SCOPE)
    message(STATUS "Tenstorrent device: simulator mode (TT_METAL_SIMULATOR=$ENV{TT_METAL_SIMULATOR})")
    return()
  endif()
  file(GLOB _tt_device_files "/dev/tenstorrent*")
  if(_tt_device_files)
    set(${OUTPUT_VAR} TRUE PARENT_SCOPE)
    message(STATUS "Tenstorrent device detected")
  else()
    set(${OUTPUT_VAR} FALSE PARENT_SCOPE)
    message(STATUS "No Tenstorrent device detected")
  endif()
endfunction()

# ttlang_debug_message(MESSAGE)
# Prints a STATUS message only if TTLANG_CMAKE_DEBUG environment variable is defined.
# Useful for verbose debug output during CMake configuration.
macro(ttlang_debug_message MESSAGE)
  if(DEFINED ENV{TTLANG_CMAKE_DEBUG})
    message(STATUS "${MESSAGE}")
  endif()
endmacro()
