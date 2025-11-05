# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# TTLangUtils.cmake - Utility macros and functions for tt-lang

# ttlang_set_version(VERSION)
# Sets tt-lang version from a version triplet string (e.g., "0.2.0")
# Sets TTLANG_VERSION_MAJOR, TTLANG_VERSION_MINOR, TTLANG_VERSION_PATCH cache variables
# and TTLANG_VERSION as the full version string
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

