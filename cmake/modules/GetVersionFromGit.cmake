# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Get version from git tags (similar to LLVM's VersionFromVCS.cmake)

find_package(Git QUIET)

if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*" --abbrev=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(GIT_TAG)
    # Strip 'v' prefix and parse version (e.g., v0.1.0 -> 0.1.0)
    string(REGEX REPLACE "^v" "" TTLANG_VERSION "${GIT_TAG}")
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${TTLANG_VERSION}")
    set(TTLANG_VERSION_MAJOR "${CMAKE_MATCH_1}" PARENT_SCOPE)
    set(TTLANG_VERSION_MINOR "${CMAKE_MATCH_2}" PARENT_SCOPE)
    set(TTLANG_VERSION_PATCH "${CMAKE_MATCH_3}" PARENT_SCOPE)
    set(TTLANG_VERSION "${TTLANG_VERSION}" PARENT_SCOPE)

    # Get commit count since tag for dev builds
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list ${GIT_TAG}..HEAD --count
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE COMMITS_SINCE_TAG
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )

    if(COMMITS_SINCE_TAG AND NOT COMMITS_SINCE_TAG EQUAL "0")
      set(TTLANG_VERSION "${TTLANG_VERSION}.dev${COMMITS_SINCE_TAG}" PARENT_SCOPE)
    endif()
  else()
    # Fallback if no tags
    set(TTLANG_VERSION "0.2.0.dev0" PARENT_SCOPE)
  endif()
else()
  # No git, use default
  set(TTLANG_VERSION "0.2.0.dev0" PARENT_SCOPE)
endif()

message(STATUS "tt-lang version: ${TTLANG_VERSION}")
