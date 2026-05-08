# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Get version from git tags (similar to LLVM's VersionFromVCS.cmake)

find_package(Git QUIET)

# Fallback used when git is unavailable or no v* tag is reachable.
# Deliberately not a plausible-looking release version.
set(_TTLANG_VERSION_FALLBACK "0.0.0+unknown")

if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*" --abbrev=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE GIT_DESCRIBE_ERR
  )

  if(GIT_TAG)
    # Strip 'v' prefix; split MAJOR.MINOR.PATCH from optional '+local' SemVer
    # build metadata (e.g., v1.0.0+uplift -> base=1.0.0, local=+uplift).
    # PEP 440 requires <release>[.devN][+local]; the local segment must come
    # last so the dev counter is inserted before '+', not appended after.
    string(REGEX REPLACE "^v" "" _bare "${GIT_TAG}")
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)([+].*)?$" _match "${_bare}")
    if(NOT _match)
      message(FATAL_ERROR
        "Could not parse git tag '${GIT_TAG}'. Expected vMAJOR.MINOR.PATCH[+LOCAL].")
    endif()
    set(TTLANG_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(TTLANG_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(TTLANG_VERSION_PATCH "${CMAKE_MATCH_3}")
    set(_local "${CMAKE_MATCH_4}")
    set(_base "${TTLANG_VERSION_MAJOR}.${TTLANG_VERSION_MINOR}.${TTLANG_VERSION_PATCH}")

    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list ${GIT_TAG}..HEAD --count
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE COMMITS_SINCE_TAG
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(COMMITS_SINCE_TAG AND NOT COMMITS_SINCE_TAG EQUAL "0")
      set(TTLANG_VERSION "${_base}.dev${COMMITS_SINCE_TAG}${_local}")
    else()
      set(TTLANG_VERSION "${_base}${_local}")
    endif()
  else()
    message(WARNING
      "tt-lang version detection: git describe found no v* tag reachable "
      "from HEAD in ${CMAKE_SOURCE_DIR}. Using fallback "
      "'${_TTLANG_VERSION_FALLBACK}'.\n"
      "Likely causes and fixes:\n"
      "  - Tags not fetched: run `git fetch --tags origin`.\n"
      "  - Shallow clone with no tags in history: run "
      "`git fetch --unshallow --tags origin`.\n"
      "  - HEAD predates the first v* tag: rebase or merge a branch that "
      "includes one (e.g. `git merge origin/main`).\n"
      "git stderr: ${GIT_DESCRIBE_ERR}")
    set(TTLANG_VERSION "${_TTLANG_VERSION_FALLBACK}")
  endif()
else()
  message(WARNING
    "tt-lang version detection: git executable not found. Using fallback "
    "'${_TTLANG_VERSION_FALLBACK}'.\n"
    "Install git (e.g. `apt-get install git`) and re-run cmake to embed an "
    "accurate version.")
  set(TTLANG_VERSION "${_TTLANG_VERSION_FALLBACK}")
endif()

message(STATUS "tt-lang version: ${TTLANG_VERSION}")
