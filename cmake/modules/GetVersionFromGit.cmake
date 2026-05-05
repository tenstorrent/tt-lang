# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Get version from git tags (similar to LLVM's VersionFromVCS.cmake)

find_package(Git QUIET)

set(_TTLANG_VERSION_FALLBACK "0.0.0.dev0")
set(TTLANG_VERSION "${_TTLANG_VERSION_FALLBACK}")
set(_describe_failure_reason "")

if(GIT_FOUND)
  # Auto-mark the source tree as a safe.directory for git when the
  # source is owned by a different uid than the user running cmake
  # (typical for a host-mounted repo inside a docker container).
  # Without this, git refuses with "fatal: detected dubious ownership"
  # and cmake silently falls back to a placeholder version.
  if(UNIX)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} config --global --get-all safe.directory
      OUTPUT_VARIABLE _safe_dirs
      ERROR_QUIET
    )
    string(FIND "${_safe_dirs}" "${CMAKE_SOURCE_DIR}" _safe_match)
    if(_safe_match EQUAL -1)
      execute_process(
        COMMAND ${GIT_EXECUTABLE} config --global --add
                safe.directory ${CMAKE_SOURCE_DIR}
        ERROR_QUIET
      )
    endif()
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*" --abbrev=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_TAG
    ERROR_VARIABLE GIT_TAG_ERR
    RESULT_VARIABLE GIT_TAG_RC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )

  if(NOT GIT_TAG_RC EQUAL 0 OR NOT GIT_TAG)
    if(GIT_TAG_ERR)
      set(_describe_failure_reason
          "git describe failed: ${GIT_TAG_ERR}")
    else()
      set(_describe_failure_reason
          "git describe found no v[0-9]* tag in this repository")
    endif()
  else()
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

    # Get commit count since tag for dev builds
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list ${GIT_TAG}..HEAD --count
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE COMMITS_SINCE_TAG
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )

    if(COMMITS_SINCE_TAG AND NOT COMMITS_SINCE_TAG EQUAL "0")
      set(TTLANG_VERSION "${_base}.dev${COMMITS_SINCE_TAG}${_local}")
    else()
      set(TTLANG_VERSION "${_base}${_local}")
    endif()
  endif()
else()
  set(_describe_failure_reason "git executable not found on PATH")
endif()

if(_describe_failure_reason)
  # Surface the actual failure rather than silently stamping a
  # placeholder. Anyone debugging "why is the version wrong?" sees the
  # root cause in the configure log.
  message(WARNING
    "tt-lang version: could not derive from git tag — using fallback "
    "'${_TTLANG_VERSION_FALLBACK}'.\n"
    "  Reason: ${_describe_failure_reason}\n"
    "  Fix: ensure git tags are fetched (e.g. `git fetch --tags`) and "
    "that the source tree is readable by git (mounted repos inside "
    "containers may need `git config --global --add safe.directory "
    "${CMAKE_SOURCE_DIR}`).")
endif()

message(STATUS "tt-lang version: ${TTLANG_VERSION}")
