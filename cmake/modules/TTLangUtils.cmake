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

# ttlang_ensure_submodules(SUBMODULES...)
# Initializes git submodules if not already present.
# Uses --depth 1 for a shallow clone of just the pinned commit.
# GitHub supports allowReachableSHA1InWant, so --depth 1 works even for
# commits that are not at a branch tip.
# Skipped when there is no .git directory (e.g. Docker build context).
function(ttlang_ensure_submodules)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
    return()
  endif()
  foreach(_sub IN LISTS ARGN)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${_sub}/.git")
      message(STATUS "Initializing submodule ${_sub}...")
      execute_process(
        COMMAND git submodule update --init --depth 1 "${_sub}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE _sub_result
      )
      if(NOT _sub_result EQUAL 0)
        message(FATAL_ERROR
          "Failed to initialize submodule ${_sub}.\n"
          "Run manually:\n"
          "  git submodule update --init --depth 1 ${_sub}")
      endif()
    endif()
  endforeach()
endfunction()

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

# ttlang_pip_install_requirements(PYTHON_EXE REQUIREMENTS_FILE [FATAL])
# Installs Python packages from a requirements file using pip.
# If FATAL is specified, a failure is a fatal error; otherwise it is a warning.
function(ttlang_pip_install_requirements PYTHON_EXE REQUIREMENTS_FILE)
  if(NOT EXISTS "${REQUIREMENTS_FILE}")
    message(WARNING "Requirements file not found: ${REQUIREMENTS_FILE}")
    return()
  endif()
  cmake_path(RELATIVE_PATH REQUIREMENTS_FILE BASE_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_VARIABLE _req_rel)
  message(STATUS "Installing Python requirements from ${_req_rel}...")
  execute_process(
    COMMAND "${PYTHON_EXE}" -m pip install --quiet -r "${REQUIREMENTS_FILE}"
    RESULT_VARIABLE _pip_result
  )
  if(NOT _pip_result EQUAL 0)
    if("FATAL" IN_LIST ARGN)
      message(FATAL_ERROR "Failed to install Python requirements from ${REQUIREMENTS_FILE}")
    else()
      message(WARNING "Failed to install Python requirements from ${REQUIREMENTS_FILE}")
    endif()
  endif()
endfunction()

# ttlang_pip_install_package(PYTHON_EXE PACKAGE_PATH [FATAL])
# Installs a Python package from a local path using pip.
# If FATAL is specified, a failure is a fatal error; otherwise it is a warning.
function(ttlang_pip_install_package PYTHON_EXE PACKAGE_PATH)
  if(NOT EXISTS "${PACKAGE_PATH}")
    message(WARNING "Package path not found: ${PACKAGE_PATH}")
    return()
  endif()
  execute_process(
    COMMAND "${PYTHON_EXE}" -m pip install "${PACKAGE_PATH}" --no-build-isolation --quiet
    RESULT_VARIABLE _pip_result
  )
  if(NOT _pip_result EQUAL 0)
    if("FATAL" IN_LIST ARGN)
      message(FATAL_ERROR "Failed to pip-install package from ${PACKAGE_PATH}")
    else()
      message(WARNING "Failed to pip-install package from ${PACKAGE_PATH}")
    endif()
  endif()
endfunction()

# ttlang_get_submodule_sha(SUBMODULE_DIR OUTPUT_VAR)
# Retrieves the HEAD commit SHA of a git submodule. Sets OUTPUT_VAR to
# "unknown" if git fails (e.g. missing .git, dubious ownership).
function(ttlang_get_submodule_sha SUBMODULE_DIR OUTPUT_VAR)
  execute_process(
    COMMAND git -C "${SUBMODULE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE _sha
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _result
  )
  if(NOT _result EQUAL 0)
    set(_sha "unknown")
  endif()
  set(${OUTPUT_VAR} "${_sha}" PARENT_SCOPE)
endfunction()

# ttlang_debug_message(MESSAGE)
# Prints a STATUS message only if TTLANG_CMAKE_DEBUG environment variable is defined.
# Useful for verbose debug output during CMake configuration.
macro(ttlang_debug_message MESSAGE)
  if(DEFINED ENV{TTLANG_CMAKE_DEBUG})
    message(STATUS "${MESSAGE}")
  endif()
endmacro()

# ttlang_verify_llvm_sha(INSTALL_PREFIX EXPECTED_SHA)
# Verifies that the LLVM installation at INSTALL_PREFIX was built from the
# expected commit. Reads the SHA from VCSRevision.h and compares against
# EXPECTED_SHA. On mismatch, emits FATAL_ERROR unless the user passes
# -DTTLANG_ACCEPT_LLVM_MISMATCH=ON to explicitly accept the risk.
function(ttlang_verify_llvm_sha INSTALL_PREFIX EXPECTED_SHA)
  set(_vcs_header "${INSTALL_PREFIX}/include/llvm/Support/VCSRevision.h")

  if(NOT EXISTS "${_vcs_header}")
    message(WARNING
      "Cannot verify LLVM commit: ${_vcs_header} not found.\n"
      "SHA verification skipped.")
    return()
  endif()

  file(STRINGS "${_vcs_header}" _vcs_lines
       REGEX "#define LLVM_REVISION")

  if(NOT _vcs_lines)
    message(WARNING
      "Cannot verify LLVM commit: LLVM_REVISION not found in ${_vcs_header}.\n"
      "SHA verification skipped.")
    return()
  endif()

  # Extract the SHA from: #define LLVM_REVISION "abc123..."
  # Also handles C++ raw string literals: R"(abc123...)"
  # CMake regex has no {n,m} quantifier, so spell out 7+ hex chars.
  string(REGEX MATCH "([a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9]+)" _match "${_vcs_lines}")
  if(NOT _match)
    message(WARNING
      "Cannot parse LLVM_REVISION from ${_vcs_header}.\n"
      "SHA verification skipped.")
    return()
  endif()
  set(_actual_sha "${CMAKE_MATCH_1}")

  execute_process(
    COMMAND "${CMAKE_SOURCE_DIR}/scripts/verify-sha.sh"
            "${EXPECTED_SHA}" "${_actual_sha}"
    RESULT_VARIABLE _sha_cmp)

  if(_sha_cmp EQUAL 0)
    message(STATUS "LLVM SHA verified: ${_actual_sha}")
    return()
  endif()

  option(TTLANG_ACCEPT_LLVM_MISMATCH
    "Accept LLVM SHA mismatch (use at your own risk)" OFF)

  message(AUTHOR_WARNING
    "LLVM SHA mismatch!\n"
    "  Expected (submodule): ${EXPECTED_SHA}\n"
    "  Actual (installed):   ${_actual_sha}\n"
    "  Install prefix:       ${INSTALL_PREFIX}\n"
    "The installed LLVM differs from what the tt-lang submodule pins.\n"
    "This is usually fine if you built the toolchain yourself.")
endfunction()

# ttlang_verify_ttmetal_sha(SUBMODULE_DIR EXPECTED_SHA)
# Verifies that the tt-metal submodule at SUBMODULE_DIR is checked out at the
# expected commit SHA. The expected SHA is read from tt-mlir's third_party
# CMakeLists.txt (TT_METAL_VERSION). On mismatch, emits FATAL_ERROR unless the
# user passes -DTTLANG_ACCEPT_TTMETAL_MISMATCH=ON.
function(ttlang_verify_ttmetal_sha SUBMODULE_DIR EXPECTED_SHA)
  ttlang_get_submodule_sha("${SUBMODULE_DIR}" _actual_sha)

  if(_actual_sha STREQUAL "unknown")
    message(WARNING
      "Cannot verify tt-metal commit: git rev-parse failed in ${SUBMODULE_DIR}.\n"
      "SHA verification skipped.")
    return()
  endif()

  execute_process(
    COMMAND "${CMAKE_SOURCE_DIR}/scripts/verify-sha.sh"
            "${EXPECTED_SHA}" "${_actual_sha}"
    RESULT_VARIABLE _sha_cmp)

  if(_sha_cmp EQUAL 0)
    ttlang_debug_message("tt-metal SHA verified: ${_actual_sha}")
    return()
  endif()

  option(TTLANG_ACCEPT_TTMETAL_MISMATCH
    "Accept tt-metal SHA mismatch (use at your own risk)" OFF)

  message(AUTHOR_WARNING
    "tt-metal SHA mismatch!\n"
    "  Expected (tt-mlir pins): ${EXPECTED_SHA}\n"
    "  Actual (submodule):      ${_actual_sha}\n"
    "  Submodule path:          ${SUBMODULE_DIR}\n"
    "Using a mismatched tt-metal may cause JIT compile failures or runtime errors.\n"
    "To update: cd ${SUBMODULE_DIR} && git fetch --unshallow && git fetch origin ${EXPECTED_SHA} && git checkout ${EXPECTED_SHA}")

  if(NOT TTLANG_ACCEPT_TTMETAL_MISMATCH)
    message(FATAL_ERROR
      "tt-metal SHA mismatch. To proceed despite this, re-run with:\n"
      "  -DTTLANG_ACCEPT_TTMETAL_MISMATCH=ON")
  endif()
endfunction()

# ttlang_apply_patches(SOURCE_DIR PATCHES_GLOB)
# Applies git patches matching PATCHES_GLOB to SOURCE_DIR.
# Skips patches that are already applied (checked via git apply --reverse --check).
function(ttlang_apply_patches SOURCE_DIR PATCHES_GLOB)
  file(GLOB _patches "${PATCHES_GLOB}")
  foreach(_patch ${_patches})
    # Skip if already applied.
    execute_process(
      COMMAND git -C "${SOURCE_DIR}" apply --check --reverse "${_patch}"
      RESULT_VARIABLE _already_applied
      OUTPUT_QUIET ERROR_QUIET
    )
    if(_already_applied EQUAL 0)
      continue()
    endif()
    get_filename_component(_name "${_patch}" NAME)
    message(STATUS "Applying patch: ${_name}")
    execute_process(
      COMMAND git -C "${SOURCE_DIR}" apply "${_patch}"
      RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
      message(WARNING "Failed to apply patch: ${_name}")
    endif()
  endforeach()
endfunction()
