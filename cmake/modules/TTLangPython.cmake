# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# TTLangPython.cmake - Python venv setup for tt-lang.
#
# Locates or creates a Python virtual environment with the packages needed
# by the MLIR Python bindings (nanobind, numpy, etc.) and by tt-lang itself.
#
# Search order for the venv:
#   1. Explicit -DTTLANG_PYTHON_VENV=<path>  (user override, always wins)
#   2. Toolchain venv at ${TTLANG_TOOLCHAIN_DIR}/venv  (toolchain mode)
#   3. Local project venv at ${CMAKE_BINARY_DIR}/venv  (submodule build mode)
#
# After this module runs, the following are set:
#   TTLANG_PYTHON_VENV  - absolute path to the venv directory
#   Python3_FIND_VIRTUALENV, Python_FIND_VIRTUALENV, ENV{VIRTUAL_ENV}
#     - configured so that downstream find_package(Python3) uses the venv

# ---------------------------------------------------------------------------
# Helper: find the Python interpreter inside a venv directory.
# Sets ${out_var} to the path if found, empty string otherwise.
# ---------------------------------------------------------------------------
function(_ttlang_find_venv_python venv_dir out_var)
  # Collect candidate interpreter paths in priority order.
  set(_candidates)
  foreach(_name python3 python)
    if(EXISTS "${venv_dir}/bin/${_name}")
      list(APPEND _candidates "${venv_dir}/bin/${_name}")
    endif()
  endforeach()
  # Fall back to versioned names (python3.X).
  file(GLOB _versioned "${venv_dir}/bin/python3.*")
  foreach(_p ${_versioned})
    get_filename_component(_fname "${_p}" NAME)
    if(NOT _fname MATCHES "\\." OR _fname MATCHES "^python3\\.[0-9]+$")
      list(APPEND _candidates "${_p}")
    endif()
  endforeach()

  # Verify each candidate is actually executable (not a dangling symlink).
  foreach(_cand ${_candidates})
    execute_process(
      COMMAND "${_cand}" --version
      RESULT_VARIABLE _rc
      OUTPUT_QUIET ERROR_QUIET
    )
    if(_rc EQUAL 0)
      set(${out_var} "${_cand}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(${out_var} "" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Helper: activate a venv for find_package(Python3) by setting VIRTUAL_ENV
# and Python3_FIND_VIRTUALENV.
# ---------------------------------------------------------------------------
macro(_ttlang_activate_venv venv_dir)
  set(ENV{VIRTUAL_ENV} "${venv_dir}")
  set(Python3_FIND_VIRTUALENV ONLY)
  set(Python_FIND_VIRTUALENV ONLY)
  # Unset Python3_ROOT_DIR from the environment so it does not override
  # the venv.  GitHub Actions' setup-python sets this to the runner's
  # system Python, which causes find_package(Python3) to ignore the venv.
  unset(ENV{Python3_ROOT_DIR})
endmacro()

# TTLANG_USE_TOOLCHAIN and TTLANG_TOOLCHAIN_DIR are declared in CMakeLists.txt.

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

if(DEFINED TTLANG_PYTHON_VENV)
  set(_TTLANG_VENV_SOURCE "specified")
elseif(DEFINED TTLANG_TOOLCHAIN_DIR AND EXISTS "${TTLANG_TOOLCHAIN_DIR}/venv")
  set(TTLANG_PYTHON_VENV "${TTLANG_TOOLCHAIN_DIR}/venv" CACHE PATH
    "Python venv (from toolchain)" FORCE)
  set(_TTLANG_VENV_SOURCE "toolchain")
elseif(EXISTS "${CMAKE_BINARY_DIR}/venv")
  set(TTLANG_PYTHON_VENV "${CMAKE_BINARY_DIR}/venv" CACHE PATH
    "Python venv (local project)" FORCE)
  set(_TTLANG_VENV_SOURCE "local project")
elseif(DEFINED TTLANG_TOOLCHAIN_DIR)
  set(TTLANG_PYTHON_VENV "${TTLANG_TOOLCHAIN_DIR}/venv" CACHE PATH
    "Python venv (created in toolchain dir)" FORCE)
  set(_TTLANG_VENV_SOURCE "toolchain")
else()
  set(TTLANG_PYTHON_VENV "${CMAKE_BINARY_DIR}/venv" CACHE PATH
    "Python venv (created in build dir)" FORCE)
  set(_TTLANG_VENV_SOURCE "local project")
endif()

_ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _TTLANG_VENV_PYTHON)

if(NOT _TTLANG_VENV_PYTHON)
  message(STATUS "Creating Python venv at ${TTLANG_PYTHON_VENV}...")

  get_filename_component(_TTLANG_VENV_PARENT "${TTLANG_PYTHON_VENV}" DIRECTORY)
  file(MAKE_DIRECTORY "${_TTLANG_VENV_PARENT}")

  set(_TTLANG_BOOTSTRAP_PYTHON "")
  if(DEFINED Python3_EXECUTABLE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" --version
      RESULT_VARIABLE _TTLANG_BOOTSTRAP_RESULT
      OUTPUT_QUIET ERROR_QUIET
    )
    if(_TTLANG_BOOTSTRAP_RESULT EQUAL 0)
      set(_TTLANG_BOOTSTRAP_PYTHON "${Python3_EXECUTABLE}")
    endif()
  endif()

  if(NOT _TTLANG_BOOTSTRAP_PYTHON)
    unset(Python3_EXECUTABLE CACHE)
    unset(_Python3_EXECUTABLE CACHE)
    unset(Python3_EXECUTABLE)
    unset(ENV{VIRTUAL_ENV})
    set(Python3_FIND_VIRTUALENV STANDARD)

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_TTLANG_BOOTSTRAP_PYTHON "${Python3_EXECUTABLE}")
  endif()

  execute_process(
    COMMAND "${_TTLANG_BOOTSTRAP_PYTHON}" -m venv --prompt ttlang "${TTLANG_PYTHON_VENV}"
    RESULT_VARIABLE _TTLANG_VENV_RESULT
  )
  if(NOT _TTLANG_VENV_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to create Python venv at ${TTLANG_PYTHON_VENV}")
  endif()

  _ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _TTLANG_VENV_PYTHON)
  if(NOT _TTLANG_VENV_PYTHON)
    message(FATAL_ERROR
      "Created Python venv at '${TTLANG_PYTHON_VENV}', but no working Python interpreter was found.")
  endif()

  execute_process(
    COMMAND "${_TTLANG_VENV_PYTHON}" -m pip install --upgrade pip --quiet
  )
endif()

_ttlang_activate_venv("${TTLANG_PYTHON_VENV}")
set(Python3_EXECUTABLE "${_TTLANG_VENV_PYTHON}" CACHE FILEPATH
  "Python interpreter (from ${_TTLANG_VENV_SOURCE} venv)" FORCE)
message(STATUS "Using ${_TTLANG_VENV_SOURCE} Python venv: ${TTLANG_PYTHON_VENV}")
message(STATUS "  Python: ${Python3_EXECUTABLE}")
