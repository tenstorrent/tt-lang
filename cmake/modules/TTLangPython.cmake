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
  foreach(_name python3 python)
    if(EXISTS "${venv_dir}/bin/${_name}")
      set(${out_var} "${venv_dir}/bin/${_name}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  # Fall back to versioned names (python3.X).
  file(GLOB _versioned "${venv_dir}/bin/python3.*")
  # Filter out paths containing ".py" or config suffixes.
  set(_filtered)
  foreach(_p ${_versioned})
    get_filename_component(_fname "${_p}" NAME)
    if(NOT _fname MATCHES "\\." OR _fname MATCHES "^python3\\.[0-9]+$")
      list(APPEND _filtered "${_p}")
    endif()
  endforeach()
  if(_filtered)
    list(GET _filtered 0 _first)
    set(${out_var} "${_first}" PARENT_SCOPE)
    return()
  endif()
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
endmacro()

# TTLANG_USE_TOOLCHAIN and TTLANG_TOOLCHAIN_DIR are declared in CMakeLists.txt.

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

# 1. Explicit user override.
if(DEFINED TTLANG_PYTHON_VENV)
  if(NOT EXISTS "${TTLANG_PYTHON_VENV}")
    message(FATAL_ERROR
      "TTLANG_PYTHON_VENV is set to '${TTLANG_PYTHON_VENV}' but the directory does not exist.")
  endif()
  _ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _TTLANG_VENV_PYTHON)
  if(NOT _TTLANG_VENV_PYTHON)
    message(FATAL_ERROR
      "TTLANG_PYTHON_VENV is set to '${TTLANG_PYTHON_VENV}' but no Python interpreter found in it.")
  endif()
  _ttlang_activate_venv("${TTLANG_PYTHON_VENV}")
  set(Python3_EXECUTABLE "${_TTLANG_VENV_PYTHON}" CACHE FILEPATH
    "Python interpreter (from user-specified venv)" FORCE)
  message(STATUS "Using user-specified Python venv: ${TTLANG_PYTHON_VENV}")
  message(STATUS "  Python: ${Python3_EXECUTABLE}")

# 2. Toolchain venv (when TTLANG_USE_TOOLCHAIN is ON or toolchain dir has a venv).
elseif(DEFINED TTLANG_TOOLCHAIN_DIR AND EXISTS "${TTLANG_TOOLCHAIN_DIR}/venv")
  set(TTLANG_PYTHON_VENV "${TTLANG_TOOLCHAIN_DIR}/venv" CACHE PATH
    "Python venv (from toolchain)" FORCE)
  _ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _TTLANG_VENV_PYTHON)
  if(NOT _TTLANG_VENV_PYTHON)
    message(FATAL_ERROR
      "Toolchain venv at '${TTLANG_PYTHON_VENV}' has no Python interpreter.\n"
      "The toolchain may be corrupted. Rebuild it or set TTLANG_PYTHON_VENV manually.")
  endif()
  _ttlang_activate_venv("${TTLANG_PYTHON_VENV}")
  set(Python3_EXECUTABLE "${_TTLANG_VENV_PYTHON}" CACHE FILEPATH
    "Python interpreter (from toolchain venv)" FORCE)
  message(STATUS "Using toolchain Python venv: ${TTLANG_PYTHON_VENV}")
  message(STATUS "  Python: ${Python3_EXECUTABLE}")

# 3. Local project venv (reuse existing or will be created during submodule build).
elseif(EXISTS "${CMAKE_BINARY_DIR}/venv")
  set(TTLANG_PYTHON_VENV "${CMAKE_BINARY_DIR}/venv" CACHE PATH
    "Python venv (local project)" FORCE)
  _ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _TTLANG_VENV_PYTHON)
  if(_TTLANG_VENV_PYTHON)
    _ttlang_activate_venv("${TTLANG_PYTHON_VENV}")
    set(Python3_EXECUTABLE "${_TTLANG_VENV_PYTHON}" CACHE FILEPATH
      "Python interpreter (from local project venv)" FORCE)
    message(STATUS "Using local project Python venv: ${TTLANG_PYTHON_VENV}")
    message(STATUS "  Python: ${Python3_EXECUTABLE}")
  endif()

# 4. No venv yet — will be created during submodule LLVM build if needed.
else()
  set(TTLANG_PYTHON_VENV "${CMAKE_BINARY_DIR}/venv" CACHE PATH
    "Python venv (will be created during LLVM build)" FORCE)
  message(STATUS "Python venv will be created at: ${TTLANG_PYTHON_VENV}")
endif()
