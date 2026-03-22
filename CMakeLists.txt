cmake_minimum_required(VERSION 3.28.0)

if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER clang CACHE STRING "C Compiler" FORCE)
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER clang++ CACHE STRING "C++ Compiler" FORCE)
endif()

project(tt-lang LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
include(TTLangUtils)

# Get version from git tags
include(GetVersionFromGit)
ttlang_set_version("${TTLANG_VERSION}")

option(CODE_COVERAGE "Enable coverage reporting" OFF)
option(TTLANG_ENABLE_DOCS "Enable Sphinx documentation build" OFF)
option(TTLANG_SIM_ONLY "Set up Python environment for simulator only (skip compiler build)" OFF)

if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
endif()

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(TTLANG_BUILD_DEBUG TTLANG_ENABLE_DEBUG_LOGS)
endif()

include(TTLangCompilerSetup)

# ---------------------------------------------------------------------------
# Toolchain options (declared early because TTLangPython needs them).
# ---------------------------------------------------------------------------
option(TTLANG_USE_TOOLCHAIN "Use pre-built LLVM from ttlang toolchain" OFF)
if(NOT DEFINED TTLANG_TOOLCHAIN_DIR)
  if(DEFINED ENV{TTLANG_TOOLCHAIN_DIR})
    set(TTLANG_TOOLCHAIN_DIR "$ENV{TTLANG_TOOLCHAIN_DIR}" CACHE PATH
      "tt-lang toolchain directory")
  elseif(TTLANG_USE_TOOLCHAIN)
    set(TTLANG_TOOLCHAIN_DIR "/opt/ttlang-toolchain" CACHE PATH
      "tt-lang toolchain directory" FORCE)
  endif()
endif()

# ---------------------------------------------------------------------------
# Python venv setup (must run before BuildLLVM so that find_package(Python3)
# resolves against the venv, not a system Python).
# ---------------------------------------------------------------------------
include(TTLangPython)

# ---------------------------------------------------------------------------
# Simulator-only mode: create venv, install requirements, generate activate
# script, then stop.  No compiler, LLVM, or tt-metal build.
# ---------------------------------------------------------------------------
if(TTLANG_SIM_ONLY)
  # Create venv if it doesn't exist
  _ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _VENV_PYTHON)
  if(NOT _VENV_PYTHON)
    message(STATUS "Creating Python venv at ${TTLANG_PYTHON_VENV}...")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -m venv --prompt ttlang "${TTLANG_PYTHON_VENV}"
      RESULT_VARIABLE _VENV_RESULT
    )
    if(NOT _VENV_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to create Python venv")
    endif()
    _ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _VENV_PYTHON)
    _ttlang_activate_venv("${TTLANG_PYTHON_VENV}")
    execute_process(
      COMMAND "${_VENV_PYTHON}" -m pip install --upgrade pip --quiet
    )
  endif()
  set(Python3_EXECUTABLE "${_VENV_PYTHON}")

  # Install runtime requirements (torch, greenlet, pydantic, etc.)
  ttlang_pip_install_requirements("${_VENV_PYTHON}"
    "${CMAKE_SOURCE_DIR}/requirements.txt" FATAL)

  # Set variables needed by env/activate.in to harmless defaults
  set(TT_LANG_HOME "${CMAKE_CURRENT_LIST_DIR}")
  set(TTLANG_HAS_DEVICE_INT 0)
  set(LLVM_INSTALL_DIR "")
  set(TT_METAL_HOME "")
  set(TT_METAL_PYTHON_PATH "")
  set(TT_METAL_LIB_PATH "")

  configure_file(
    "${PROJECT_SOURCE_DIR}/env/activate.in"
    "${PROJECT_BINARY_DIR}/env/activate"
    @ONLY
  )

  message(STATUS "")
  message(STATUS "=== Simulator-only mode ===")
  message(STATUS "  Python venv:  ${TTLANG_PYTHON_VENV}")
  message(STATUS "  Activate:     source ${PROJECT_BINARY_DIR}/env/activate")
  message(STATUS "  Run:          ./bin/ttlang-sim examples/eltwise_add.py")
  message(STATUS "===")

  if(TTLANG_ENABLE_DOCS)
    add_subdirectory(docs/sphinx)
  endif()

  return()
endif()

# ---------------------------------------------------------------------------
# LLVM/MLIR dependency (pre-built or from submodule)
# ---------------------------------------------------------------------------
include(BuildLLVM)

# LLVM/MLIR include dirs must be set before any subdirectory that uses TableGen.
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# Minimal tt-mlir build (from submodule)
# ---------------------------------------------------------------------------
include(BuildTTMLIRMinimal)

# ---------------------------------------------------------------------------
# tt-metal runtime (from submodule)
# ---------------------------------------------------------------------------
include(BuildTTMetal)

# Detect Python dev packages (Python3, pybind11, nanobind) using MLIR's
# helpers.  This must run after BuildLLVM (which sets Python3_EXECUTABLE to the
# venv when building from submodule) and before add_subdirectory(python).
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

set(TTLANG_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TTLANG_BINARY_DIR ${PROJECT_BINARY_DIR})
set(TT_LANG_HOME "${CMAKE_CURRENT_LIST_DIR}")
set(LLVM_TOOLS_DIR "${LLVM_INSTALL_DIR}/bin")

ttlang_check_device_available(TTLANG_HAS_DEVICE)
if(TTLANG_HAS_DEVICE)
  set(TTLANG_HAS_DEVICE_INT 1)
else()
  set(TTLANG_HAS_DEVICE_INT 0)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/include/ttlang/Config.h.in"
  "${PROJECT_BINARY_DIR}/include/ttlang/Config.h"
  @ONLY
)

configure_file(
  "${PROJECT_SOURCE_DIR}/env/activate.in"
  "${PROJECT_BINARY_DIR}/env/activate"
  @ONLY
)

include_directories(${TTLANG_SOURCE_DIR}/include)
include_directories(${TTLANG_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

set(TTLANG_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
add_subdirectory(python)

ttlang_pip_install_requirements("${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/dev-requirements.txt")

add_subdirectory(test)

if(TTLANG_ENABLE_DOCS)
  add_subdirectory(docs/sphinx)
endif()

# Install wheel files to prefix/wheels
file(GLOB_RECURSE WHEEL_FILES "${CMAKE_BINARY_DIR}/*.whl")
if(WHEEL_FILES)
  install(FILES ${WHEEL_FILES} DESTINATION wheels)
endif()

# Install examples, tests, and environment activation script for Docker images
install(DIRECTORY examples/ DESTINATION examples
        PATTERN "*.pyc" EXCLUDE
        PATTERN "__pycache__" EXCLUDE)

install(DIRECTORY test/ DESTINATION test
        PATTERN "*.pyc" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN "build" EXCLUDE)

# Install environment activation script for installed location
install(PROGRAMS "${CMAKE_SOURCE_DIR}/.github/containers/activate-install.sh"
        DESTINATION env
        RENAME activate)

# Install pure Python packages (sim, pykernel) - simple copy
install(DIRECTORY python/sim python/pykernel DESTINATION python_packages
        FILES_MATCHING PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE)

# Install ttlang-sim script to bin directory
install(PROGRAMS "${CMAKE_SOURCE_DIR}/bin/ttlang-sim" DESTINATION bin)

# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== tt-lang configuration summary ===")
message(STATUS "  Version:          ${TTLANG_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build dir:        ${CMAKE_BINARY_DIR}")
if(DEFINED TTLANG_TOOLCHAIN_DIR)
  message(STATUS "  Toolchain dir:    ${TTLANG_TOOLCHAIN_DIR}")
  if(TTLANG_USE_TOOLCHAIN)
    message(STATUS "  Toolchain mode:   use (pre-built)")
  else()
    message(STATUS "  Toolchain mode:   build (from submodule)")
  endif()
endif()
message(STATUS "  LLVM install:     ${LLVM_INSTALL_DIR}")
message(STATUS "  tt-metal:         ${TT_METAL_HOME}")
message(STATUS "  Python:           ${Python3_EXECUTABLE}")
message(STATUS "  Python venv:      ${TTLANG_PYTHON_VENV}")
message(STATUS "  Perf tracing:     ${TTLANG_ENABLE_PERF_TRACE}")
message(STATUS "  Device detected:  ${TTLANG_HAS_DEVICE}")
message(STATUS "")
message(STATUS "  Activate env:     source ${CMAKE_BINARY_DIR}/env/activate")
message(STATUS "===")
