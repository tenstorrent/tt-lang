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

option(TTLANG_ENABLE_BINDINGS_PYTHON "Enable Python bindings" ON)
option(CODE_COVERAGE "Enable coverage reporting" OFF)
option(TTLANG_ENABLE_DOCS "Enable Sphinx documentation build" OFF)

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
if(TTLANG_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
else()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

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

if(TTLANG_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  set(TTLANG_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  add_subdirectory(python)

  ttlang_pip_install_requirements("${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/dev-requirements.txt")
endif()

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
