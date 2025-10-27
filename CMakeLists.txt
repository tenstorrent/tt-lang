cmake_minimum_required(VERSION 3.24.0)

if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER clang CACHE STRING "C Compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER clang++ CACHE STRING "C++ Compiler" FORCE)
endif()

project(tt-lang LANGUAGES CXX C)

# tt-lang reuses tt-mlir's environment
if (NOT DEFINED ENV{TTMLIR_ENV_ACTIVATED})
  message(FATAL_ERROR "tt-mlir environment not activated. Please run 'source env/activate' from tt-lang directory (which sources tt-mlir's environment).")
endif()

# Build options
option(TTLANG_ENABLE_BINDINGS_PYTHON "Enable Python bindings" OFF)
option(TTLANG_ENABLE_RUNTIME "Enable runtime support" OFF)
option(CODE_COVERAGE "Enable coverage reporting" OFF)

if (CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
endif()

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
    message(FATAL_ERROR "TTMLIR_TOOLCHAIN_DIR environment variable not set. Please run 'source env/activate'.")
endif()

if (NOT DEFINED ENV{TT_MLIR_HOME})
    message(FATAL_ERROR "TT_MLIR_HOME environment variable not set. Please run 'source env/activate'.")
endif()

# Add both tt-lang's and tt-mlir's CMake modules to the path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "$ENV{TT_MLIR_HOME}/cmake/modules")

# Enable debug asserts and debug logs only in Debug mode
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(TTLANG_BUILD_DEBUG TTLANG_ENABLE_DEBUG_LOGS)
endif()

# Compiler and linker setup
include(CompilerSetup)

# Use tt-mlir's build types
include(TTMLIRBuildTypes)

# tt-lang version
set(TTLANG_VERSION_MAJOR 0 CACHE STRING "tt-lang major version")
set(TTLANG_VERSION_MINOR 1 CACHE STRING "tt-lang minor version")
set(TTLANG_VERSION_PATCH 0 CACHE STRING "tt-lang patch version")
set(TTLANG_VERSION "${TTLANG_VERSION_MAJOR}.${TTLANG_VERSION_MINOR}.${TTLANG_VERSION_PATCH}")
message(STATUS "tt-lang version: ${TTLANG_VERSION}")

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

# Set Python3 executable explicitly (from tt-mlir's venv)
set(Python3_EXECUTABLE $ENV{TTMLIR_VENV_DIR}/bin/python3)

# Handle tt-mlir dependency (reads version from third-party/tt-mlir.commit)
# This will either find pre-built tt-mlir or fetch and build it
include(ExternTTMLIR)

# Find MLIR (using tt-mlir's FindMLIR module)
include(FindMLIR)

set(TTLANG_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TTLANG_BINARY_DIR ${PROJECT_BINARY_DIR})
set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})
set(LLVM_TOOLS_DIR "${TTMLIR_TOOLCHAIN_DIR}/bin")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${TTMLIR_INCLUDE_DIRS})
include_directories(${TTLANG_SOURCE_DIR}/include)
include_directories(${TTLANG_BINARY_DIR}/include)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
link_directories(${TTMLIR_LIBRARY_DIRS})

add_definitions(${LLVM_DEFINITIONS})

# Add subdirectories
add_subdirectory(third-party)
add_subdirectory(include)
add_subdirectory(lib)

if(MLIR_ENABLE_BINDINGS_PYTHON AND TTLANG_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  set(TTLANG_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  add_subdirectory(python)
endif()

add_subdirectory(test)
