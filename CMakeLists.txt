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
ttlang_set_version("0.2.0")

option(TTLANG_ENABLE_BINDINGS_PYTHON "Enable Python bindings" ON)
option(TTLANG_ENABLE_RUNTIME "Enable runtime support" OFF)
option(CODE_COVERAGE "Enable coverage reporting" OFF)
option(TTLANG_ENABLE_PERF_TRACE "Enable performance trace" OFF)

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
include(ExternTTMLIR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

set(TTLANG_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TTLANG_BINARY_DIR ${PROJECT_BINARY_DIR})
set(LLVM_TOOLS_DIR "${TTMLIR_TOOLCHAIN_DIR}/bin")

set(TT_LANG_HOME "${CMAKE_CURRENT_LIST_DIR}")

configure_file(
  "${PROJECT_SOURCE_DIR}/env/activate.in"
  "${PROJECT_BINARY_DIR}/env/activate"
  @ONLY
)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${TTMLIR_INCLUDE_DIRS})
include_directories(${TTLANG_SOURCE_DIR}/include)
include_directories(${TTLANG_BINARY_DIR}/include)
include_directories(${TTMLIR_TOOLCHAIN_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
link_directories(${TTMLIR_LIBRARY_DIRS})

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(third-party)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(MLIR_ENABLE_BINDINGS_PYTHON AND TTLANG_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  set(TTLANG_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  add_subdirectory(python)
endif()

add_subdirectory(test)
