# External tt-mlir dependency management
# Reads the required tt-mlir commit from third-party/tt-mlir.commit

# Try to find pre-built tt-mlir first
set(TTMLIR_HINTS)
if(DEFINED TTMLIR_DIR)
  list(APPEND TTMLIR_HINTS "${TTMLIR_DIR}")
endif()
if(DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
  list(APPEND TTMLIR_HINTS "$ENV{TTMLIR_TOOLCHAIN_DIR}/lib/cmake/ttmlir")
endif()
if(DEFINED ENV{TT_MLIR_HOME})
  list(APPEND TTMLIR_HINTS "$ENV{TT_MLIR_HOME}/build/lib/cmake/ttmlir")
endif()

find_package(TTMLIR QUIET CONFIG HINTS ${TTMLIR_HINTS})

if(TTMLIR_FOUND)
  message(STATUS "Using pre-built tt-mlir from: ${TTMLIR_CMAKE_DIR}")
else()
  set(TTMLIR_COMMIT_FILE "${CMAKE_SOURCE_DIR}/third-party/tt-mlir.commit")
  file(READ "${TTMLIR_COMMIT_FILE}" TTMLIR_GIT_TAG)
  string(STRIP "${TTMLIR_GIT_TAG}" TTMLIR_GIT_TAG)
  
  if("${TTMLIR_GIT_TAG}" STREQUAL "")
    message(FATAL_ERROR "tt-mlir.commit file does not contain a valid commit hash")
  endif()
  
  message(STATUS "Fetching and building tt-mlir version: ${TTMLIR_GIT_TAG}")
  
  include(FetchContent)
  FetchContent_Declare(
      tt-mlir
      GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
      GIT_TAG ${TTMLIR_GIT_TAG}
      GIT_SUBMODULES_RECURSE TRUE
      SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/tt-mlir-src"
      BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/tt-mlir-build"
  )
  FetchContent_MakeAvailable(tt-mlir)
  
  # After building, find the package
  find_package(TTMLIR REQUIRED CONFIG)
  message(STATUS "Built and using tt-mlir from: ${TTMLIR_CMAKE_DIR}")
endif()
