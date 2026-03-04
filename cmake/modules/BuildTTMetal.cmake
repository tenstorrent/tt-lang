# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# BuildTTMetal.cmake -- optional tt-metal build from submodule
#
# Usage:
#   -DTTLANG_ENABLE_RUNTIME=ON    Build tt-metal from submodule
#
# When TTLANG_ENABLE_RUNTIME is OFF (default), this file is a no-op and
# tt-lang can still compile kernels to EmitC -- only runtime execution
# requires tt-metal.
#
# Variables set (visible to caller via include()):
#   TT_METAL_HOME         - root of tt-metal source
#   TT_METAL_PYTHON_PATH  - path to add to PYTHONPATH for ttnn Python
#   TT_METAL_LIB_PATH     - path to add to LD_LIBRARY_PATH

if(NOT TTLANG_ENABLE_RUNTIME)
  message(STATUS "tt-metal runtime: DISABLED (set -DTTLANG_ENABLE_RUNTIME=ON to enable)")
else()
  include(ExternalProject)

  set(TT_METAL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/tt-metal")

  if(NOT EXISTS "${TT_METAL_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "TTLANG_ENABLE_RUNTIME=ON but tt-metal submodule not found at "
      "${TT_METAL_SOURCE_DIR}. Run: git submodule update --init third-party/tt-metal")
  endif()

  message(STATUS "tt-metal runtime: building from submodule at ${TT_METAL_SOURCE_DIR}")

  # Build configuration
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(TTMETAL_BUILD_TYPE "Release")
    set(TTMETAL_BUILD_DIR "${TT_METAL_SOURCE_DIR}/build")
  else()
    set(TTMETAL_BUILD_TYPE "RelWithDebInfo")
    set(TTMETAL_BUILD_DIR "${TT_METAL_SOURCE_DIR}/build_Debug")
  endif()
  set(TTMETAL_LIBRARY_DIR "${TTMETAL_BUILD_DIR}/lib")

  # CPM cache location (tt-metal uses CPM for its dependencies)
  if(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_SOURCE_CACHE "$ENV{CPM_SOURCE_CACHE}")
  else()
    set(CPM_SOURCE_CACHE "${TT_METAL_SOURCE_DIR}/.cpmcache")
  endif()

  # ccache forwarding
  set(TTMETAL_ENABLE_CCACHE OFF)
  set(TTMETAL_DISABLE_PRECOMPILE_HEADERS OFF)
  if("${CMAKE_CXX_COMPILER_LAUNCHER}" STREQUAL "ccache")
    set(TTMETAL_ENABLE_CCACHE ON)
    set(TTMETAL_DISABLE_PRECOMPILE_HEADERS ON)
  endif()

  # Distributed support only on x86_64
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(TT_METAL_ENABLE_DISTRIBUTED ON)
  else()
    set(TT_METAL_ENABLE_DISTRIBUTED OFF)
  endif()

  # Library paths (build byproducts)
  set(TTMETAL_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/libtt_metal.so")
  set(TT_STL_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/libtt_stl.so")
  set(TTNN_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/_ttnncpp.so")
  set(TTNN_PY_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/_ttnn.so")
  set(DEVICE_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/libdevice.so")
  set(TRACY_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/libtracy.so")

  ExternalProject_Add(tt-metal
    PREFIX "${CMAKE_BINARY_DIR}/tt-metal-build"
    SOURCE_DIR "${TT_METAL_SOURCE_DIR}"
    BINARY_DIR "${TTMETAL_BUILD_DIR}"
    PATCH_COMMAND ${CMAKE_COMMAND} -E make_directory "${TTMETAL_BUILD_DIR}"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${TTMETAL_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${TTMETAL_BUILD_DIR}
      -DCMAKE_INSTALL_MESSAGE=NEVER
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      -DCPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}
      # Python bindings
      -DWITH_PYTHON_BINDINGS=ON
      -DEXPERIMENTAL_NANOBIND_BINDINGS=ON
      # Minimal build flags
      -DBUILD_SHARED_LIBS=ON
      -DBUILD_PROGRAMMING_EXAMPLES=OFF
      -DTT_METAL_BUILD_TESTS=OFF
      -DTTNN_BUILD_TESTS=OFF
      -DBUILD_TT_TRAIN=OFF
      -DENABLE_DISTRIBUTED=${TT_METAL_ENABLE_DISTRIBUTED}
      -DENABLE_TRACY=${TTLANG_ENABLE_PERF_TRACE}
      -DENABLE_LIBCXX=OFF
      -DENABLE_CCACHE=${TTMETAL_ENABLE_CCACHE}
      -DCMAKE_DISABLE_PRECOMPILE_HEADERS=${TTMETAL_DISABLE_PRECOMPILE_HEADERS}
      -DTT_UNITY_BUILDS=ON
    BUILD_BYPRODUCTS
      ${TTMETAL_LIBRARY_PATH}
      ${TT_STL_LIBRARY_PATH}
      ${TTNN_LIBRARY_PATH}
      ${TTNN_PY_LIBRARY_PATH}
      ${DEVICE_LIBRARY_PATH}
      ${TRACY_LIBRARY_PATH}
  )

  # Make tt-metal EXCLUDE_FROM_ALL so it only builds when explicitly requested
  # via `cmake --build build --target tt-metal` or when a dependent target needs it.
  set_target_properties(tt-metal PROPERTIES EXCLUDE_FROM_ALL TRUE)

  # Create imported library targets
  set(_lib_names TTMETAL_LIBRARY TT_STL_LIBRARY TTNN_LIBRARY TTNN_PY_LIBRARY DEVICE_LIBRARY TRACY_LIBRARY)
  set(_lib_paths
    "${TTMETAL_LIBRARY_PATH}" "${TT_STL_LIBRARY_PATH}" "${TTNN_LIBRARY_PATH}"
    "${TTNN_PY_LIBRARY_PATH}" "${DEVICE_LIBRARY_PATH}" "${TRACY_LIBRARY_PATH}")
  foreach(_name _path IN ZIP_LISTS _lib_names _lib_paths)
    add_library(${_name} SHARED IMPORTED GLOBAL)
    set_target_properties(${_name} PROPERTIES
      EXCLUDE_FROM_ALL TRUE
      IMPORTED_LOCATION "${_path}"
    )
    add_dependencies(${_name} tt-metal)
  endforeach()

  # Set variables for activate.in
  set(TT_METAL_HOME "${TT_METAL_SOURCE_DIR}")
  set(TT_METAL_PYTHON_PATH "${TT_METAL_SOURCE_DIR}")
  set(TT_METAL_LIB_PATH "${TTMETAL_LIBRARY_DIR}")
endif()
