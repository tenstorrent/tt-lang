# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# BuildTTMetal.cmake -- build tt-metal from submodule
#
# tt-metal always builds unconditionally. It provides the ttnn Python package
# needed for compilation. Device presence (TTLANG_HAS_DEVICE) is a separate
# auto-detected concern that controls which tests run.
#
# Variables set (visible to caller via include()):
#   TT_METAL_HOME         - root of tt-metal source
#   TT_METAL_PYTHON_PATH  - path to add to PYTHONPATH for ttnn Python
#   TT_METAL_LIB_PATH     - path to add to LD_LIBRARY_PATH

include(ExternalProject)

set(TT_METAL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/tt-metal")

if(NOT EXISTS "${TT_METAL_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "tt-metal submodule not found at ${TT_METAL_SOURCE_DIR}. Run:\n"
    "  git submodule update --init third-party/tt-metal")
endif()

# Check that nested submodules are initialized (tt_llk uses pyproject.toml, not CMakeLists.txt)
foreach(_sub tt_metal/third_party/tracy/CMakeLists.txt
             tt_metal/third_party/tt_llk/README.md
             tt_metal/third_party/umd/CMakeLists.txt)
  if(NOT EXISTS "${TT_METAL_SOURCE_DIR}/${_sub}")
    get_filename_component(_sub_dir "${_sub}" DIRECTORY)
    message(FATAL_ERROR
      "tt-metal nested submodule ${_sub_dir} not initialized. Run:\n"
      "  cd ${TT_METAL_SOURCE_DIR} && git submodule update --init --recursive")
  endif()
endforeach()

message(STATUS "tt-metal runtime: building from submodule at ${TT_METAL_SOURCE_DIR}")

# Install minimal Python dependencies required to import ttnn at runtime
ttlang_pip_install_requirements("${Python3_EXECUTABLE}"
  "${CMAKE_SOURCE_DIR}/third-party/requirements-metal.txt")

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

# Library paths (build byproducts)
set(TTMETAL_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/libtt_metal.so")
set(TT_STL_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/libtt_stl.so")
set(TTNN_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/_ttnncpp.so")
set(TTNN_PY_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/_ttnn.so")
set(DEVICE_LIBRARY_PATH "${TTMETAL_LIBRARY_DIR}/libdevice.so")

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
    -DCMAKE_TOOLCHAIN_FILE=${TT_METAL_SOURCE_DIR}/cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}
    # Python bindings
    -DWITH_PYTHON_BINDINGS=ON
    # Minimal build flags
    -DTT_UNITY_BUILDS=ON
    -DENABLE_CCACHE=${TTMETAL_ENABLE_CCACHE}
    -DENABLE_TRACY=OFF
    -DENABLE_DISTRIBUTED=OFF
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_PROGRAMMING_EXAMPLES=OFF
    -DTT_METAL_BUILD_TESTS=OFF
    -DTTNN_BUILD_TESTS=OFF
    -DBUILD_TT_TRAIN=OFF
    -DBUILD_TELEMETRY=OFF
    -DENABLE_TTNN_SHARED_SUBLIBS=OFF
    -DTT_ENABLE_LIGHT_METAL_TRACE=OFF
    -DENABLE_LIBCXX=OFF
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=${TTMETAL_DISABLE_PRECOMPILE_HEADERS}
  BUILD_BYPRODUCTS
    ${TTMETAL_LIBRARY_PATH}
    ${TT_STL_LIBRARY_PATH}
    ${TTNN_LIBRARY_PATH}
    ${TTNN_PY_LIBRARY_PATH}
    ${DEVICE_LIBRARY_PATH}
)

# Expose the build step as a target so we can attach post-build commands
ExternalProject_Add_StepTargets(tt-metal build)

# Copy ttnn Python extensions into the source tree so that `import ttnn` works
add_custom_command(TARGET tt-metal-build POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TTMETAL_BUILD_DIR}/ttnn/_ttnn.so
    ${TT_METAL_SOURCE_DIR}/ttnn/ttnn/_ttnn.so
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TTMETAL_BUILD_DIR}/ttnn/_ttnncpp.so
    ${TT_METAL_SOURCE_DIR}/ttnn/ttnn/_ttnncpp.so
  COMMENT "Copying ttnn Python extensions to source tree"
)

# Create imported library targets
set(_lib_names TTMETAL_LIBRARY TT_STL_LIBRARY TTNN_LIBRARY TTNN_PY_LIBRARY DEVICE_LIBRARY)
set(_lib_paths
  "${TTMETAL_LIBRARY_PATH}" "${TT_STL_LIBRARY_PATH}" "${TTNN_LIBRARY_PATH}"
  "${TTNN_PY_LIBRARY_PATH}" "${DEVICE_LIBRARY_PATH}")
foreach(_name _path IN ZIP_LISTS _lib_names _lib_paths)
  add_library(${_name} SHARED IMPORTED GLOBAL)
  set_target_properties(${_name} PROPERTIES
    IMPORTED_LOCATION "${_path}"
  )
  add_dependencies(${_name} tt-metal)
endforeach()

# Set variables for activate.in
set(TT_METAL_HOME "${TT_METAL_SOURCE_DIR}")
set(TT_METAL_PYTHON_PATH "${TT_METAL_SOURCE_DIR}/ttnn")
set(TT_METAL_LIB_PATH "${TTMETAL_BUILD_DIR}/lib:${TTMETAL_BUILD_DIR}/tt_metal:${TTMETAL_BUILD_DIR}/ttnn")
