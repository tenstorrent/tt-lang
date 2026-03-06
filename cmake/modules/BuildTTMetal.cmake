# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# BuildTTMetal.cmake -- build tt-metal from submodule at configure time
#
# Builds tt-metal once during cmake configure and caches the result.
# Subsequent configures skip the build if _ttnn.so already exists.
#
# Variables set (visible to caller via include()):
#   TT_METAL_HOME         - root of tt-metal source
#   TT_METAL_PYTHON_PATH  - path to add to PYTHONPATH for ttnn Python
#   TT_METAL_LIB_PATH     - path to add to LD_LIBRARY_PATH

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

option(TTLANG_ENABLE_PERF_TRACE "Enable performance tracing (Tracy) in tt-metal" ON)

message(STATUS "tt-metal runtime: building from submodule at ${TT_METAL_SOURCE_DIR}")

# Install minimal Python dependencies required to import ttnn at runtime
ttlang_pip_install_requirements("${Python3_EXECUTABLE}"
  "${CMAKE_SOURCE_DIR}/third-party/requirements-metal.txt")

# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------
set(TTMETAL_BUILD_DIR "${TT_METAL_SOURCE_DIR}/build")
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

# Sentinel file: if this exists, tt-metal is already built.
set(_TTNN_SO "${TTMETAL_BUILD_DIR}/ttnn/_ttnn.so")

if(EXISTS "${_TTNN_SO}")
  message(STATUS "tt-metal already built at ${TTMETAL_BUILD_DIR}, skipping rebuild")
else()
  # --- Configure ---
  set(_TTMETAL_CMAKE_ARGS
    -G Ninja
    -S "${TT_METAL_SOURCE_DIR}"
    -B "${TTMETAL_BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${TTMETAL_BUILD_DIR}
    -DCMAKE_INSTALL_MESSAGE=NEVER
    -DCMAKE_TOOLCHAIN_FILE=${TT_METAL_SOURCE_DIR}/cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}
    # Python bindings -- use the same interpreter as the tt-lang venv
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DPython3_FIND_VIRTUALENV=ONLY
    -DWITH_PYTHON_BINDINGS=ON
    # Minimal build flags
    -DTT_UNITY_BUILDS=ON
    -DENABLE_CCACHE=${TTMETAL_ENABLE_CCACHE}
    -DENABLE_TRACY=${TTLANG_ENABLE_PERF_TRACE}
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
  )

  message(STATUS "Configuring tt-metal...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} ${_TTMETAL_CMAKE_ARGS}
    RESULT_VARIABLE _TTMETAL_CONFIG_RESULT
  )
  if(NOT _TTMETAL_CONFIG_RESULT EQUAL 0)
    message(FATAL_ERROR "tt-metal configure failed (exit ${_TTMETAL_CONFIG_RESULT})")
  endif()

  # --- Build ---
  message(STATUS "Building tt-metal (this may take a while)...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${TTMETAL_BUILD_DIR}"
    RESULT_VARIABLE _TTMETAL_BUILD_RESULT
  )
  if(NOT _TTMETAL_BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "tt-metal build failed (exit ${_TTMETAL_BUILD_RESULT})")
  endif()

  # Verify the sentinel was produced
  if(NOT EXISTS "${_TTNN_SO}")
    message(FATAL_ERROR
      "tt-metal build completed but ${_TTNN_SO} was not produced")
  endif()
endif()

# ---------------------------------------------------------------------------
# Copy ttnn Python extensions into the source tree so that `import ttnn` works
# ---------------------------------------------------------------------------
file(COPY_FILE
  "${TTMETAL_BUILD_DIR}/ttnn/_ttnn.so"
  "${TT_METAL_SOURCE_DIR}/ttnn/ttnn/_ttnn.so"
  ONLY_IF_DIFFERENT)
file(COPY_FILE
  "${TTMETAL_BUILD_DIR}/ttnn/_ttnncpp.so"
  "${TT_METAL_SOURCE_DIR}/ttnn/ttnn/_ttnncpp.so"
  ONLY_IF_DIFFERENT)

# ---------------------------------------------------------------------------
# Set variables for activate.in
# ---------------------------------------------------------------------------
set(TT_METAL_HOME "${TT_METAL_SOURCE_DIR}")
set(TT_METAL_PYTHON_PATH "${TT_METAL_SOURCE_DIR}/ttnn:${TT_METAL_SOURCE_DIR}/tools")
set(TT_METAL_LIB_PATH "${TTMETAL_BUILD_DIR}/lib:${TTMETAL_BUILD_DIR}/tt_metal:${TTMETAL_BUILD_DIR}/ttnn")
