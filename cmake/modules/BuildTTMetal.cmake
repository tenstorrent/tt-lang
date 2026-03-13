# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# BuildTTMetal.cmake -- build tt-metal from submodule at configure time
#
# Builds tt-metal once during cmake configure and caches the result.
# Subsequent configures skip the build if _ttnn.so already exists.
#
# Variables set (visible to caller via include()):
# TT_METAL_HOME         - root of tt-metal source
# TT_METAL_PYTHON_PATH  - path to add to PYTHONPATH for ttnn Python
# TT_METAL_LIB_PATH     - path to add to LD_LIBRARY_PATH

set(TT_METAL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/tt-metal")

# ---------------------------------------------------------------------------
# macOS: skip tt-metal build (Linux-only runtime dependencies).
# Still set TT_METAL_HOME so headers are available, and provide empty
# paths for PYTHONPATH / LD_LIBRARY_PATH.
# ---------------------------------------------------------------------------
if(APPLE)
  message(STATUS "tt-metal runtime: skipped on macOS (Linux-only dependencies)")
  message(STATUS "  ttnn will not be available; use the simulator instead")

  ttlang_ensure_submodules(third-party/tt-metal)

  set(TT_METAL_HOME "${TT_METAL_SOURCE_DIR}")
  set(TT_METAL_PYTHON_PATH "")
  set(TT_METAL_LIB_PATH "")

  # Provide a no-op clean target for consistency.
  add_custom_target(clean-ttmetal
    COMMENT "tt-metal is not built on macOS; nothing to clean."
  )

  return()
endif()

# ---------------------------------------------------------------------------
# Pre-built toolchain: tt-metal artifacts are already installed.
# Set variables for activate.in and skip the build.
# ---------------------------------------------------------------------------
if(TTLANG_USE_TOOLCHAIN)
  set(TTMETAL_BUILD_DIR "${TTLANG_TOOLCHAIN_DIR}/tt-metal" CACHE PATH
    "tt-metal build directory (from toolchain)" FORCE)
  set(TT_METAL_HOME "${TTMETAL_BUILD_DIR}")
  set(TT_METAL_PYTHON_PATH "${TTMETAL_BUILD_DIR}/python_packages/ttnn:${TTMETAL_BUILD_DIR}/python_packages/tools")
  set(TT_METAL_LIB_PATH "${TTMETAL_BUILD_DIR}/lib")

  add_custom_target(clean-ttmetal
    COMMENT "tt-metal uses pre-built toolchain; nothing to clean."
  )

  return()
endif()

# ---------------------------------------------------------------------------
# Linux path: full tt-metal build from submodule.
# ---------------------------------------------------------------------------
ttlang_ensure_submodules(third-party/tt-metal)

# tt-metal has nested submodules (tracy, tt_llk, umd) required for building
# from source. Initialize them recursively if not already present.
# When using a pre-built toolchain the nested submodules are not required —
# only the top-level source tree is needed for JIT headers at device runtime.
if(NOT TTLANG_USE_TOOLCHAIN)
  set(_nested_missing FALSE)

  foreach(_sub tt_metal/third_party/tracy/CMakeLists.txt
    tt_metal/third_party/tt_llk/README.md
    tt_metal/third_party/umd/CMakeLists.txt)
    if(NOT EXISTS "${TT_METAL_SOURCE_DIR}/${_sub}")
      set(_nested_missing TRUE)
      break()
    endif()
  endforeach()

  if(_nested_missing AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    message(STATUS "Initializing tt-metal nested submodules...")
    execute_process(
      COMMAND git submodule update --init --recursive --depth 1
      WORKING_DIRECTORY "${TT_METAL_SOURCE_DIR}"
      RESULT_VARIABLE _sub_result
    )

    if(NOT _sub_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to initialize tt-metal nested submodules. Run manually:\n"
        "  cd ${TT_METAL_SOURCE_DIR} && git submodule update --init --recursive")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Verify tt-metal submodule matches the version expected by tt-mlir.
# ---------------------------------------------------------------------------
set(_TTMLIR_THIRD_PARTY_CMAKELISTS "${CMAKE_SOURCE_DIR}/third-party/tt-mlir/third_party/CMakeLists.txt")

if(EXISTS "${_TTMLIR_THIRD_PARTY_CMAKELISTS}")
  file(STRINGS "${_TTMLIR_THIRD_PARTY_CMAKELISTS}" _ttmetal_version_line
    REGEX "set\\(TT_METAL_VERSION")

  if(_ttmetal_version_line)
    string(REGEX MATCH "\"([a-f0-9]+)\"" _match "${_ttmetal_version_line}")

    if(_match)
      ttlang_verify_ttmetal_sha("${TT_METAL_SOURCE_DIR}" "${CMAKE_MATCH_1}")
    endif()
  endif()
endif()

option(TTLANG_ENABLE_PERF_TRACE "Enable performance tracing (Tracy) in tt-metal" ON)

ttlang_get_submodule_sha("${TT_METAL_SOURCE_DIR}" _TTMETAL_SUBMODULE_SHA)
string(SUBSTRING "${_TTMETAL_SUBMODULE_SHA}" 0 7 _TTMETAL_SHORT_SHA)

message(STATUS "tt-metal runtime: building from submodule at ${TT_METAL_SOURCE_DIR}")
message(STATUS "  Commit SHA: ${_TTMETAL_SHORT_SHA}")

# Apply patches to tt-metal source tree.
ttlang_apply_patches("${TT_METAL_SOURCE_DIR}"
  "${CMAKE_SOURCE_DIR}/third-party/patches/ttmetal-*.patch")

# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------
set(TTMETAL_BUILD_DIR "${CMAKE_BINARY_DIR}/tt-metal" CACHE PATH
  "tt-metal build directory")
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
  # Remove any stale build dir (e.g. from a previous failed configure) to
  # avoid CMakeCache.txt conflicts with the new configuration.
  if(EXISTS "${TTMETAL_BUILD_DIR}")
    message(STATUS "Removing stale tt-metal build directory at ${TTMETAL_BUILD_DIR}")
    file(REMOVE_RECURSE "${TTMETAL_BUILD_DIR}")
  endif()

  # --- Configure ---
  set(_TTMETAL_CMAKE_ARGS
    -G Ninja
    -S "${TT_METAL_SOURCE_DIR}"
    -B "${TTMETAL_BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${TTMETAL_BUILD_DIR}
    -DCMAKE_INSTALL_MESSAGE=NEVER
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
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

  # Save runtime artifacts into the toolchain build dir so they survive
  # across machines/caches.
  execute_process(
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/copy-ttmetal-runtime-artifacts.sh"
      "${TT_METAL_SOURCE_DIR}" "${TTMETAL_BUILD_DIR}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

# ---------------------------------------------------------------------------
# Copy ttnn Python extensions into the source tree so that `import ttnn` works
# ---------------------------------------------------------------------------
foreach(_so _ttnn.so _ttnncpp.so)
  set(_src "${TTMETAL_BUILD_DIR}/ttnn/${_so}")
  set(_dst "${TT_METAL_SOURCE_DIR}/ttnn/ttnn/${_so}")

  if(EXISTS "${_src}")
    file(COPY_FILE "${_src}" "${_dst}" ONLY_IF_DIFFERENT)
  else()
    message(WARNING "tt-metal artifact not found: ${_src}")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# Restore runtime artifacts from toolchain into the source tree.
# The JIT build system resolves these via TT_METAL_HOME at device runtime.
# ---------------------------------------------------------------------------
execute_process(
  COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/copy-ttmetal-runtime-artifacts.sh"
    --restore "${TTMETAL_BUILD_DIR}" "${TT_METAL_SOURCE_DIR}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Install tt-metal artifacts into toolchain directory.
# In build mode with TTLANG_TOOLCHAIN_DIR set, copy shared libraries, Python
# packages, runtime artifacts, and JIT source trees so the toolchain is
# self-contained for Docker image builds and cross-machine caching.
# ---------------------------------------------------------------------------
if(DEFINED TTLANG_TOOLCHAIN_DIR AND NOT TTLANG_USE_TOOLCHAIN)
  set(_TTMETAL_INSTALL_DIR "${TTLANG_TOOLCHAIN_DIR}/tt-metal")
  message(STATUS "Installing tt-metal artifacts into ${_TTMETAL_INSTALL_DIR}")
  execute_process(
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/install-ttmetal.sh"
      "${TT_METAL_SOURCE_DIR}" "${TTMETAL_BUILD_DIR}" "${_TTMETAL_INSTALL_DIR}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _TTMETAL_INSTALL_RESULT
  )
  if(NOT _TTMETAL_INSTALL_RESULT EQUAL 0)
    message(WARNING "tt-metal install into toolchain failed (exit ${_TTMETAL_INSTALL_RESULT})")
  endif()
endif()

# ---------------------------------------------------------------------------
# Set variables for activate.in
# ---------------------------------------------------------------------------
set(TT_METAL_HOME "${TT_METAL_SOURCE_DIR}")
if(DEFINED TTLANG_TOOLCHAIN_DIR)
  # Use the installed layout so that env/activate works after the source tree
  # is removed (e.g. inside Docker images built from the toolchain).
  set(TT_METAL_PYTHON_PATH "${TTLANG_TOOLCHAIN_DIR}/tt-metal/python_packages/ttnn:${TTLANG_TOOLCHAIN_DIR}/tt-metal/python_packages/tools")
else()
  # Pure source-tree build without a toolchain install: reference the source
  # tree directly (mirrors tt-metal's own layout).
  set(TT_METAL_PYTHON_PATH "${TT_METAL_SOURCE_DIR}/ttnn:${TT_METAL_SOURCE_DIR}/tools")
endif()
set(TT_METAL_LIB_PATH "${TTMETAL_BUILD_DIR}/lib:${TTMETAL_BUILD_DIR}/tt_metal:${TTMETAL_BUILD_DIR}/ttnn:${TTMETAL_BUILD_DIR}/tt_stl:${TTMETAL_BUILD_DIR}/_deps/fmt-build:${TTMETAL_BUILD_DIR}/tt_metal/third_party/umd/device")

# ---------------------------------------------------------------------------
# clean-ttmetal target: removes tt-metal build dir and copied extensions so
# the next cmake configure rebuilds from scratch.
# ---------------------------------------------------------------------------
add_custom_target(clean-ttmetal
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${TTMETAL_BUILD_DIR}"
  COMMAND ${CMAKE_COMMAND} -E rm -f
    "${TT_METAL_SOURCE_DIR}/ttnn/ttnn/_ttnn.so"
    "${TT_METAL_SOURCE_DIR}/ttnn/ttnn/_ttnncpp.so"
  COMMENT "Removing tt-metal build directory. Re-run cmake configure to rebuild."
)
