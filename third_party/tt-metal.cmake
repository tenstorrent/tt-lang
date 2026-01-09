# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# tt-metal Minimal Build via ExternalProject
#
# This file builds tt-metal with minimal configuration for tt-lang runtime:
# - ttnn_core: Core tensor operations
# - ttnncpp: TTNN C++ with JIT compilation
# - _ttnncpp.so: Python bindings for TTNN C++
# - ttnn: Full Python bindings module
#
# Based on proven configuration from /home/bnorris/tt/tt-metal/build_ultra_minimal.sh
# which produces ~645MB build (vs 6-8GB full) in ~5-6 min (vs 15-20 min).
#
# Part of "Plan: Minimal tt-metal and tt-mlir Integration for tt-lang"

include(ExternalProject)

# tt-metal version - pinned to same commit used by tt-mlir
# This should match third_party/tt-metal/src/tt-metal commit in tt-mlir
set(TT_METAL_VERSION "dcae048e5901b2528f5eba67180bb2bc0c227481" CACHE STRING
    "tt-metal git commit hash to build")

# Allow overriding tt-metal source location for local development
set(TT_METAL_SOURCE_DIR "" CACHE PATH
    "Path to existing tt-metal source (if empty, will clone from GitHub)")

# Build configuration
set(TT_METAL_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
    "Build type for tt-metal (Release, Debug, RelWithDebInfo)")

# Build directory under tt-lang's build tree
set(TT_METAL_BUILD_DIR "${CMAKE_BINARY_DIR}/third_party/tt-metal-build" CACHE PATH
    "Build directory for tt-metal")

set(TT_METAL_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/tt-metal-install" CACHE PATH
    "Install directory for tt-metal")

# Toolchain file - tt-metal requires specific clang-17 setup on Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(TT_METAL_TOOLCHAIN_FILE "cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake")
else()
    set(TT_METAL_TOOLCHAIN_FILE "")
endif()

# Find ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(TT_METAL_ENABLE_CCACHE ON)
    message(STATUS "tt-metal: Using ccache")
else()
    set(TT_METAL_ENABLE_CCACHE OFF)
endif()

# Determine if we're cloning or using existing source
if(TT_METAL_SOURCE_DIR)
    message(STATUS "tt-metal: Using existing source at ${TT_METAL_SOURCE_DIR}")
    set(_TT_METAL_DOWNLOAD_COMMAND "")
    set(_TT_METAL_SOURCE_DIR "${TT_METAL_SOURCE_DIR}")
else()
    message(STATUS "tt-metal: Will clone version ${TT_METAL_VERSION}")
    set(_TT_METAL_SOURCE_DIR "${CMAKE_BINARY_DIR}/third_party/tt-metal-src")
endif()

# CMake arguments for minimal build
# These match the proven configuration from build_ultra_minimal.sh
set(TT_METAL_CMAKE_ARGS
    -G Ninja
    -DCMAKE_BUILD_TYPE=${TT_METAL_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${TT_METAL_INSTALL_DIR}
    # Unity builds - faster compilation
    -DTT_UNITY_BUILDS=ON
    # Ccache support
    -DENABLE_CCACHE=${TT_METAL_ENABLE_CCACHE}
    # Disable profiler (Tracy)
    -DENABLE_TRACY=OFF
    # Disable OpenMPI distributed compute
    -DENABLE_DISTRIBUTED=OFF
    # Enable Python bindings (needed for ttnn)
    -DWITH_PYTHON_BINDINGS=ON
    # Disable tests (we don't need them for tt-lang runtime)
    -DTTNN_BUILD_TESTS=OFF
    -DTT_METAL_BUILD_TESTS=OFF
    # Skip examples
    -DBUILD_PROGRAMMING_EXAMPLES=OFF
    # Skip tt-train
    -DBUILD_TT_TRAIN=OFF
    # Skip telemetry
    -DBUILD_TELEMETRY=OFF
    # Monolithic build (no shared sublibs)
    -DENABLE_TTNN_SHARED_SUBLIBS=OFF
    # Disable light metal tracing
    -DTT_ENABLE_LIGHT_METAL_TRACE=OFF
)

# Add toolchain file if specified
if(TT_METAL_TOOLCHAIN_FILE)
    list(APPEND TT_METAL_CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=<SOURCE_DIR>/${TT_METAL_TOOLCHAIN_FILE}
    )
endif()

# Build targets - only build what we need
# These targets are built sequentially because they have dependencies
set(TT_METAL_BUILD_TARGETS
    ttnn_core
    ttnncpp
    _ttnncpp.so
    ttnn
)

# Convert target list to build commands
set(TT_METAL_BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttnn_core &&
    ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttnncpp &&
    ${CMAKE_COMMAND} --build <BINARY_DIR> --target _ttnncpp.so &&
    ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttnn
)

# ExternalProject setup
if(TT_METAL_SOURCE_DIR)
    # Use existing source directory
    ExternalProject_Add(tt-metal
        SOURCE_DIR ${_TT_METAL_SOURCE_DIR}
        BINARY_DIR ${TT_METAL_BUILD_DIR}
        INSTALL_DIR ${TT_METAL_INSTALL_DIR}

        CMAKE_ARGS ${TT_METAL_CMAKE_ARGS}

        BUILD_COMMAND
            ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttnn_core
            COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttnncpp
            COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target _ttnncpp.so
            COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttnn

        # Skip install - we use the build tree directly
        INSTALL_COMMAND ""

        # Don't update/patch existing source
        UPDATE_COMMAND ""
        PATCH_COMMAND ""

        BUILD_BYPRODUCTS
            ${TT_METAL_BUILD_DIR}/ttnn/_ttnn.so
            ${TT_METAL_BUILD_DIR}/ttnn/_ttnncpp.so
            ${TT_METAL_BUILD_DIR}/tt_metal/libtt_metal.so
    )
else()
    # Clone from GitHub with submodules
    ExternalProject_Add(tt-metal
        GIT_REPOSITORY https://github.com/tenstorrent/tt-metal.git
        GIT_TAG ${TT_METAL_VERSION}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        GIT_SUBMODULES_RECURSE TRUE

        SOURCE_DIR ${_TT_METAL_SOURCE_DIR}
        BINARY_DIR ${TT_METAL_BUILD_DIR}
        INSTALL_DIR ${TT_METAL_INSTALL_DIR}

        CMAKE_ARGS ${TT_METAL_CMAKE_ARGS}

        BUILD_COMMAND
            ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttnn_core
            COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttnncpp
            COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target _ttnncpp.so
            COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target ttnn

        # Skip install - we use the build tree directly
        INSTALL_COMMAND ""

        BUILD_BYPRODUCTS
            ${TT_METAL_BUILD_DIR}/ttnn/_ttnn.so
            ${TT_METAL_BUILD_DIR}/ttnn/_ttnncpp.so
            ${TT_METAL_BUILD_DIR}/tt_metal/libtt_metal.so
    )
endif()

# Export variables for use by parent CMake
set(TT_METAL_HOME ${_TT_METAL_SOURCE_DIR} PARENT_SCOPE)
set(TT_METAL_LIBRARY_DIR ${TT_METAL_BUILD_DIR}/lib PARENT_SCOPE)
set(TT_METAL_TTNN_DIR ${TT_METAL_BUILD_DIR}/ttnn PARENT_SCOPE)

# Libraries and paths needed at runtime
set(TTNN_LIBRARY_PATH ${TT_METAL_BUILD_DIR}/ttnn/_ttnncpp.so PARENT_SCOPE)
set(TTNN_PY_LIBRARY_PATH ${TT_METAL_BUILD_DIR}/ttnn/_ttnn.so PARENT_SCOPE)
set(TTMETAL_LIBRARY_PATH ${TT_METAL_BUILD_DIR}/tt_metal/libtt_metal.so PARENT_SCOPE)

# Helper target to print build info
add_custom_target(tt-metal-info
    COMMAND ${CMAKE_COMMAND} -E echo "tt-metal minimal build information:"
    COMMAND ${CMAKE_COMMAND} -E echo "  Source: ${_TT_METAL_SOURCE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Build: ${TT_METAL_BUILD_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Version: ${TT_METAL_VERSION}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Runtime environment:"
    COMMAND ${CMAKE_COMMAND} -E echo "  export TT_METAL_HOME=${_TT_METAL_SOURCE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "  export LD_LIBRARY_PATH=${TT_METAL_BUILD_DIR}/tt_metal:${TT_METAL_BUILD_DIR}/ttnn:${TT_METAL_BUILD_DIR}/lib:\$LD_LIBRARY_PATH"
    VERBATIM
)

message(STATUS "tt-metal minimal build configured:")
message(STATUS "  Source: ${_TT_METAL_SOURCE_DIR}")
message(STATUS "  Build: ${TT_METAL_BUILD_DIR}")
message(STATUS "  Version: ${TT_METAL_VERSION}")
message(STATUS "  Build type: ${TT_METAL_BUILD_TYPE}")
