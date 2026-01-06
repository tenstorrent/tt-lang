# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Copy tt-metal's ttnn Python package to build tree.
# TTNN is required for compiling and running tt-lang kernels.

# Find ttnn source directory from TT_METAL_HOME or tt-mlir's third_party.
# _TTMLIR_CMAKE_HOME_DIRECTORY is loaded from the tt-mlir build cache when
# using TTMLIR_BUILD_DIR (see ttlang_setup_ttmlir_build_tree in TTLangUtils.cmake).
if(DEFINED TT_METAL_HOME AND EXISTS "${TT_METAL_HOME}/ttnn/ttnn")
  set(_TTNN_SOURCE_DIR "${TT_METAL_HOME}/ttnn/ttnn")
elseif(DEFINED _TTMLIR_CMAKE_HOME_DIRECTORY AND EXISTS "${_TTMLIR_CMAKE_HOME_DIRECTORY}/third_party/tt-metal/src/tt-metal/ttnn/ttnn")
  set(_TTNN_SOURCE_DIR "${_TTMLIR_CMAKE_HOME_DIRECTORY}/third_party/tt-metal/src/tt-metal/ttnn/ttnn")
endif()

if(DEFINED _TTNN_SOURCE_DIR)
  message(STATUS "Found ttnn Python package at: ${_TTNN_SOURCE_DIR}")

  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/python_packages/ttnn/__init__.py
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${_TTNN_SOURCE_DIR}
      ${CMAKE_BINARY_DIR}/python_packages/ttnn
    COMMENT "Copying tt-metal ttnn Python package to build tree"
  )

  add_custom_target(copy-ttnn-python-package ALL
    DEPENDS ${CMAKE_BINARY_DIR}/python_packages/ttnn/__init__.py
  )
else()
  message(STATUS "ttnn Python package not found - Python lit tests may be skipped")
endif()
