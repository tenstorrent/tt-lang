# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BuildExternalDialects.cmake - Build StableHLO and Shardy dialect IR
# definitions from submodules.
#
# StableHLO: native CMake, STABLEHLO_BUILD_EMBEDDED mode.
# Shardy: CMake added via tt-mlir patch (adds CMakeLists.txt to submodule).
#
# Only dialect IR definitions are built. No transform passes needed for the
# convert-stablehlo-to-ttl pass.

# ---------------------------------------------------------------------------
# StableHLO
# ---------------------------------------------------------------------------
set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "" FORCE)
set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)

set(STABLEHLO_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/stablehlo")
ttlang_ensure_submodules(third-party/stablehlo)

include_directories(SYSTEM "${STABLEHLO_SOURCE_DIR}")

add_subdirectory(
  "${STABLEHLO_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/stablehlo"
  EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Shardy
# ---------------------------------------------------------------------------
set(SHARDY_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/shardy")
ttlang_ensure_submodules(third-party/shardy)

include_directories(SYSTEM "${SHARDY_SOURCE_DIR}")

add_subdirectory(
  "${SHARDY_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/shardy"
  EXCLUDE_FROM_ALL)

# Suppress warnings in third-party code
foreach(_target obj.SdyDialect obj.SdyRegister obj.SdyCommonFileUtils)
  if(TARGET ${_target})
    target_compile_options(${_target} PRIVATE
      -Wno-deprecated-declarations
      -Wno-covered-switch-default)
  endif()
endforeach()
