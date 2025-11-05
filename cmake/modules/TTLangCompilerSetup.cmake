# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Compiler and linker configuration for tt-lang.

# ccache support - automatically enable if found.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
else()
  message(STATUS "ccache not found - builds will not be cached")
endif()

# Enable compilation database for clangd support.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags
add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter --system-header-prefix=ENV{TTMLIR_TOOLCHAIN_DIR})

# Suppress redundant -U option warning on macOS when building Python extensions.
if(APPLE)
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-w")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS " -Wl,-w")
endif()

# LLD linker detection and setup
set(CMAKE_LINKER_TYPE DEFAULT)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE CLANG_VERSION_INFO ERROR_QUIET)
  string(REGEX MATCH "version ([0-9]+)\\.[0-9]+" CLANG_VERSION "${CLANG_VERSION_INFO}")
  set(CLANG_VERSION_MAJOR ${CMAKE_MATCH_1})
  set(LD_LLD_EXECUTABLE_VERSIONED "ld.lld-${CLANG_VERSION_MAJOR}")
  find_program(LLD NAMES ${LD_LLD_EXECUTABLE_VERSIONED} ld.lld)
  if(LLD)
    execute_process(COMMAND ${LLD} --version OUTPUT_VARIABLE LLD_VERSION_INFO ERROR_QUIET)
    string(REGEX MATCH "LLD ([0-9]+)\\.[0-9]+" LLD_VERSION "${LLD_VERSION_INFO}")
    set(LLD_VERSION_MAJOR ${CMAKE_MATCH_1})
    if (CLANG_VERSION_MAJOR EQUAL LLD_VERSION_MAJOR)
      message(STATUS "Using LLD linker: ${LLD}")
      set(CMAKE_LINKER_TYPE LLD)
    endif()
  endif()
endif()
