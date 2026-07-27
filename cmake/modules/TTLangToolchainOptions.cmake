# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

option(TTLANG_USE_TOOLCHAIN "Use pre-built LLVM from ttlang toolchain" OFF)
option(TTLANG_BUILD_TOOLCHAIN
  "Build LLVM and tt-metal into a reusable toolchain directory" OFF)
option(TTLANG_INSTALL_RUNTIME_REQUIREMENTS
  "Install tt-lang runtime requirements while configuring LLVM" ON)
set(TTLANG_TOOLCHAIN_COMPONENT "all" CACHE STRING
  "Toolchain component to configure: all, llvm, or tt-metal")
set_property(CACHE TTLANG_TOOLCHAIN_COMPONENT PROPERTY STRINGS all llvm tt-metal)
set(_TTLANG_TOOLCHAIN_COMPONENTS all llvm tt-metal)
if(NOT TTLANG_TOOLCHAIN_COMPONENT IN_LIST _TTLANG_TOOLCHAIN_COMPONENTS)
  message(FATAL_ERROR
    "TTLANG_TOOLCHAIN_COMPONENT must be one of: all, llvm, tt-metal")
endif()

# Per-component toolchain overrides follow TTLANG_USE_TOOLCHAIN by default but
# can be set independently to rebuild tt-metal while consuming pre-built LLVM.
option(TTLANG_USE_TOOLCHAIN_TTMETAL
  "Use pre-built tt-metal from ttlang toolchain (defaults to TTLANG_USE_TOOLCHAIN)"
  ${TTLANG_USE_TOOLCHAIN})

# An external tt-metal install uses the install-ttmetal.sh layout and takes
# precedence over TTLANG_USE_TOOLCHAIN_TTMETAL. BuildTTMetal.cmake documents
# the required directory structure.
set(TTLANG_EXTERNAL_TT_METAL_DIR "" CACHE PATH
  "Path to an existing tt-metal installation; overrides the toolchain or submodule build.")
if(NOT DEFINED TTLANG_TOOLCHAIN_DIR)
  if(DEFINED ENV{TTLANG_TOOLCHAIN_DIR})
    set(TTLANG_TOOLCHAIN_DIR "$ENV{TTLANG_TOOLCHAIN_DIR}" CACHE PATH
      "tt-lang toolchain directory")
  elseif(TTLANG_USE_TOOLCHAIN)
    set(TTLANG_TOOLCHAIN_DIR "/opt/ttlang-toolchain" CACHE PATH
      "tt-lang toolchain directory" FORCE)
  endif()
endif()
