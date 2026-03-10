# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BuildTTMLIRMinimal.cmake - Build only the tt-mlir dialects needed by tt-lang.
#
# Uses MLIR's standard CMake infrastructure (add_mlir_dialect, mlir_tablegen,
# add_mlir_dialect_library, etc.) to compile tt-mlir sources directly from
# the submodule.
#
# TableGen is handled by add_subdirectory() on tt-mlir's include directories,
# which ensures generated .inc files end up at the canonical include paths.
# C++ library builds use add_mlir_dialect_library() in lib/ttmlir-minimal/.
#
# Components built:
#   - TTCore dialect (IR + Transforms)
#   - TTMetal dialect (IR only)
#   - TTKernel dialect (IR + Transforms)
#   - TTKernelToEmitC conversion
#   - TTKernelToCpp target translation
#   - CAPI layer for Python bindings

set(TT_MLIR_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/tt-mlir")
ttlang_ensure_submodules(third-party/tt-mlir)

set(TT_MLIR_INCLUDE_DIR "${TT_MLIR_SOURCE_DIR}/include")

# Include paths: tt-mlir source headers + generated headers.
# Generated .inc files land under ${CMAKE_BINARY_DIR}/include/ via the
# add_subdirectory() BINARY_DIR settings below.
include_directories(SYSTEM "${TT_MLIR_INCLUDE_DIR}")
include_directories("${CMAKE_BINARY_DIR}/include")

# TableGen needs tt-mlir include path for .td file resolution.
list(APPEND MLIR_TABLEGEN_FLAGS "-I${TT_MLIR_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Aggregate targets used by tt-mlir's include CMakeLists.txt files.
# In a full LLVM source build these exist; we create them if absent (which is
# the case when using a pre-built MLIR install via find_package).
# ---------------------------------------------------------------------------
if(NOT TARGET mlir-headers)
  add_custom_target(mlir-headers)
endif()
if(NOT TARGET mlir-doc)
  add_custom_target(mlir-doc)
endif()

# ---------------------------------------------------------------------------
# TableGen: process tt-mlir's include directories directly.
#
# Each add_subdirectory() uses BINARY_DIR set to mirror the source layout
# under ${CMAKE_BINARY_DIR}/include/, so mlir_tablegen() generates .inc files
# at the expected include paths, e.g.:
#   ${CMAKE_BINARY_DIR}/include/ttmlir/Dialect/TTCore/IR/TTCoreOps.h.inc
#
# This is the standard LLVM pattern: source .td files live in include/,
# tablegen runs from that directory, and generated headers go to the
# corresponding binary dir.
# ---------------------------------------------------------------------------

# TTCore
add_subdirectory(
  "${TT_MLIR_INCLUDE_DIR}/ttmlir/Dialect/TTCore/IR"
  "${CMAKE_BINARY_DIR}/include/ttmlir/Dialect/TTCore/IR")
add_subdirectory(
  "${TT_MLIR_INCLUDE_DIR}/ttmlir/Dialect/TTCore/Transforms"
  "${CMAKE_BINARY_DIR}/include/ttmlir/Dialect/TTCore/Transforms")

# TTMetal
add_subdirectory(
  "${TT_MLIR_INCLUDE_DIR}/ttmlir/Dialect/TTMetal/IR"
  "${CMAKE_BINARY_DIR}/include/ttmlir/Dialect/TTMetal/IR")
add_subdirectory(
  "${TT_MLIR_INCLUDE_DIR}/ttmlir/Dialect/TTMetal/Transforms"
  "${CMAKE_BINARY_DIR}/include/ttmlir/Dialect/TTMetal/Transforms")

# TTKernel
add_subdirectory(
  "${TT_MLIR_INCLUDE_DIR}/ttmlir/Dialect/TTKernel/IR"
  "${CMAKE_BINARY_DIR}/include/ttmlir/Dialect/TTKernel/IR")
add_subdirectory(
  "${TT_MLIR_INCLUDE_DIR}/ttmlir/Dialect/TTKernel/Transforms"
  "${CMAKE_BINARY_DIR}/include/ttmlir/Dialect/TTKernel/Transforms")

# Conversion passes (TTMLIR_ENABLE_STABLEHLO is unset so stablehlo branch is skipped)
add_subdirectory(
  "${TT_MLIR_INCLUDE_DIR}/ttmlir/Conversion"
  "${CMAKE_BINARY_DIR}/include/ttmlir/Conversion")

# ---------------------------------------------------------------------------
# Flatbuffers stub headers
#
# TTCoreOpsTypes.cpp includes ttmlir/Target/Common/Target.h which pulls in
# flatbuffers-generated headers (*_generated.h). We don't build flatbuffers;
# instead we provide minimal stubs with just enough declarations for the
# upstream code to compile. Methods that depend on flatbuffers at runtime
# (SystemDescAttr::getFromPath, getFromBuffer) will fail gracefully.
#
# Stubs live in cmake/stubs/flatbuffers/ as proper .h files.
# ---------------------------------------------------------------------------
file(COPY "${CMAKE_SOURCE_DIR}/cmake/stubs/flatbuffers/"
     DESTINATION "${CMAKE_BINARY_DIR}/include/ttmlir/Target/Common")

# ---------------------------------------------------------------------------
# LLK generated headers (for TTKernelToCpp target translation)
#
# Handled manually because tt-mlir's CMakeLists.txt uses ${CMAKE_SOURCE_DIR}
# and ${PROJECT_SOURCE_DIR} paths that don't resolve in our project tree.
# ---------------------------------------------------------------------------
set(LLK_SOURCE_DIR "${TT_MLIR_INCLUDE_DIR}/ttmlir/Target/TTKernel/LLKs")
set(LLK_GEN_DIR "${CMAKE_BINARY_DIR}/include/ttmlir/Target/TTKernel/LLKs")
file(MAKE_DIRECTORY "${LLK_GEN_DIR}")

set(LLK_HEADERS
  "${LLK_SOURCE_DIR}/experimental_tilize_llks.h"
  "${LLK_SOURCE_DIR}/experimental_untilize_llks.h"
  "${LLK_SOURCE_DIR}/experimental_pack_untilize_llks.h"
  "${LLK_SOURCE_DIR}/experimental_invoke_sfpi_llks.h"
  "${LLK_SOURCE_DIR}/experimental_dataflow_api.h"
  "${LLK_SOURCE_DIR}/experimental_matmul_llks.h"
  "${LLK_SOURCE_DIR}/experimental_padding_llks.h"
  "${LLK_SOURCE_DIR}/experimental_coord_translation.h"
  "${LLK_SOURCE_DIR}/experimental_fabric_topology_info.h"
  "${LLK_SOURCE_DIR}/experimental_fabric_1d_routing.h"
  "${LLK_SOURCE_DIR}/experimental_fabric_2d_routing.h"
  "${LLK_SOURCE_DIR}/experimental_fabric_api.h"
)

set(GENERATED_LLK_HEADERS)
foreach(llk_header ${LLK_HEADERS})
  get_filename_component(header_name ${llk_header} NAME_WE)
  set(output_file "${LLK_GEN_DIR}/${header_name}_generated.h")
  add_custom_command(
    OUTPUT ${output_file}
    COMMAND ${CMAKE_COMMAND}
      -DINPUT_FILE=${llk_header}
      -DOUTPUT_FILE=${output_file}
      -DVARIABLE_NAME=${header_name}_generated
      -P "${TT_MLIR_SOURCE_DIR}/cmake/modules/GenerateRawStringHeader.cmake"
    DEPENDS ${llk_header}
    COMMENT "Generating LLK header ${header_name}_generated.h"
    VERBATIM
  )
  list(APPEND GENERATED_LLK_HEADERS ${output_file})
endforeach()

add_custom_target(TTKernelGeneratedLLKHeaders DEPENDS ${GENERATED_LLK_HEADERS})

# ---------------------------------------------------------------------------
# C++ libraries
# ---------------------------------------------------------------------------
add_subdirectory(lib/ttmlir-minimal)

# Suppress warnings in tt-mlir code that we cannot fix (submodule).
# add_mlir_dialect_library creates obj.* OBJECT library targets that hold
# the actual compile rules, so we must set options on those.
set(_TTMLIR_TARGETS
  obj.MLIRTTCoreDialect obj.MLIRTTTransforms obj.MLIRTTMetalDialect
  obj.MLIRTTKernelDialect obj.MLIRTTKernelTransforms
  obj.TTMLIRTTKernelToEmitC obj.TTKernelTargetCpp)
foreach(_target ${_TTMLIR_TARGETS})
  if(TARGET ${_target})
    target_compile_options(${_target} PRIVATE
      -Wno-deprecated-declarations
      -Wno-switch
      -Wno-covered-switch-default)
  endif()
endforeach()
