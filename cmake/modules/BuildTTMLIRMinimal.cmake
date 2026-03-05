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
# ---------------------------------------------------------------------------
set(FBS_STUB_DIR "${CMAKE_BINARY_DIR}/include/ttmlir/Target/Common")
file(MAKE_DIRECTORY "${FBS_STUB_DIR}")

# types_generated.h - provides ::tt::target enums (Arch, DataType, etc.)
file(WRITE "${FBS_STUB_DIR}/types_generated.h" [=[
// Stub: minimal flatbuffers type declarations for compilation without flatc.
#pragma once
namespace tt::target {
enum class Arch : uint32_t { Grayskull = 0, Wormhole_b0 = 1, Blackhole = 2 };
enum class DataType : uint32_t {
  Float32 = 0, Float16 = 1, BFloat16 = 2, BFP_Float8 = 3, BFP_BFloat8 = 4,
  BFP_Float4 = 5, BFP_BFloat4 = 6, BFP_Float2 = 7, BFP_BFloat2 = 8,
  UInt8 = 9, UInt16 = 10, UInt32 = 11, Int32 = 12, Bool = 15,
  Float64 = 16, Int8 = 17, Int16 = 18, Int64 = 19, UInt64 = 20
};
enum class CPURole : uint32_t { Host = 0, Device = 1 };
enum class ChipCapability : uint32_t { HostMMIO = 1 };
enum class MemorySpace : uint32_t { System = 0, SystemMMIO = 1, DeviceDRAM = 2, DeviceL1 = 3 };
enum class TensorMemoryLayout : uint32_t {
  None = 0, Interleaved = 1, SingleBank = 2, HeightSharded = 3,
  WidthSharded = 4, BlockSharded = 5
};
enum class OOBVal : uint32_t { Undef = 0, Zero = 1, One = 2, Inf = 3, NegInf = 4 };
} // namespace tt::target
]=])

# system_desc_generated.h - provides mock types for compilation
file(WRITE "${FBS_STUB_DIR}/system_desc_generated.h" [=[
// Stub: mock flatbuffers types for compilation without flatc.
// These types are never instantiated at runtime — tt-lang uses
// SystemDescAttr::getDefault() which doesn't touch flatbuffers.
#pragma once
#include "ttmlir/Target/Common/types_generated.h"
#include <cstdint>
#include <string_view>
namespace tt::target {

// Mock vector: iterators yield pointers (matching flatbuffers API).
template<typename T> struct FBVec {
  struct Iter {
    const T *operator*() const { return nullptr; }
    Iter &operator++() { return *this; }
    bool operator!=(const Iter &) const { return false; }
  };
  Iter begin() const { return {}; }
  Iter end() const { return {}; }
};
// Specialization for scalar types (flatbuffers vectors of scalars yield values).
template<> struct FBVec<uint32_t> {
  const uint32_t *begin() const { return nullptr; }
  const uint32_t *end() const { return nullptr; }
};
template<> struct FBVec<DataType> {
  const DataType *begin() const { return nullptr; }
  const DataType *end() const { return nullptr; }
};
template<> struct FBVec<ChipCapability> {
  const ChipCapability *begin() const { return nullptr; }
  const ChipCapability *end() const { return nullptr; }
};

struct Dim2d { int32_t y() const { return 0; } int32_t x() const { return 0; } };
struct FBStr { const char *c_str() const { return ""; } uint32_t size() const { return 0; } std::string_view string_view() const { return {}; } };
struct TileSize { int32_t y() const { return 0; } int32_t x() const { return 0; } };

struct CpuDesc {
  uint32_t role() const { return 0; }
  const FBStr *target_triple() const { static FBStr s; return &s; }
};

struct ChipDesc {
  Arch arch() const { return Arch::Wormhole_b0; }
  const Dim2d *grid_size() const { static Dim2d d; return &d; }
  const Dim2d *coord_translation_offsets() const { static Dim2d d; return &d; }
  uint32_t l1_size() const { return 0; }
  uint32_t num_dram_channels() const { return 0; }
  uint64_t dram_channel_size() const { return 0; }
  uint32_t noc_l1_address_align_bytes() const { return 0; }
  uint32_t pcie_address_align_bytes() const { return 0; }
  uint32_t noc_dram_address_align_bytes() const { return 0; }
  uint32_t l1_unreserved_base() const { return 0; }
  uint32_t erisc_l1_unreserved_base() const { return 0; }
  uint32_t dram_unreserved_base() const { return 0; }
  uint32_t dram_unreserved_end() const { return 0; }
  const FBVec<DataType> *supported_data_types() const { static FBVec<DataType> v; return &v; }
  const FBVec<TileSize> *supported_tile_sizes() const { static FBVec<TileSize> v; return &v; }
  uint32_t dst_physical_size_tiles() const { return 0; }
  uint32_t num_cbs() const { return 0; }
  uint32_t num_compute_threads() const { return 0; }
  uint32_t num_datamovement_threads() const { return 0; }
};

struct ChipCoord {
  uint32_t rack() const { return 0; }
  uint32_t shelf() const { return 0; }
  uint32_t y() const { return 0; }
  uint32_t x() const { return 0; }
};

struct EthCoreCoord { int32_t y() const { return 0; } int32_t x() const { return 0; } };
struct ChipChannel {
  uint32_t device_id0() const { return 0; }
  EthCoreCoord ethernet_core_coord0() const { return {}; }
  uint32_t device_id1() const { return 0; }
  EthCoreCoord ethernet_core_coord1() const { return {}; }
};

struct SystemDesc {
  const FBVec<CpuDesc> *cpu_descs() const { static FBVec<CpuDesc> v; return &v; }
  const FBVec<ChipDesc> *chip_descs() const { static FBVec<ChipDesc> v; return &v; }
  const FBVec<uint32_t> *chip_desc_indices() const { static FBVec<uint32_t> v; return &v; }
  const FBVec<ChipCapability> *chip_capabilities() const { static FBVec<ChipCapability> v; return &v; }
  const FBVec<ChipCoord> *chip_coords() const { static FBVec<ChipCoord> v; return &v; }
  const FBVec<ChipChannel> *chip_channels() const { static FBVec<ChipChannel> v; return &v; }
};

struct SystemDescRoot {
  const FBStr *schema_hash() const { static FBStr s; return &s; }
  const SystemDesc *system_desc() const { static SystemDesc s; return &s; }
};

inline const SystemDescRoot *GetSizePrefixedSystemDescRoot(const void *) { return nullptr; }
} // namespace tt::target
]=])

# system_desc_bfbs_hash_generated.h - provides schema hash constant
file(WRITE "${FBS_STUB_DIR}/system_desc_bfbs_hash_generated.h" [=[
// Stub: schema hash not available without flatbuffers.
#pragma once
#include <string_view>
namespace tt::target::common {
inline constexpr std::string_view system_desc_bfbs_schema_hash = "";
} // namespace tt::target::common
]=])

# debug_info_generated.h - empty stub
file(WRITE "${FBS_STUB_DIR}/debug_info_generated.h" [=[
// Stub: debug info not available without flatbuffers.
#pragma once
]=])

# version_generated.h - empty stub
file(WRITE "${FBS_STUB_DIR}/version_generated.h" [=[
// Stub: version info not available without flatbuffers.
#pragma once
]=])

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
      -Wno-error=switch
      -Wno-error=covered-switch-default)
  endif()
endforeach()
