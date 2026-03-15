// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Stub: mock flatbuffers types for compilation without flatc.
// These types are never instantiated at runtime — tt-lang uses
// SystemDescAttr::getDefault() which doesn't touch flatbuffers.
#pragma once
#include "ttmlir/Target/Common/types_generated.h"
#include <cassert>
#include <cstdint>
#include <string_view>
namespace tt::target {

// Mock vector: iterators yield pointers (matching flatbuffers API).
template <typename T>
struct FBVec {
  struct Iter {
    const T *operator*() const { return nullptr; }
    Iter &operator++() { return *this; }
    bool operator!=(const Iter &) const { return false; }
  };
  Iter begin() const { return {}; }
  Iter end() const { return {}; }
};
// Specialization for scalar types (flatbuffers vectors of scalars yield
// values).
template <>
struct FBVec<uint32_t> {
  const uint32_t *begin() const { return nullptr; }
  const uint32_t *end() const { return nullptr; }
};
template <>
struct FBVec<DataType> {
  const DataType *begin() const { return nullptr; }
  const DataType *end() const { return nullptr; }
};
template <>
struct FBVec<ChipCapability> {
  const ChipCapability *begin() const { return nullptr; }
  const ChipCapability *end() const { return nullptr; }
};

struct Dim2d {
  int32_t y() const { return 0; }
  int32_t x() const { return 0; }
};
struct FBStr {
  const char *c_str() const { return ""; }
  uint32_t size() const { return 0; }
  std::string_view string_view() const { return {}; }
};
struct TileSize {
  int32_t y() const { return 0; }
  int32_t x() const { return 0; }
};

struct CpuDesc {
  uint32_t role() const { return 0; }
  const FBStr *target_triple() const {
    static FBStr s;
    return &s;
  }
};

struct ChipDesc {
  Arch arch() const { return Arch::Wormhole_b0; }
  const Dim2d *grid_size() const {
    static Dim2d d;
    return &d;
  }
  const Dim2d *coord_translation_offsets() const {
    static Dim2d d;
    return &d;
  }
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
  const FBVec<DataType> *supported_data_types() const {
    static FBVec<DataType> v;
    return &v;
  }
  const FBVec<TileSize> *supported_tile_sizes() const {
    static FBVec<TileSize> v;
    return &v;
  }
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

struct EthCoreCoord {
  int32_t y() const { return 0; }
  int32_t x() const { return 0; }
};
struct ChipChannel {
  uint32_t device_id0() const { return 0; }
  EthCoreCoord ethernet_core_coord0() const { return {}; }
  uint32_t device_id1() const { return 0; }
  EthCoreCoord ethernet_core_coord1() const { return {}; }
};

struct SystemDesc {
  const FBVec<CpuDesc> *cpu_descs() const {
    static FBVec<CpuDesc> v;
    return &v;
  }
  const FBVec<ChipDesc> *chip_descs() const {
    static FBVec<ChipDesc> v;
    return &v;
  }
  const FBVec<uint32_t> *chip_desc_indices() const {
    static FBVec<uint32_t> v;
    return &v;
  }
  const FBVec<ChipCapability> *chip_capabilities() const {
    static FBVec<ChipCapability> v;
    return &v;
  }
  const FBVec<ChipCoord> *chip_coords() const {
    static FBVec<ChipCoord> v;
    return &v;
  }
  const FBVec<ChipChannel> *chip_channels() const {
    static FBVec<ChipChannel> v;
    return &v;
  }
};

struct SystemDescRoot {
  const FBStr *schema_hash() const {
    static FBStr s;
    return &s;
  }
  const SystemDesc *system_desc() const {
    static SystemDesc s;
    return &s;
  }
};

inline const SystemDescRoot *GetSizePrefixedSystemDescRoot(const void *) {
  assert(false && "flatbuffers stub: GetSizePrefixedSystemDescRoot not "
                  "implemented — use SystemDescAttr::getDefault() instead");
  return nullptr;
}
} // namespace tt::target
