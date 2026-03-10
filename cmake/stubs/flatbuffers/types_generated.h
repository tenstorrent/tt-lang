// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Stub: minimal flatbuffers type declarations for compilation without flatc.
// These provide the tt::target enums that tt-mlir headers reference.
//
// Stubbed from flatbuffers 24.3.25 as used by tt-mlir/tt-metal.
//
// TODO: Remove these stubs when tt-mlir decouples TTCore dialect types from
// flatbuffers-generated headers (TTCoreOpsTypes.cpp includes Target.h which
// transitively pulls in *_generated.h).
#pragma once
namespace tt::target {
enum class Arch : uint32_t { Grayskull = 0, Wormhole_b0 = 1, Blackhole = 2 };
enum class DataType : uint32_t {
  Float32 = 0,
  Float16 = 1,
  BFloat16 = 2,
  BFP_Float8 = 3,
  BFP_BFloat8 = 4,
  BFP_Float4 = 5,
  BFP_BFloat4 = 6,
  BFP_Float2 = 7,
  BFP_BFloat2 = 8,
  UInt8 = 9,
  UInt16 = 10,
  UInt32 = 11,
  Int32 = 12,
  Bool = 15,
  Float64 = 16,
  Int8 = 17,
  Int16 = 18,
  Int64 = 19,
  UInt64 = 20
};
enum class CPURole : uint32_t { Host = 0, Device = 1 };
enum class ChipCapability : uint32_t { HostMMIO = 1 };
enum class MemorySpace : uint32_t {
  System = 0,
  SystemMMIO = 1,
  DeviceDRAM = 2,
  DeviceL1 = 3
};
enum class TensorMemoryLayout : uint32_t {
  None = 0,
  Interleaved = 1,
  SingleBank = 2,
  HeightSharded = 3,
  WidthSharded = 4,
  BlockSharded = 5
};
enum class OOBVal : uint32_t {
  Undef = 0,
  Zero = 1,
  One = 2,
  Inf = 3,
  NegInf = 4
};
} // namespace tt::target
