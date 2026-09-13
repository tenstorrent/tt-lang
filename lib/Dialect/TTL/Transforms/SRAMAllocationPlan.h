// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATIONPLAN_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATIONPLAN_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"

#include <optional>

namespace mlir::tt::ttl {

inline constexpr uint64_t kSRAMControlRecordBytes = 2 * sizeof(uint32_t);

/// One logical DFB; storageIndex identifies its shared storage owner.
struct SRAMRegion {
  int64_t logicalId;
  CircularBufferType type;
  TensorBackingAttr tensorBacking;
  DFBAllocationGroupAttr allocationGroup;
  LaunchNodeDomain launchDomain;
  uint64_t pages;
  uint64_t pageBytes;
  uint64_t capacityPages;
  uint64_t allocationBytes;
  unsigned storageIndex = 0;
  SmallVector<BindCBOp> declarations;
};

/// One control record and its arena payload; members index plan regions.
struct SRAMStorage {
  uint64_t capacityPages = 0;
  uint64_t allocationBytes = 0;
  uint64_t offset = 0;
  uint64_t stateOffset = 0;
  SmallVector<unsigned> members;
};

/// A worker core's payload layout; absent payloads retain no arena extent.
struct SRAMCoreLayout {
  LaunchNodeCoord node;
  llvm::SmallVector<std::optional<uint64_t>> payloadOffsets;
  uint64_t arenaBytes;
  unsigned domain;
};

/// Validated per-core placement, consumed before any IR mutation.
struct SRAMAllocationPlan {
  SmallVector<SRAMRegion> regions;
  SmallVector<SRAMStorage> storage;
  uint64_t arenaBytes;
  llvm::SmallVector<SRAMCoreLayout> coreLayouts;
};

} // namespace mlir::tt::ttl
#endif
