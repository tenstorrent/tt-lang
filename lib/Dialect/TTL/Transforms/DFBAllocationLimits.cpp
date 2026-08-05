// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBAllocationLimits.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTCore/IR/Utils.h"

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <optional>

namespace mlir::tt::ttl {

namespace {

constexpr uint64_t kFallbackUsableL1Bytes = static_cast<uint64_t>(1432 * 1024);

std::optional<uint64_t> tryBudgetFromModule(ModuleOp module) {
  auto systemDesc = module->getAttrOfType<ttcore::SystemDescAttr>(
      ttcore::SystemDescAttr::name);
  if (!systemDesc) {
    return std::nullopt;
  }
  auto device = ttcore::lookupDeviceOp(module, ttcore::getDefaultDeviceName());
  if (!device) {
    return std::nullopt;
  }
  auto chipIds = device.getDeviceAttr().getChipIds();
  if (chipIds.empty()) {
    return std::nullopt;
  }
  return *llvm::min_element(llvm::map_range(chipIds, [&](unsigned chipId) {
    return systemDesc.getChipDesc(chipId).getUsableL1Size();
  }));
}

} // namespace

FailureOr<uint64_t> getDFBAllocationSizeBytes(CircularBufferType type) {
  int64_t totalElements = type.getTotalElements();
  if (totalElements < 0) {
    return failure();
  }
  Type elementType = type.getElementType();
  uint64_t elementBytes;
  if (auto tileType = dyn_cast<ttcore::TileType>(elementType)) {
    elementBytes = tileType.getSizeBytes();
  } else {
    elementBytes = ttcore::TileType::get(elementType).getSizeBytes();
  }
  return static_cast<uint64_t>(totalElements) * elementBytes;
}

FailureOr<bool> DFBAllocationFootprint::add(int64_t physicalIndex,
                                            CircularBufferType type) {
  FailureOr<uint64_t> allocationBytes = getDFBAllocationSizeBytes(type);
  if (failed(allocationBytes)) {
    return failure();
  }
  auto indexIt = maxBytesByIndex.find(physicalIndex);
  if (indexIt != maxBytesByIndex.end() && indexIt->second >= *allocationBytes) {
    return false;
  }
  maxBytesByIndex[physicalIndex] = *allocationBytes;
  return true;
}

uint64_t DFBAllocationFootprint::getTotalBytes() const {
  uint64_t totalBytes = 0;
  for (uint64_t allocationBytes : llvm::make_second_range(maxBytesByIndex)) {
    totalBytes += allocationBytes;
  }
  return totalBytes;
}

uint64_t DFBAllocationFootprint::getBytes(int64_t physicalIndex) const {
  auto indexIt = maxBytesByIndex.find(physicalIndex);
  assert(indexIt != maxBytesByIndex.end() &&
         "physical index must be present in the footprint");
  return indexIt->second;
}

llvm::SmallVector<int64_t, kMaxCircularBuffers>
DFBAllocationFootprint::getSortedPhysicalIndices() const {
  llvm::SmallVector<int64_t, kMaxCircularBuffers> physicalIndices;
  physicalIndices.reserve(maxBytesByIndex.size());
  for (int64_t physicalIndex : llvm::make_first_range(maxBytesByIndex)) {
    physicalIndices.push_back(physicalIndex);
  }
  llvm::sort(physicalIndices);
  return physicalIndices;
}

uint64_t getUsableDFBL1Bytes(ModuleOp module,
                             std::optional<uint64_t> overrideBytes) {
  if (overrideBytes) {
    return *overrideBytes;
  }
  return tryBudgetFromModule(module).value_or(kFallbackUsableL1Bytes);
}

} // namespace mlir::tt::ttl
