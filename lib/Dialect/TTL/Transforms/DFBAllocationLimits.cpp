// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBAllocationLimits.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTCore/IR/Utils.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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

FailureOr<uint64_t> getDFBAllocationSizeBytes(CircularBufferType type,
                                              std::string &failureReason) {
  FailureOr<uint64_t> pagesPerBlock = getDFBPagesPerBlock(type);
  if (failed(pagesPerBlock)) {
    failureReason = "DFB dimensions are not representable";
    return failure();
  }
  if (type.getBlockCount() <= 0) {
    failureReason = "DFB block count must be positive";
    return failure();
  }
  Type elementType = type.getElementType();
  uint64_t elementBytes;
  if (auto tileType = dyn_cast<ttcore::TileType>(elementType)) {
    elementBytes = tileType.getSizeBytes();
  } else {
    std::optional<ttcore::DataType> dataType =
        ttcore::elementTypeToDataTypeImpl(elementType);
    if (!dataType) {
      llvm::raw_string_ostream message(failureReason);
      message << "cannot determine DFB page size for element type "
              << elementType;
      return failure();
    }
    elementBytes =
        ttcore::TileType::get(elementType.getContext(),
                              ttcore::TileType::getDefaultShape(), *dataType)
            .getSizeBytes();
  }
  std::optional<uint64_t> totalPages = llvm::checkedMulUnsigned(
      *pagesPerBlock, static_cast<uint64_t>(type.getBlockCount()));
  if (!totalPages) {
    failureReason = "DFB allocation size is not representable";
    return failure();
  }
  std::optional<uint64_t> allocationBytes =
      llvm::checkedMulUnsigned(*totalPages, elementBytes);
  if (!allocationBytes) {
    failureReason = "DFB allocation size is not representable";
    return failure();
  }
  return *allocationBytes;
}

FailureOr<bool> DFBAllocationFootprint::add(int64_t physicalIndex,
                                            CircularBufferType type,
                                            std::string &failureReason) {
  FailureOr<uint64_t> allocationBytes =
      getDFBAllocationSizeBytes(type, failureReason);
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

FailureOr<uint64_t> DFBAllocationFootprint::getTotalBytes() const {
  uint64_t totalBytes = 0;
  for (uint64_t allocationBytes : llvm::make_second_range(maxBytesByIndex)) {
    std::optional<uint64_t> updatedTotal =
        llvm::checkedAddUnsigned(totalBytes, allocationBytes);
    if (!updatedTotal) {
      return failure();
    }
    totalBytes = *updatedTotal;
  }
  return totalBytes;
}

FailureOr<uint64_t> DFBAllocationFootprint::getTotalBytesWithMinimumAllocations(
    const llvm::DenseMap<int64_t, uint64_t> &minimumBytesByIndex) const {
  uint64_t totalBytes = 0;
  for (const auto &[physicalIndex, allocationBytes] : maxBytesByIndex) {
    uint64_t minimumBytes = minimumBytesByIndex.lookup(physicalIndex);
    std::optional<uint64_t> updatedTotal = llvm::checkedAddUnsigned(
        totalBytes, std::max(allocationBytes, minimumBytes));
    if (!updatedTotal) {
      return failure();
    }
    totalBytes = *updatedTotal;
  }
  for (const auto &[physicalIndex, minimumBytes] : minimumBytesByIndex) {
    if (maxBytesByIndex.contains(physicalIndex)) {
      continue;
    }
    std::optional<uint64_t> updatedTotal =
        llvm::checkedAddUnsigned(totalBytes, minimumBytes);
    if (!updatedTotal) {
      return failure();
    }
    totalBytes = *updatedTotal;
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

FailureOr<DFBAllocationFootprint> getDFBAllocationFootprint(ModuleOp module) {
  DFBAllocationFootprint footprint;
  WalkResult walkResult = module.walk([&](BindCBOp bindOp) {
    std::string failureReason;
    FailureOr<bool> increased = footprint.add(
        bindOp.getCbIndex().getSExtValue(),
        cast<CircularBufferType>(bindOp.getResult().getType()), failureReason);
    return failed(increased) ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }
  return footprint;
}

uint64_t getUsableDFBL1Bytes(ModuleOp module,
                             std::optional<uint64_t> overrideBytes) {
  if (overrideBytes) {
    return *overrideBytes;
  }
  return tryBudgetFromModule(module).value_or(kFallbackUsableL1Bytes);
}

} // namespace mlir::tt::ttl
