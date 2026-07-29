// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/DFBAllocation.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTCore/IR/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <optional>

namespace mlir::tt::ttl {

FailureOr<uint64_t> DFBAllocationSummary::getTotalBytesWithMinimumAllocation(
    int64_t dfbIndex, uint64_t minimumBytes) const {
  uint64_t currentBytes = 0;
  auto allocation = allocations.find(dfbIndex);
  if (allocation != allocations.end()) {
    currentBytes = allocation->second.bytes;
  }
  if (minimumBytes <= currentBytes) {
    return totalBytes;
  }
  std::optional<uint64_t> total =
      llvm::checkedAddUnsigned(totalBytes, minimumBytes - currentBytes);
  if (!total) {
    return failure();
  }
  return *total;
}

uint64_t getL1DFBBudgetBytes(ModuleOp moduleOp, uint64_t overrideBytes) {
  if (overrideBytes > 0) {
    return overrideBytes;
  }

  auto systemDesc = moduleOp->getAttrOfType<mlir::tt::ttcore::SystemDescAttr>(
      mlir::tt::ttcore::SystemDescAttr::name);
  if (!systemDesc) {
    return kDefaultL1DFBBudgetBytes;
  }

  auto deviceOp = mlir::tt::ttcore::lookupDeviceOp(
      moduleOp, mlir::tt::ttcore::getDefaultDeviceName());
  if (!deviceOp) {
    return kDefaultL1DFBBudgetBytes;
  }

  auto chipIds = deviceOp.getDeviceAttr().getChipIds();
  if (chipIds.empty()) {
    return kDefaultL1DFBBudgetBytes;
  }

  return *llvm::min_element(llvm::map_range(chipIds, [&](unsigned chipId) {
    return systemDesc.getChipDesc(chipId).getUsableL1Size();
  }));
}

FailureOr<uint64_t>
getDFBAllocationSizeBytes(CircularBufferType dfbType) {
  uint64_t elementCount = 1;
  for (int64_t dimension : dfbType.getShape()) {
    if (dimension <= 0) {
      return failure();
    }
    auto count =
        llvm::checkedMulUnsigned(elementCount, static_cast<uint64_t>(dimension));
    if (!count) {
      return failure();
    }
    elementCount = *count;
  }
  if (dfbType.getBlockCount() <= 0) {
    return failure();
  }
  auto totalElements = llvm::checkedMulUnsigned(
      elementCount, static_cast<uint64_t>(dfbType.getBlockCount()));
  if (!totalElements) {
    return failure();
  }

  Type elementType = dfbType.getElementType();
  uint64_t elementBytes = 0;
  if (auto tileType = dyn_cast<mlir::tt::ttcore::TileType>(elementType)) {
    elementBytes = tileType.getSizeBytes();
  } else {
    elementBytes =
        mlir::tt::ttcore::TileType::get(elementType).getSizeBytes();
  }
  std::optional<uint64_t> allocationBytes =
      llvm::checkedMulUnsigned(*totalElements, elementBytes);
  if (!allocationBytes) {
    return failure();
  }
  return *allocationBytes;
}

FailureOr<DFBAllocationSummary>
getDFBAllocationSummary(ModuleOp moduleOp) {
  DFBAllocationSummary summary;
  WalkResult walkResult = moduleOp.walk([&](BindCBOp bindOp) {
    FailureOr<uint64_t> bytes = getDFBAllocationSizeBytes(
        cast<CircularBufferType>(bindOp.getResult().getType()));
    if (failed(bytes)) {
      return WalkResult::interrupt();
    }

    int64_t dfbIndex = bindOp.getCbIndex().getSExtValue();
    DFBIndexAllocation &allocation = summary.allocations[dfbIndex];
    if (*bytes > allocation.bytes) {
      allocation = {*bytes, bindOp};
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  for (const auto &entry : summary.allocations) {
    auto total =
        llvm::checkedAddUnsigned(summary.totalBytes, entry.second.bytes);
    if (!total) {
      return failure();
    }
    summary.totalBytes = *total;
  }
  return summary;
}

} // namespace mlir::tt::ttl
