// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBAllocationLimits.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"

#include "ttlang/Dialect/TTL/Transforms/PipeConstants.h"

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTCore/IR/Utils.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Target/TargetInfo.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <optional>

namespace mlir::tt::ttl {

namespace {

constexpr uint64_t kFallbackUsableL1Bytes = static_cast<uint64_t>(1432 * 1024);
constexpr uint64_t kGlobalSemaphorePayloadBytes = 4;
constexpr uint64_t kDFBReconfigurationWordsPerCore = 264;
constexpr uint64_t kDFBReconfigurationWordBytes = 4;

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

FailureOr<uint64_t> getL1AllocationSizeBytes(ModuleOp module,
                                             uint64_t payloadBytes) {
  if (payloadBytes == 0) {
    return 0;
  }
  std::string failureReason;
  FailureOr<uint64_t> allocationQuantum =
      resolveTargetL1AllocationQuantumBytes(module, failureReason);
  if (failed(allocationQuantum) || *allocationQuantum == 0) {
    return failure();
  }
  std::optional<uint64_t> roundedNumerator =
      llvm::checkedAddUnsigned(payloadBytes, *allocationQuantum - 1);
  if (!roundedNumerator) {
    return failure();
  }
  return (*roundedNumerator / *allocationQuantum) * *allocationQuantum;
}

LogicalResult collectSynchronizedDFBResets(
    ModuleOp module, SmallVectorImpl<SynchronizedDFBResetAttr> &resets) {
  resets.clear();
  llvm::MapVector<int64_t, SynchronizedDFBResetAttr> resetByOrdinal;
  Operation *invalidReset = nullptr;
  module.walk([&](Operation *operation) -> WalkResult {
    SynchronizedDFBResetAttr reset;
    if (auto selected = dyn_cast<ResetDFBsOp>(operation)) {
      reset = selected.getReset();
    } else if (auto all = dyn_cast<ResetAllDFBsOp>(operation)) {
      reset = all.getReset();
    } else {
      return WalkResult::advance();
    }
    auto [resetIt, inserted] =
        resetByOrdinal.try_emplace(reset.getOrdinal(), reset);
    if (!inserted && resetIt->second != reset) {
      invalidReset = operation;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (invalidReset) {
    invalidReset->emitOpError(
        "reset ordinal identifies inconsistent participant sets");
    return failure();
  }

  resets.reserve(resets.size() + resetByOrdinal.size());
  for (const auto &entry : resetByOrdinal) {
    resets.push_back(entry.second);
  }
  llvm::sort(resets,
             [](SynchronizedDFBResetAttr lhs, SynchronizedDFBResetAttr rhs) {
               return lhs.getOrdinal() < rhs.getOrdinal();
             });
  return success();
}

FailureOr<uint64_t> getSynchronizedDFBResetStateBytes(ModuleOp module) {
  SmallVector<SynchronizedDFBResetAttr> resets;
  if (failed(collectSynchronizedDFBResets(module, resets))) {
    return failure();
  }
  std::optional<uint64_t> stateBytes =
      llvm::checkedMulUnsigned(static_cast<uint64_t>(resets.size()),
                               static_cast<uint64_t>(kDFBResetStateBytes));
  if (!stateBytes) {
    module.emitOpError("DFB reset synchronization state is not representable");
    return failure();
  }
  return *stateBytes;
}

LogicalResult validateDFBReconfigurationTarget(ModuleOp module) {
  DFBReconfigurationOp firstBoundary;
  module.walk([&](DFBReconfigurationOp boundary) -> WalkResult {
    firstBoundary = boundary;
    return WalkResult::interrupt();
  });
  if (!firstBoundary) {
    return success();
  }

  std::string failureReason;
  FailureOr<std::optional<ttcore::Arch>> targetArch =
      resolveTargetArch(module, failureReason);
  if (failed(targetArch)) {
    module.emitOpError(failureReason);
    return failure();
  }
  if (!*targetArch || **targetArch == ttcore::Arch::Blackhole) {
    return success();
  }
  firstBoundary.emitOpError()
      << "is supported only for Blackhole; selected target is "
      << ttcore::ArchAttr::get(module.getContext(), **targetArch);
  return failure();
}

FailureOr<uint64_t> getDFBReconfigurationStateBytes(ModuleOp module) {
  llvm::DenseSet<int64_t> boundaryOrdinals;
  module.walk([&](DFBReconfigurationOp reconfiguration) {
    boundaryOrdinals.insert(reconfiguration.getBoundary().getOrdinal());
  });
  std::optional<uint64_t> stateBytes = llvm::checkedMulUnsigned(
      static_cast<uint64_t>(boundaryOrdinals.size()),
      kDFBReconfigurationWordsPerCore * kDFBReconfigurationWordBytes);
  if (!stateBytes) {
    module.emitOpError("DFB reconfiguration state is not representable");
    return failure();
  }
  return *stateBytes;
}

FailureOr<uint64_t> getDFBReconfigurationStateAllocationBytes(ModuleOp module) {
  FailureOr<uint64_t> stateBytes = getDFBReconfigurationStateBytes(module);
  if (failed(stateBytes) || *stateBytes == 0) {
    return stateBytes;
  }
  constexpr uint64_t payloadBytesPerBoundary =
      kDFBReconfigurationWordsPerCore * kDFBReconfigurationWordBytes;
  FailureOr<uint64_t> allocationBytesPerBoundary =
      getL1AllocationSizeBytes(module, payloadBytesPerBoundary);
  if (failed(allocationBytesPerBoundary)) {
    module.emitOpError(
        "DFB reconfiguration scratch allocation is not representable");
    return failure();
  }
  std::optional<uint64_t> allocationBytes = llvm::checkedMulUnsigned(
      *stateBytes / payloadBytesPerBoundary, *allocationBytesPerBoundary);
  if (!allocationBytes) {
    module.emitOpError(
        "DFB reconfiguration scratch allocation is not representable");
    return failure();
  }
  return *allocationBytes;
}

FailureOr<uint64_t>
getSynchronizedDFBResetStateAllocationBytes(ModuleOp module) {
  FailureOr<uint64_t> stateBytes = getSynchronizedDFBResetStateBytes(module);
  if (failed(stateBytes)) {
    return failure();
  }
  FailureOr<uint64_t> allocationBytes =
      getL1AllocationSizeBytes(module, *stateBytes);
  if (failed(allocationBytes)) {
    module.emitOpError("DFB reset scratch allocation is not representable");
    return failure();
  }
  return *allocationBytes;
}

LogicalResult validateSynchronizedDFBResetTarget(ModuleOp module) {
  Operation *firstReset = nullptr;
  module.walk([&](Operation *operation) -> WalkResult {
    if (!isa<ResetDFBsOp, ResetAllDFBsOp>(operation)) {
      return WalkResult::advance();
    }
    firstReset = operation;
    return WalkResult::interrupt();
  });
  if (!firstReset) {
    return success();
  }

  std::string failureReason;
  FailureOr<std::optional<ttcore::Arch>> targetArch =
      resolveTargetArch(module, failureReason);
  if (failed(targetArch)) {
    module.emitOpError(failureReason);
    return failure();
  }
  if (!*targetArch) {
    firstReset->emitOpError(
        "requires a resolved target architecture; synchronized DFB reset is "
        "supported only for Blackhole");
    return failure();
  }
  if (**targetArch == ttcore::Arch::Blackhole) {
    return success();
  }
  firstReset->emitOpError()
      << "is supported only for Blackhole; selected target is "
      << ttcore::ArchAttr::get(module.getContext(), **targetArch);
  return failure();
}

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

FailureOr<uint64_t> getDFBL1AllocationSizeBytes(ModuleOp module,
                                                CircularBufferType type,
                                                std::string &failureReason) {
  FailureOr<uint64_t> payloadBytes =
      getDFBAllocationSizeBytes(type, failureReason);
  if (failed(payloadBytes)) {
    return failure();
  }
  FailureOr<uint64_t> allocationBytes =
      getL1AllocationSizeBytes(module, *payloadBytes);
  if (failed(allocationBytes)) {
    failureReason = "DFB L1 allocation size is not representable";
    return failure();
  }
  return *allocationBytes;
}

FailureOr<DFBStorageLayout>
mergeDFBStorageLayout(const DFBStorageLayout &layout, uint64_t memberBytes,
                      uint64_t memberPageSize, std::string &failureReason) {
  if (memberPageSize == 0) {
    failureReason = "DFB storage page size must be positive";
    return failure();
  }
  uint64_t commonDivisor = std::gcd(layout.alignmentBytes, memberPageSize);
  std::optional<uint64_t> mergedAlignment = llvm::checkedMulUnsigned(
      layout.alignmentBytes / commonDivisor, memberPageSize);
  if (!mergedAlignment) {
    failureReason = "DFB storage alignment is not representable";
    return failure();
  }
  uint64_t requiredCapacity = std::max(layout.capacityBytes, memberBytes);
  std::optional<uint64_t> roundedNumerator =
      llvm::checkedAddUnsigned(requiredCapacity, *mergedAlignment - 1);
  if (!roundedNumerator) {
    failureReason = "DFB storage allocation size is not representable";
    return failure();
  }
  uint64_t pageCount = *roundedNumerator / *mergedAlignment;
  std::optional<uint64_t> mergedCapacity =
      llvm::checkedMulUnsigned(pageCount, *mergedAlignment);
  if (!mergedCapacity) {
    failureReason = "DFB storage allocation size is not representable";
    return failure();
  }
  return DFBStorageLayout{*mergedCapacity, *mergedAlignment};
}

LogicalResult DFBStorageFootprint::add(int64_t storageIndex,
                                       CircularBufferType type,
                                       std::string &failureReason) {
  FailureOr<uint64_t> allocationBytes =
      getDFBAllocationSizeBytes(type, failureReason);
  FailureOr<uint64_t> pageSize = getDFBPageSizeBytes(type);
  if (failed(allocationBytes) || failed(pageSize)) {
    if (failureReason.empty()) {
      failureReason = "DFB page size is not representable";
    }
    return failure();
  }
  DFBStorageLayout &layout = layoutByIndex[storageIndex];
  FailureOr<DFBStorageLayout> mergedLayout =
      mergeDFBStorageLayout(layout, *allocationBytes, *pageSize, failureReason);
  if (failed(mergedLayout)) {
    return failure();
  }
  layout = *mergedLayout;
  return success();
}

FailureOr<uint64_t>
DFBStorageFootprint::getL1AllocationBytes(ModuleOp module) const {
  uint64_t totalBytes = 0;
  for (int64_t storageIndex : getSortedStorageIndices()) {
    FailureOr<uint64_t> allocationBytes =
        getL1AllocationSizeBytes(module, getBytes(storageIndex));
    std::optional<uint64_t> updatedTotal =
        succeeded(allocationBytes)
            ? llvm::checkedAddUnsigned(totalBytes, *allocationBytes)
            : std::nullopt;
    if (!updatedTotal) {
      return failure();
    }
    totalBytes = *updatedTotal;
  }
  return totalBytes;
}

uint64_t DFBStorageFootprint::getBytes(int64_t storageIndex) const {
  auto indexIt = layoutByIndex.find(storageIndex);
  assert(indexIt != layoutByIndex.end() &&
         "storage index must be present in the footprint");
  return indexIt->second.capacityBytes;
}

llvm::SmallVector<int64_t>
DFBStorageFootprint::getSortedStorageIndices() const {
  llvm::SmallVector<int64_t> storageIndices;
  storageIndices.reserve(layoutByIndex.size());
  for (int64_t storageIndex : llvm::make_first_range(layoutByIndex)) {
    storageIndices.push_back(storageIndex);
  }
  llvm::sort(storageIndices);
  return storageIndices;
}

FailureOr<bool> DFBAllocationFootprint::add(ModuleOp module,
                                            int64_t physicalIndex,
                                            CircularBufferType type,
                                            std::string &failureReason) {
  FailureOr<uint64_t> allocationBytes =
      getDFBL1AllocationSizeBytes(module, type, failureReason);
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

llvm::SmallVector<int64_t>
DFBAllocationFootprint::getSortedPhysicalIndices() const {
  llvm::SmallVector<int64_t> physicalIndices;
  physicalIndices.reserve(maxBytesByIndex.size());
  for (int64_t physicalIndex : llvm::make_first_range(maxBytesByIndex)) {
    physicalIndices.push_back(physicalIndex);
  }
  llvm::sort(physicalIndices);
  return physicalIndices;
}

FailureOr<uint64_t> FinalizedDFBStorageFootprint::getPeakL1AllocationBytes(
    ModuleOp module, std::optional<LaunchNodeCoord> *peakNode) const {
  if (!usesPerNodeAccounting) {
    return globalFootprint.getL1AllocationBytes(module);
  }
  uint64_t peakBytes = 0;
  for (auto indexedFootprint : llvm::enumerate(footprintsByNode)) {
    FailureOr<uint64_t> nodeBytes =
        indexedFootprint.value().getL1AllocationBytes(module);
    if (failed(nodeBytes)) {
      return failure();
    }
    if (indexedFootprint.index() == 0 || *nodeBytes > peakBytes) {
      peakBytes = *nodeBytes;
      if (peakNode) {
        *peakNode = launchNodes[indexedFootprint.index()];
      }
    }
  }
  return peakBytes;
}

static FailureOr<LaunchNodeDomain>
parseDFBNodeDomain(ModuleOp module, ArrayAttr nodeEntries,
                   size_t allocationIndex, llvm::StringRef fieldName) {
  LaunchNodeDomain domain;
  for (auto indexedNode : llvm::enumerate(nodeEntries)) {
    auto coordinates = dyn_cast<ArrayAttr>(indexedNode.value());
    if (!coordinates || coordinates.size() != 2 ||
        !isa<IntegerAttr>(coordinates[0]) ||
        !isa<IntegerAttr>(coordinates[1])) {
      module.emitOpError() << kDFBAllocationsAttrName << " entry "
                           << allocationIndex << " " << fieldName << " entry "
                           << indexedNode.index()
                           << " must contain two integer coordinates";
      return failure();
    }
    int64_t coreX = cast<IntegerAttr>(coordinates[0]).getInt();
    int64_t coreY = cast<IntegerAttr>(coordinates[1]).getInt();
    if (coreX < 0 || coreY < 0) {
      module.emitOpError() << kDFBAllocationsAttrName << " entry "
                           << allocationIndex << " " << fieldName << " entry "
                           << indexedNode.index()
                           << " requires nonnegative coordinates";
      return failure();
    }
    domain.nodes.insert({coreX, coreY});
  }
  return domain;
}

FailureOr<FinalizedDFBStorageFootprint>
getFinalizedDFBStorageFootprint(ModuleOp module) {
  FinalizedDFBStorageFootprint result;
  DenseMap<int64_t, LaunchNodeDomain> domainByPhysicalIndex;
  auto allocations = module->getAttrOfType<ArrayAttr>(kDFBAllocationsAttrName);
  if (allocations) {
    for (auto indexedEntry : llvm::enumerate(allocations)) {
      auto entry = dyn_cast<DictionaryAttr>(indexedEntry.value());
      if (!entry) {
        module.emitOpError() << kDFBAllocationsAttrName << " entry "
                             << indexedEntry.index() << " must be a dictionary";
        return failure();
      }
      IntegerAttr physicalIndexAttr = entry.getAs<IntegerAttr>("dfb_index");
      if (!physicalIndexAttr || physicalIndexAttr.getInt() < 0) {
        module.emitOpError()
            << kDFBAllocationsAttrName << " entry " << indexedEntry.index()
            << " requires a nonnegative dfb_index";
        return failure();
      }
      int64_t physicalIndex = physicalIndexAttr.getInt();
      Attribute storageIndexValue = entry.get("storage_index");
      IntegerAttr storageIndexAttr =
          dyn_cast_or_null<IntegerAttr>(storageIndexValue);
      if (storageIndexValue && !storageIndexAttr) {
        module.emitOpError()
            << kDFBAllocationsAttrName << " entry " << indexedEntry.index()
            << " requires an integer storage_index";
        return failure();
      }
      int64_t storageIndex =
          storageIndexAttr ? storageIndexAttr.getInt() : physicalIndex;
      if (storageIndex < 0) {
        module.emitOpError()
            << kDFBAllocationsAttrName << " entry " << indexedEntry.index()
            << " requires a nonnegative storage_index";
        return failure();
      }
      if (!result.storageIndexByPhysicalIndex
               .try_emplace(physicalIndex, storageIndex)
               .second) {
        module.emitOpError()
            << kDFBAllocationsAttrName << " contains duplicate dfb_index "
            << physicalIndex;
        return failure();
      }
      LaunchNodeDomain allocationDomain = LaunchNodeDomain::unknown();
      if (Attribute allocationNodesValue = entry.get("allocation_nodes")) {
        auto allocationNodes = dyn_cast<ArrayAttr>(allocationNodesValue);
        if (!allocationNodes) {
          module.emitOpError()
              << kDFBAllocationsAttrName << " entry " << indexedEntry.index()
              << " requires allocation_nodes to be an array";
          return failure();
        }
        FailureOr<LaunchNodeDomain> parsedDomain = parseDFBNodeDomain(
            module, allocationNodes, indexedEntry.index(), "allocation_nodes");
        if (failed(parsedDomain)) {
          return failure();
        }
        allocationDomain = std::move(*parsedDomain);
      }
      LaunchNodeDomain staticStorageDomain = allocationDomain;
      if (Attribute storageSegmentsValue = entry.get("storage_segments")) {
        auto storageSegments = dyn_cast<ArrayAttr>(storageSegmentsValue);
        if (!storageSegments) {
          module.emitOpError()
              << kDFBAllocationsAttrName << " entry " << indexedEntry.index()
              << " requires storage_segments to be an array";
          return failure();
        }
        staticStorageDomain = LaunchNodeDomain{};
        for (auto indexedSegment : llvm::enumerate(storageSegments)) {
          auto segment = dyn_cast<DictionaryAttr>(indexedSegment.value());
          if (!segment) {
            module.emitOpError()
                << kDFBAllocationsAttrName << " entry " << indexedEntry.index()
                << " storage_segments entry " << indexedSegment.index()
                << " must be a dictionary";
            return failure();
          }
          auto segmentNodes = segment.getAs<ArrayAttr>("nodes");
          if (!segmentNodes) {
            module.emitOpError()
                << kDFBAllocationsAttrName << " entry " << indexedEntry.index()
                << " storage_segments entry " << indexedSegment.index()
                << " requires a nodes array";
            return failure();
          }
          std::string segmentNodeField =
              (llvm::Twine("storage_segments entry ") +
               llvm::Twine(indexedSegment.index()) + " nodes")
                  .str();
          FailureOr<LaunchNodeDomain> segmentDomain = parseDFBNodeDomain(
              module, segmentNodes, indexedEntry.index(), segmentNodeField);
          if (failed(segmentDomain)) {
            return failure();
          }
          Attribute tensorBacking = segment.get("tensor_backing");
          if (tensorBacking && !isa<TensorBackingAttr>(tensorBacking)) {
            module.emitOpError()
                << kDFBAllocationsAttrName << " entry " << indexedEntry.index()
                << " storage_segments entry " << indexedSegment.index()
                << " has an invalid tensor_backing attribute";
            return failure();
          }
          if (!tensorBacking) {
            staticStorageDomain = staticStorageDomain.unionWith(*segmentDomain);
          }
        }
      }
      domainByPhysicalIndex.try_emplace(physicalIndex,
                                        std::move(staticStorageDomain));
    }
  }

  bool hasUnknownDomain = llvm::any_of(
      llvm::make_second_range(domainByPhysicalIndex),
      [](const LaunchNodeDomain &domain) { return !domain.known; });
  std::set<LaunchNodeCoord> launchNodes;
  if (hasUnknownDomain) {
    SmallVector<int64_t> launchGrid;
    if (auto dense =
            module->getAttrOfType<DenseI64ArrayAttr>(kLaunchGridAttrName)) {
      llvm::append_range(launchGrid, dense.asArrayRef());
    } else if (auto array =
                   module->getAttrOfType<ArrayAttr>(kLaunchGridAttrName)) {
      for (Attribute dimension : array) {
        auto integer = dyn_cast<IntegerAttr>(dimension);
        if (!integer) {
          launchGrid.clear();
          break;
        }
        launchGrid.push_back(integer.getInt());
      }
    }
    if (launchGrid.size() == 2 && launchGrid[0] > 0 && launchGrid[1] > 0) {
      launchNodes = getFullLaunchNodeDomain(launchGrid[0], launchGrid[1]).nodes;
    }
  } else {
    for (const LaunchNodeDomain &domain :
         llvm::make_second_range(domainByPhysicalIndex)) {
      launchNodes.insert(domain.nodes.begin(), domain.nodes.end());
    }
  }
  result.launchNodes.assign(launchNodes.begin(), launchNodes.end());
  result.usesPerNodeAccounting =
      static_cast<bool>(allocations) &&
      (!result.launchNodes.empty() ||
       llvm::all_of(
           llvm::make_second_range(domainByPhysicalIndex),
           [](const LaunchNodeDomain &domain) { return domain.known; }));
  result.footprintsByNode.resize(result.launchNodes.size());
  result.membersByNode.resize(result.launchNodes.size());

  auto recordMember = [](FinalizedDFBStorageFootprint::MembersByStorageIndex
                             &membersByStorageIndex,
                         int64_t storageIndex, int64_t physicalIndex) {
    SmallVector<int64_t> &members = membersByStorageIndex[storageIndex];
    if (!llvm::is_contained(members, physicalIndex)) {
      members.push_back(physicalIndex);
    }
  };

  WalkResult walkResult = module.walk([&](BindCBOp bindOp) -> WalkResult {
    if (bindOp.getTensorBackingAttr()) {
      return WalkResult::advance();
    }
    int64_t physicalIndex = bindOp.getCbIndex().getSExtValue();
    auto storageIndexIt =
        result.storageIndexByPhysicalIndex.find(physicalIndex);
    if (allocations &&
        storageIndexIt == result.storageIndexByPhysicalIndex.end()) {
      bindOp.emitOpError() << "physical DFB index " << physicalIndex
                           << " is missing from " << kDFBAllocationsAttrName;
      return WalkResult::interrupt();
    }
    int64_t storageIndex =
        storageIndexIt == result.storageIndexByPhysicalIndex.end()
            ? physicalIndex
            : storageIndexIt->second;
    auto dfbType = cast<CircularBufferType>(bindOp.getResult().getType());
    std::string failureReason;
    if (failed(
            result.globalFootprint.add(storageIndex, dfbType, failureReason))) {
      bindOp.emitOpError() << failureReason;
      return WalkResult::interrupt();
    }
    recordMember(result.globalMembers, storageIndex, physicalIndex);
    if (!result.usesPerNodeAccounting) {
      return WalkResult::advance();
    }
    const LaunchNodeDomain &domain = domainByPhysicalIndex.at(physicalIndex);
    for (auto indexedNode : llvm::enumerate(result.launchNodes)) {
      if (domain.known &&
          domain.nodes.find(indexedNode.value()) == domain.nodes.end()) {
        continue;
      }
      if (failed(result.footprintsByNode[indexedNode.index()].add(
              storageIndex, dfbType, failureReason))) {
        bindOp.emitOpError() << failureReason;
        return WalkResult::interrupt();
      }
      recordMember(result.membersByNode[indexedNode.index()], storageIndex,
                   physicalIndex);
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted()
             ? FailureOr<FinalizedDFBStorageFootprint>(failure())
             : FailureOr<FinalizedDFBStorageFootprint>(std::move(result));
}

FailureOr<DFBAllocationFootprint>
getLogicalDFBAllocationFootprint(ModuleOp module,
                                 const DFBLogicalIdentityAnalysis &identities) {
  DFBAllocationFootprint footprint;
  WalkResult walkResult = module.walk([&](BindCBOp bindOp) {
    if (bindOp.getTensorBackingAttr()) {
      return WalkResult::advance();
    }
    std::string failureReason;
    FailureOr<bool> increased = footprint.add(
        module, identities.getLogicalId(bindOp),
        cast<CircularBufferType>(bindOp.getResult().getType()), failureReason);
    return failed(increased) ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }
  return footprint;
}

FailureOr<uint64_t> getGlobalSemaphoreL1Bytes(ModuleOp module,
                                              int64_t semaphoreCount) {
  if (semaphoreCount < 0) {
    return failure();
  }
  FailureOr<uint64_t> semaphoreAllocationBytes =
      getL1AllocationSizeBytes(module, kGlobalSemaphorePayloadBytes);
  if (failed(semaphoreAllocationBytes)) {
    return failure();
  }
  std::optional<uint64_t> allocationBytes = llvm::checkedMulUnsigned(
      static_cast<uint64_t>(semaphoreCount), *semaphoreAllocationBytes);
  if (!allocationBytes) {
    return failure();
  }
  return *allocationBytes;
}

LogicalResult validateCombinedDFBResourceL1Bytes(
    ModuleOp module, uint64_t dfbBytes, uint64_t scratchBytes,
    int64_t globalSemaphoreCount, std::optional<uint64_t> overrideBytes) {
  FailureOr<uint64_t> scratchAllocationBytes =
      getL1AllocationSizeBytes(module, scratchBytes);
  FailureOr<uint64_t> globalSemaphoreBytes =
      getGlobalSemaphoreL1Bytes(module, globalSemaphoreCount);
  FailureOr<uint64_t> reconfigurationStateBytes =
      getDFBReconfigurationStateAllocationBytes(module);
  std::optional<uint64_t> requiredBytes =
      succeeded(scratchAllocationBytes)
          ? llvm::checkedAddUnsigned(dfbBytes, *scratchAllocationBytes)
          : std::nullopt;
  if (requiredBytes && succeeded(globalSemaphoreBytes)) {
    requiredBytes =
        llvm::checkedAddUnsigned(*requiredBytes, *globalSemaphoreBytes);
  } else if (failed(globalSemaphoreBytes)) {
    requiredBytes = std::nullopt;
  }
  if (requiredBytes && succeeded(reconfigurationStateBytes)) {
    requiredBytes =
        llvm::checkedAddUnsigned(*requiredBytes, *reconfigurationStateBytes);
  } else if (failed(reconfigurationStateBytes)) {
    requiredBytes = std::nullopt;
  }
  if (!requiredBytes) {
    module.emitOpError("combined L1 allocation size is not representable");
    return failure();
  }
  uint64_t budgetBytes = getUsableDFBL1Bytes(module, overrideBytes);
  if (*requiredBytes <= budgetBytes) {
    return success();
  }
  module.emitOpError() << "combined DFB and runtime resources require "
                       << *requiredBytes << " L1 bytes but the budget is "
                       << budgetBytes << " (DFB=" << dfbBytes
                       << ", scratch=" << *scratchAllocationBytes
                       << ", global semaphores=" << *globalSemaphoreBytes
                       << ", reconfiguration state="
                       << *reconfigurationStateBytes << ")";
  return failure();
}

uint64_t getUsableDFBL1Bytes(ModuleOp module,
                             std::optional<uint64_t> overrideBytes) {
  if (overrideBytes) {
    return *overrideBytes;
  }
  return tryBudgetFromModule(module).value_or(kFallbackUsableL1Bytes);
}

} // namespace mlir::tt::ttl
