// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

namespace ttk = mlir::tt::ttkernel;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELANALYZEDFBRESOURCES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kLaunchGridAttrName = "ttl.launch_grid";
constexpr llvm::StringLiteral kCoreCoordAttrName = "ttl.core_coord";
constexpr llvm::StringLiteral kLogicalConfigsAttrName =
    "ttl.logical_dfb_configs";
constexpr llvm::StringLiteral kLogicalIndexAttrName =
    "ttl.dfb_logical_index";
constexpr llvm::StringLiteral kPerCoreConfigsAttrName =
    "ttl.per_core_dfb_configs";
constexpr llvm::StringLiteral kEpochPhysicalConfigsAttrName =
    "ttl.dfb_epoch_physical_configs";
constexpr llvm::StringLiteral kKernelUnpackToDestFp32AttrName =
    "ttl.unpack_to_dest_fp32";
constexpr llvm::StringLiteral kResetEpochAttrName = "ttl.dfb_reset_epoch";
constexpr llvm::StringLiteral kResetPreservedIndicesAttrName =
    "ttl.dfb_reset_preserved_indices";
constexpr llvm::StringLiteral kResetCallee =
    "ttlang::reset_dataflow_buffers";
constexpr size_t kResetConfigWords = 11;

enum class AddressScope { Local, RemoteUniform, Legacy };

struct CoreCoord {
  int64_t x;
  int64_t y;
};

struct LogicalConfig {
  int64_t logicalIndex;
  int64_t physicalIndex;
  int64_t epoch;
  int64_t numPages;
  uint64_t pageBytes;
  Type elementType;
  bool unpackToDestFp32;
  bool compilerAllocated;
  int64_t blockCount;
  int64_t elemsPerBlock;
  AddressScope scope;
};

struct PhysicalInfo {
  int64_t initialEpoch = std::numeric_limits<int64_t>::max();
  int64_t initialLogicalIndex = std::numeric_limits<int64_t>::max();
  uint64_t initialPageBytes = 0;
  AddressScope scope = AddressScope::Local;
};

struct PhysicalConfig {
  int64_t physicalIndex;
  int64_t numPages;
  AddressScope scope;

  bool operator==(const PhysicalConfig &other) const {
    return physicalIndex == other.physicalIndex &&
           numPages == other.numPages && scope == other.scope;
  }
};

struct CoreGroup {
  SmallVector<PhysicalConfig> configs;
  SmallVector<CoreCoord> coords;
};

struct EpochPhysicalConfig {
  uint64_t bytes;
  uint64_t pageBytes;

  bool operator==(const EpochPhysicalConfig &other) const {
    return bytes == other.bytes && pageBytes == other.pageBytes;
  }

  bool operator!=(const EpochPhysicalConfig &other) const {
    return !(*this == other);
  }
};

struct EpochSlotUse {
  bool present = false;
  bool active = false;
  AddressScope scope = AddressScope::Local;
  uint64_t pageBytes = 0;
  SmallVector<uint64_t> bytesByCore;
};

struct PhysicalSlotUse {
  bool present = false;
  bool active = false;
  AddressScope scope = AddressScope::Local;
  uint64_t pageBytes = 0;
  SmallVector<uint64_t> bytesByCore;
};

static uint64_t roundUpTo(uint64_t value, uint64_t alignment) {
  assert(alignment > 0);
  return (value / alignment + (value % alignment != 0)) * alignment;
}

static AddressScope joinScope(AddressScope lhs, AddressScope rhs) {
  if (lhs == AddressScope::RemoteUniform ||
      rhs == AddressScope::RemoteUniform) {
    return AddressScope::RemoteUniform;
  }
  if (lhs == AddressScope::Legacy || rhs == AddressScope::Legacy) {
    return AddressScope::Legacy;
  }
  return AddressScope::Local;
}

static StringRef stringifyScope(AddressScope scope) {
  switch (scope) {
  case AddressScope::Local:
    return "local";
  case AddressScope::RemoteUniform:
    return "remote_uniform";
  case AddressScope::Legacy:
    return "legacy";
  }
  llvm_unreachable("unknown DFB address scope");
}

static FailureOr<AddressScope> parseScope(DictionaryAttr config) {
  auto attr = config.getAs<StringAttr>("address_scope");
  if (!attr) {
    return AddressScope::Legacy;
  }
  if (attr.getValue() == "local") {
    return AddressScope::Local;
  }
  if (attr.getValue() == "remote_uniform") {
    return AddressScope::RemoteUniform;
  }
  return failure();
}

static FailureOr<std::pair<int64_t, int64_t>> readGrid(ModuleOp module) {
  auto attr = module->getAttrOfType<ArrayAttr>(kLaunchGridAttrName);
  if (!attr || attr.size() != 2) {
    return failure();
  }
  auto x = dyn_cast<IntegerAttr>(attr[0]);
  auto y = dyn_cast<IntegerAttr>(attr[1]);
  if (!x || !y || x.getInt() <= 0 || y.getInt() <= 0) {
    return failure();
  }
  return std::pair<int64_t, int64_t>{x.getInt(), y.getInt()};
}

static FailureOr<uint64_t> getPageBytes(Type elementType) {
  if (auto tileType = dyn_cast<ttcore::TileType>(elementType)) {
    return tileType.getSizeBytes();
  }
  if (!elementType.isIntOrFloat()) {
    return failure();
  }
  return ttcore::TileType::get(elementType).getSizeBytes();
}

static FailureOr<int64_t> getInteger(DictionaryAttr config, StringRef name) {
  auto attr = config.getAs<IntegerAttr>(name);
  if (!attr) {
    return failure();
  }
  return attr.getInt();
}

static FailureOr<SmallVector<size_t>>
getCoveredCores(func::FuncOp func, int64_t gridX, int64_t gridY) {
  SmallVector<size_t> result;
  auto coords = func->getAttrOfType<ArrayAttr>(kCoreCoordAttrName);
  if (!coords) {
    result.reserve(static_cast<size_t>(gridX * gridY));
    for (int64_t y = 0; y < gridY; ++y) {
      for (int64_t x = 0; x < gridX; ++x) {
        result.push_back(static_cast<size_t>(y * gridX + x));
      }
    }
    return result;
  }
  if (coords.empty()) {
    return failure();
  }

  llvm::SmallDenseSet<size_t, 4> seen;
  for (Attribute coordAttr : coords) {
    auto coord = dyn_cast<ArrayAttr>(coordAttr);
    if (!coord || coord.size() != 2) {
      return failure();
    }
    auto x = dyn_cast<IntegerAttr>(coord[0]);
    auto y = dyn_cast<IntegerAttr>(coord[1]);
    if (!x || !y || x.getInt() < 0 || x.getInt() >= gridX ||
        y.getInt() < 0 || y.getInt() >= gridY) {
      return failure();
    }
    size_t index = static_cast<size_t>(y.getInt() * gridX + x.getInt());
    if (seen.insert(index).second) {
      result.push_back(index);
    }
  }
  return result;
}

static FailureOr<llvm::SmallDenseSet<int64_t, 8>>
getPreservedPhysicalIndices(ttk::OpaqueCallOp call) {
  llvm::SmallDenseSet<int64_t, 8> result;
  auto attr = call->getAttrOfType<ArrayAttr>(kResetPreservedIndicesAttrName);
  if (!attr) {
    return result;
  }
  for (Attribute value : attr) {
    auto index = dyn_cast<IntegerAttr>(value);
    if (!index || index.getInt() < 0 || !result.insert(index.getInt()).second) {
      return failure();
    }
  }
  return result;
}

static bool isMetadataOnlyResetUse(Operation *user) {
  auto call = dyn_cast<ttk::OpaqueCallOp>(user);
  return call && call.getCallee() == kResetCallee;
}

struct EpochRemapPlan {
  int64_t localSlotCount = 0;
  SmallVector<SmallVector<int64_t>> oldSlotByPhysical;
  std::map<int64_t, int64_t> pinnedPhysicalByOld;

  int64_t physicalIndex(int64_t epoch, int64_t oldPhysicalIndex) const {
    auto pinned = pinnedPhysicalByOld.find(oldPhysicalIndex);
    if (pinned != pinnedPhysicalByOld.end()) {
      return pinned->second;
    }
    if (oldPhysicalIndex >= localSlotCount || epoch < 0 ||
        epoch >= static_cast<int64_t>(oldSlotByPhysical.size())) {
      return oldPhysicalIndex;
    }
    const auto &assignment = oldSlotByPhysical[epoch];
    for (auto [physicalIndex, oldIndex] : llvm::enumerate(assignment)) {
      if (oldIndex == oldPhysicalIndex) {
        return static_cast<int64_t>(physicalIndex);
      }
    }
    return oldPhysicalIndex;
  }
};

static EpochRemapPlan buildEpochRemapPlan(
    const llvm::MapVector<int64_t, LogicalConfig> &logicalConfigs,
    ArrayRef<llvm::SmallDenseSet<int64_t, 8>> logicalsByCore,
    const llvm::SmallDenseSet<int64_t, 8> &pinnedPhysicalIndices,
    const std::map<int64_t, llvm::SmallDenseSet<int64_t, 8>>
        &pinnedLiveEpochs) {
  EpochRemapPlan plan;
  int64_t maxEpoch = -1;
  int64_t maxPhysicalIndex = -1;
  for (const auto &[logicalIndex, logical] : logicalConfigs) {
    (void)logicalIndex;
    maxEpoch = std::max(maxEpoch, logical.epoch);
    maxPhysicalIndex = std::max(maxPhysicalIndex, logical.physicalIndex);
    if (!pinnedPhysicalIndices.contains(logical.physicalIndex)) {
      plan.localSlotCount =
          std::max(plan.localSlotCount, logical.physicalIndex + 1);
    }
  }
  for (const auto &[physicalIndex, epochs] : pinnedLiveEpochs) {
    (void)physicalIndex;
    for (int64_t epoch : epochs) {
      maxEpoch = std::max(maxEpoch, epoch);
    }
  }
  if (maxEpoch <= 0 ||
      (plan.localSlotCount <= 1 && pinnedPhysicalIndices.empty())) {
    return plan;
  }
  for (int64_t pinned : pinnedPhysicalIndices) {
    if (pinned < plan.localSlotCount) {
      return EpochRemapPlan{};
    }
  }

  const int64_t physicalSlotCount = maxPhysicalIndex + 1;
  SmallVector<int64_t> sortedPinnedIndices(pinnedPhysicalIndices.begin(),
                                           pinnedPhysicalIndices.end());
  llvm::sort(sortedPinnedIndices);
  for (auto [newPhysicalIndex, oldPhysicalIndex] :
       llvm::enumerate(sortedPinnedIndices)) {
    plan.pinnedPhysicalByOld[oldPhysicalIndex] =
        static_cast<int64_t>(newPhysicalIndex);
  }

  const size_t coreCount = logicalsByCore.size();
  SmallVector<SmallVector<EpochSlotUse>> epochUses;
  epochUses.resize(static_cast<size_t>(maxEpoch + 1));
  for (auto &uses : epochUses) {
    uses.resize(static_cast<size_t>(plan.localSlotCount));
  }
  SmallVector<SmallVector<PhysicalSlotUse>> pinnedUsesByEpoch;
  pinnedUsesByEpoch.resize(static_cast<size_t>(maxEpoch + 1));
  for (auto &uses : pinnedUsesByEpoch) {
    uses.resize(static_cast<size_t>(physicalSlotCount));
  }

  auto recordUse = [&](auto &use, const LogicalConfig &logical) {
    if (!use.present) {
      use.present = true;
      use.pageBytes = logical.pageBytes;
      use.bytesByCore.assign(coreCount, 0);
    } else {
      assert(use.pageBytes == logical.pageBytes &&
             "one epoch slot must have one page size");
    }
    const uint64_t bytes =
        static_cast<uint64_t>(logical.numPages) * logical.pageBytes;
    if (logical.scope == AddressScope::Legacy) {
      use.scope = use.active ? joinScope(use.scope, logical.scope)
                             : logical.scope;
      use.active = true;
      for (uint64_t &coreBytes : use.bytesByCore) {
        coreBytes = std::max(coreBytes, bytes);
      }
      return;
    }
    bool logicalIsActive = false;
    for (size_t core = 0; core < coreCount; ++core) {
      if (logicalsByCore[core].contains(logical.logicalIndex)) {
        logicalIsActive = true;
        use.bytesByCore[core] = std::max(use.bytesByCore[core], bytes);
      }
    }
    if (logicalIsActive) {
      use.scope = use.active ? joinScope(use.scope, logical.scope)
                             : logical.scope;
      use.active = true;
    }
  };

  for (const auto &[logicalIndex, logical] : logicalConfigs) {
    (void)logicalIndex;
    if (pinnedPhysicalIndices.contains(logical.physicalIndex)) {
      auto liveEpochs = pinnedLiveEpochs.find(logical.physicalIndex);
      assert(liveEpochs != pinnedLiveEpochs.end() &&
             "preserved DFB must have a live epoch set");
      for (int64_t epoch : liveEpochs->second) {
        recordUse(pinnedUsesByEpoch[epoch][logical.physicalIndex], logical);
      }
      continue;
    }
    recordUse(epochUses[logical.epoch][logical.physicalIndex], logical);
  }

  plan.oldSlotByPhysical.resize(epochUses.size());
  for (size_t epoch = 0; epoch < epochUses.size(); ++epoch) {
    auto &assignment = plan.oldSlotByPhysical[epoch];
    const int64_t assignmentSize = pinnedPhysicalIndices.empty()
                                       ? plan.localSlotCount
                                       : physicalSlotCount;
    assignment.assign(static_cast<size_t>(assignmentSize), -1);
    for (int64_t oldIndex = 0; oldIndex < plan.localSlotCount; ++oldIndex) {
      if (epochUses[epoch][oldIndex].present) {
        assignment[oldIndex] = oldIndex;
      }
    }
  }

  auto evaluate = [&](const auto &assignments) {
    SmallVector<PhysicalSlotUse> physicalUses;
    physicalUses.resize(static_cast<size_t>(physicalSlotCount));
    auto mergeUse = [&](PhysicalSlotUse &physical,
                        const auto &logicalUse) {
      if (!logicalUse.present) {
        return;
      }
      if (!physical.present) {
        physical.present = true;
        physical.pageBytes = logicalUse.pageBytes;
        physical.bytesByCore.assign(coreCount, 0);
      }
      if (!logicalUse.active) {
        return;
      }
      physical.scope = physical.active
                           ? joinScope(physical.scope, logicalUse.scope)
                           : logicalUse.scope;
      physical.active = true;
      for (size_t core = 0; core < coreCount; ++core) {
        physical.bytesByCore[core] = std::max(
            physical.bytesByCore[core], logicalUse.bytesByCore[core]);
      }
    };
    for (size_t epoch = 0; epoch < assignments.size(); ++epoch) {
      for (size_t physical = 0; physical < assignments[epoch].size();
           ++physical) {
        int64_t oldIndex = assignments[epoch][physical];
        if (oldIndex >= 0) {
          mergeUse(physicalUses[physical], epochUses[epoch][oldIndex]);
        }
      }
      for (int64_t oldPinned : sortedPinnedIndices) {
        const int64_t physical = plan.pinnedPhysicalByOld.at(oldPinned);
        mergeUse(physicalUses[physical],
                 pinnedUsesByEpoch[epoch][oldPinned]);
      }
    }

    SmallVector<uint64_t> totals(coreCount, 0);
    for (const PhysicalSlotUse &physical : physicalUses) {
      if (!physical.active) {
        continue;
      }
      if (physical.scope == AddressScope::Local) {
        for (size_t core = 0; core < coreCount; ++core) {
          totals[core] +=
              roundUpTo(physical.bytesByCore[core], physical.pageBytes);
        }
        continue;
      }
      uint64_t uniformBytes = 0;
      for (uint64_t coreBytes : physical.bytesByCore) {
        uniformBytes = std::max(uniformBytes, coreBytes);
      }
      for (size_t core = 0; core < coreCount; ++core) {
        if (physical.bytesByCore[core] != 0) {
          totals[core] += roundUpTo(uniformBytes, physical.pageBytes);
        }
      }
    }
    llvm::sort(totals, std::greater<uint64_t>());
    return totals;
  };
  auto isBetter = [](ArrayRef<uint64_t> lhs, ArrayRef<uint64_t> rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                        rhs.end());
  };

  const auto identityAssignments = plan.oldSlotByPhysical;
  auto identityObjective = evaluate(identityAssignments);
  auto greedyAssignments = identityAssignments;
  const size_t firstGreedyEpoch = pinnedPhysicalIndices.empty() ? 1 : 0;
  for (size_t epoch = firstGreedyEpoch;
       epoch < greedyAssignments.size(); ++epoch) {
    greedyAssignments[epoch].assign(
        greedyAssignments[epoch].size(), -1);
  }
  SmallVector<size_t> epochOrder;
  for (size_t epoch = firstGreedyEpoch; epoch < epochUses.size(); ++epoch) {
    epochOrder.push_back(epoch);
  }
  llvm::sort(epochOrder, [&](size_t lhs, size_t rhs) {
    uint64_t lhsBytes = 0;
    uint64_t rhsBytes = 0;
    for (const EpochSlotUse &use : epochUses[lhs]) {
      for (uint64_t bytes : use.bytesByCore) {
        lhsBytes += bytes;
      }
    }
    for (const EpochSlotUse &use : epochUses[rhs]) {
      for (uint64_t bytes : use.bytesByCore) {
        rhsBytes += bytes;
      }
    }
    if (lhsBytes != rhsBytes) {
      return lhsBytes > rhsBytes;
    }
    return lhs < rhs;
  });
  for (size_t epoch : epochOrder) {
    SmallVector<int64_t> oldIndices;
    for (int64_t oldIndex = 0; oldIndex < plan.localSlotCount; ++oldIndex) {
      if (epochUses[epoch][oldIndex].present) {
        oldIndices.push_back(oldIndex);
      }
    }
    llvm::sort(oldIndices, [&](int64_t lhs, int64_t rhs) {
      uint64_t lhsBytes = 0;
      uint64_t rhsBytes = 0;
      for (uint64_t bytes : epochUses[epoch][lhs].bytesByCore) {
        lhsBytes = std::max(lhsBytes, bytes);
      }
      for (uint64_t bytes : epochUses[epoch][rhs].bytesByCore) {
        rhsBytes = std::max(rhsBytes, bytes);
      }
      if (lhsBytes != rhsBytes) {
        return lhsBytes > rhsBytes;
      }
      return lhs < rhs;
    });
    for (int64_t oldIndex : oldIndices) {
      int64_t bestPhysical = -1;
      SmallVector<uint64_t> bestObjective;
      for (int64_t physical = 0;
           physical < static_cast<int64_t>(greedyAssignments[epoch].size());
           ++physical) {
        if (greedyAssignments[epoch][physical] >= 0) {
          continue;
        }
        bool occupiedByPinned = false;
        for (int64_t oldPinned : sortedPinnedIndices) {
          auto liveEpochs = pinnedLiveEpochs.find(oldPinned);
          if (plan.pinnedPhysicalByOld.at(oldPinned) == physical &&
              liveEpochs != pinnedLiveEpochs.end() &&
              liveEpochs->second.contains(static_cast<int64_t>(epoch))) {
            occupiedByPinned = true;
            break;
          }
        }
        if (occupiedByPinned) {
          continue;
        }
        greedyAssignments[epoch][physical] = oldIndex;
        auto objective = evaluate(greedyAssignments);
        greedyAssignments[epoch][physical] = -1;
        if (bestPhysical < 0 || isBetter(objective, bestObjective)) {
          bestPhysical = physical;
          bestObjective = std::move(objective);
        }
      }
      assert(bestPhysical >= 0 && "epoch has more DFB slots than its arena");
      greedyAssignments[epoch][bestPhysical] = oldIndex;
    }
  }

  auto greedyObjective = evaluate(greedyAssignments);
  if (!pinnedPhysicalIndices.empty() ||
      isBetter(greedyObjective, identityObjective)) {
    plan.oldSlotByPhysical = std::move(greedyAssignments);
  }
  auto currentObjective = evaluate(plan.oldSlotByPhysical);
  while (true) {
    int64_t bestEpoch = -1;
    int64_t bestLhs = -1;
    int64_t bestRhs = -1;
    SmallVector<uint64_t> bestObjective = currentObjective;
    for (size_t epoch = 1; epoch < plan.oldSlotByPhysical.size(); ++epoch) {
      auto &assignment = plan.oldSlotByPhysical[epoch];
      for (int64_t lhs = 0; lhs < static_cast<int64_t>(assignment.size());
           ++lhs) {
        for (int64_t rhs = lhs + 1;
             rhs < static_cast<int64_t>(assignment.size()); ++rhs) {
          if (assignment[lhs] < 0 && assignment[rhs] < 0) {
            continue;
          }
          bool swapsIntoPinned = false;
          for (int64_t oldPinned : sortedPinnedIndices) {
            auto liveEpochs = pinnedLiveEpochs.find(oldPinned);
            if (liveEpochs == pinnedLiveEpochs.end() ||
                !liveEpochs->second.contains(static_cast<int64_t>(epoch))) {
              continue;
            }
            int64_t pinnedPhysical = plan.pinnedPhysicalByOld.at(oldPinned);
            swapsIntoPinned |= pinnedPhysical == lhs || pinnedPhysical == rhs;
          }
          if (swapsIntoPinned) {
            continue;
          }
          std::swap(assignment[lhs], assignment[rhs]);
          auto objective = evaluate(plan.oldSlotByPhysical);
          std::swap(assignment[lhs], assignment[rhs]);
          if (isBetter(objective, bestObjective)) {
            bestEpoch = static_cast<int64_t>(epoch);
            bestLhs = lhs;
            bestRhs = rhs;
            bestObjective = std::move(objective);
          }
        }
      }
    }
    if (bestEpoch < 0) {
      break;
    }
    std::swap(plan.oldSlotByPhysical[bestEpoch][bestLhs],
              plan.oldSlotByPhysical[bestEpoch][bestRhs]);
    currentObjective = std::move(bestObjective);
  }

  if (!pinnedPhysicalIndices.empty()) {
    llvm::SmallDenseSet<int64_t, 8> usedPhysicalIndices;
    for (const auto &[oldPhysical, newPhysical] :
         plan.pinnedPhysicalByOld) {
      (void)oldPhysical;
      usedPhysicalIndices.insert(newPhysical);
    }
    for (const auto &assignment : plan.oldSlotByPhysical) {
      for (auto [physicalIndex, oldIndex] : llvm::enumerate(assignment)) {
        if (oldIndex >= 0) {
          usedPhysicalIndices.insert(static_cast<int64_t>(physicalIndex));
        }
      }
    }
    SmallVector<int64_t> sortedUsedPhysicalIndices(
        usedPhysicalIndices.begin(), usedPhysicalIndices.end());
    llvm::sort(sortedUsedPhysicalIndices);
    std::map<int64_t, int64_t> compactPhysicalIndex;
    for (auto [newIndex, oldIndex] :
         llvm::enumerate(sortedUsedPhysicalIndices)) {
      compactPhysicalIndex[oldIndex] = static_cast<int64_t>(newIndex);
    }
    for (auto &[oldPhysical, newPhysical] : plan.pinnedPhysicalByOld) {
      (void)oldPhysical;
      newPhysical = compactPhysicalIndex.at(newPhysical);
    }
    for (auto &assignment : plan.oldSlotByPhysical) {
      SmallVector<int64_t> compactAssignment(sortedUsedPhysicalIndices.size(),
                                             -1);
      for (auto [physicalIndex, oldIndex] : llvm::enumerate(assignment)) {
        if (oldIndex >= 0) {
          compactAssignment[compactPhysicalIndex.at(
              static_cast<int64_t>(physicalIndex))] = oldIndex;
        }
      }
      assignment = std::move(compactAssignment);
    }
  }
  return plan;
}

struct TTKernelAnalyzeDFBResourcesPass
    : impl::TTKernelAnalyzeDFBResourcesBase<
          TTKernelAnalyzeDFBResourcesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto configsAttr =
        module->getAttrOfType<ArrayAttr>(kLogicalConfigsAttrName);
    if (!configsAttr) {
      return;
    }

    FailureOr<std::pair<int64_t, int64_t>> grid = readGrid(module);
    if (failed(grid)) {
      module.emitOpError() << "requires `" << kLaunchGridAttrName
                           << "` to be a length-2 array of positive extents";
      signalPassFailure();
      return;
    }
    auto [gridX, gridY] = *grid;

    llvm::MapVector<int64_t, LogicalConfig> logicalConfigs;
    for (Attribute attr : configsAttr) {
      auto config = dyn_cast<DictionaryAttr>(attr);
      if (!config) {
        module.emitOpError()
            << "`" << kLogicalConfigsAttrName
            << "` entries must be dictionaries";
        signalPassFailure();
        return;
      }
      FailureOr<int64_t> logicalIndex = getInteger(config, "logical_index");
      FailureOr<int64_t> physicalIndex = getInteger(config, "physical_index");
      FailureOr<int64_t> epoch = getInteger(config, "epoch");
      FailureOr<int64_t> numPages = getInteger(config, "num_pages");
      auto elementType = config.getAs<TypeAttr>("element_type");
      auto unpackToDestFp32 =
          config.getAs<BoolAttr>("unpack_to_dest_fp32");
      auto compilerAllocated = config.getAs<BoolAttr>("compiler_allocated");
      auto blockCount = config.getAs<IntegerAttr>("block_count");
      auto elemsPerBlock = config.getAs<IntegerAttr>("elems_per_block");
      FailureOr<AddressScope> scope = parseScope(config);
      if (failed(logicalIndex) || failed(physicalIndex) || failed(epoch) ||
          failed(numPages) || !elementType || !unpackToDestFp32 ||
          failed(scope) ||
          *logicalIndex < 0 || *physicalIndex < 0 || *epoch < 0 ||
          *numPages <= 0 || (blockCount && blockCount.getInt() <= 0) ||
          (elemsPerBlock && elemsPerBlock.getInt() <= 0)) {
        module.emitOpError() << "has malformed `" << kLogicalConfigsAttrName
                             << "` entry " << config;
        signalPassFailure();
        return;
      }
      FailureOr<uint64_t> pageBytes = getPageBytes(elementType.getValue());
      if (failed(pageBytes) || *pageBytes == 0 ||
          static_cast<uint64_t>(*numPages) >
              std::numeric_limits<uint64_t>::max() / *pageBytes) {
        module.emitOpError() << "cannot size logical DFB " << *logicalIndex;
        signalPassFailure();
        return;
      }
      if (logicalConfigs.count(*logicalIndex)) {
        module.emitOpError() << "has duplicate logical DFB index "
                             << *logicalIndex;
        signalPassFailure();
        return;
      }

      LogicalConfig logical{*logicalIndex,
                            *physicalIndex,
                            *epoch,
                            *numPages,
                            *pageBytes,
                            elementType.getValue(),
                            unpackToDestFp32.getValue(),
                            compilerAllocated && compilerAllocated.getValue(),
                            blockCount ? blockCount.getInt() : 1,
                            elemsPerBlock ? elemsPerBlock.getInt() : *numPages,
                            *scope};
      logicalConfigs.insert({logical.logicalIndex, logical});
    }

    const size_t coreCount = static_cast<size_t>(gridX * gridY);
    SmallVector<llvm::SmallDenseSet<int64_t, 8>> logicalsByCore(coreCount);
    bool walkFailed = false;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      llvm::SmallDenseSet<int64_t, 8> usedLogicals;
      WalkResult result = func.walk([&](ttk::GetCompileArgValOp argOp) {
        if (llvm::all_of(argOp.getResult().getUsers(),
                         isMetadataOnlyResetUse)) {
          return WalkResult::advance();
        }
        auto logicalIndex =
            argOp->getAttrOfType<IntegerAttr>(kLogicalIndexAttrName);
        if (!logicalIndex) {
          if (isa<ttk::CBType>(argOp.getResult().getType())) {
            argOp.emitOpError()
                << "is CB-typed but missing `" << kLogicalIndexAttrName << "`";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        }
        auto config = logicalConfigs.find(logicalIndex.getInt());
        if (config == logicalConfigs.end()) {
          argOp.emitOpError() << "references logical DFB "
                              << logicalIndex.getInt() << " absent from `"
                              << kLogicalConfigsAttrName << "`";
          return WalkResult::interrupt();
        }
        if (static_cast<int64_t>(argOp.getArgIndex()) !=
            config->second.physicalIndex) {
          argOp.emitOpError()
              << "reads physical DFB " << argOp.getArgIndex()
              << " but logical DFB " << logicalIndex.getInt() << " maps to "
              << config->second.physicalIndex;
          return WalkResult::interrupt();
        }
        usedLogicals.insert(logicalIndex.getInt());
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        walkFailed = true;
        break;
      }
      FailureOr<SmallVector<size_t>> covered =
          getCoveredCores(func, gridX, gridY);
      if (failed(covered)) {
        func.emitOpError() << "has malformed `" << kCoreCoordAttrName << "`";
        walkFailed = true;
        break;
      }
      for (size_t core : *covered) {
        for (int64_t logicalIndex : usedLogicals) {
          logicalsByCore[core].insert(logicalIndex);
        }
      }
    }
    if (walkFailed) {
      signalPassFailure();
      return;
    }

    SmallVector<ttk::OpaqueCallOp> resetCalls;
    llvm::SmallDenseSet<int64_t, 8> pinnedPhysicalIndices;
    std::map<int64_t, llvm::SmallDenseSet<int64_t, 8>> pinnedLiveEpochs;
    module.walk([&](ttk::OpaqueCallOp call) {
      if (call.getCallee() != kResetCallee) {
        return;
      }
      resetCalls.push_back(call);
      FailureOr<llvm::SmallDenseSet<int64_t, 8>> preserved =
          getPreservedPhysicalIndices(call);
      if (failed(preserved)) {
        call.emitOpError() << "has malformed `"
                           << kResetPreservedIndicesAttrName << "`";
        walkFailed = true;
        return;
      }
      auto epoch = call->getAttrOfType<IntegerAttr>(kResetEpochAttrName);
      if (!epoch || epoch.getInt() < 0) {
        call.emitOpError() << "is missing a valid `" << kResetEpochAttrName
                           << "`";
        walkFailed = true;
        return;
      }
      pinnedPhysicalIndices.insert(preserved->begin(), preserved->end());
      for (int64_t physicalIndex : *preserved) {
        pinnedLiveEpochs[physicalIndex].insert(epoch.getInt());
      }
    });
    if (walkFailed) {
      signalPassFailure();
      return;
    }

    std::map<int64_t, int64_t> pinnedLogicalByPhysical;
    for (int64_t physicalIndex : pinnedPhysicalIndices) {
      for (const auto &[logicalIndex, logical] : logicalConfigs) {
        if (logical.physicalIndex != physicalIndex) {
          continue;
        }
        if (!pinnedLogicalByPhysical
                 .try_emplace(physicalIndex, logicalIndex)
                 .second) {
          module.emitOpError()
              << "preserved physical DFB " << physicalIndex
              << " is shared by more than one logical DFB before epoch "
                 "packing";
          signalPassFailure();
          return;
        }
        pinnedLiveEpochs[physicalIndex].insert(logical.epoch);
      }
      if (pinnedLogicalByPhysical.find(physicalIndex) ==
          pinnedLogicalByPhysical.end()) {
        module.emitOpError() << "preserves unknown physical DFB "
                             << physicalIndex;
        signalPassFailure();
        return;
      }
    }

    EpochRemapPlan remapPlan;
    if (!resetCalls.empty() &&
        module->hasAttr(kEpochPhysicalConfigsAttrName)) {
      remapPlan = buildEpochRemapPlan(logicalConfigs, logicalsByCore,
                                      pinnedPhysicalIndices,
                                      pinnedLiveEpochs);
    }
    if (!remapPlan.oldSlotByPhysical.empty()) {
      for (ttk::OpaqueCallOp call : resetCalls) {
        auto epoch = call->getAttrOfType<IntegerAttr>(kResetEpochAttrName);
        ArrayAttr oldArgs = call.getTemplateArgsAttr();
        auto oldCount = oldArgs && !oldArgs.empty()
                            ? dyn_cast<IntegerAttr>(oldArgs[0])
                            : IntegerAttr();
        if (!epoch || !oldCount || oldCount.getInt() < 0 ||
            oldArgs.size() !=
                1 + static_cast<size_t>(oldCount.getInt()) *
                        kResetConfigWords) {
          call.emitOpError()
              << "has malformed reset metadata before DFB epoch packing";
          signalPassFailure();
          return;
        }
        OpBuilder builder(call);
        SmallVector<Attribute> remappedRecords;
        int64_t remappedCount = 0;
        for (int64_t record = 0; record < oldCount.getInt(); ++record) {
          size_t base = 1 + static_cast<size_t>(record) * kResetConfigWords;
          auto oldPhysical = dyn_cast<IntegerAttr>(oldArgs[base]);
          if (!oldPhysical) {
            call.emitOpError() << "has a non-integer DFB reset slot";
            signalPassFailure();
            return;
          }
          if (pinnedPhysicalIndices.contains(oldPhysical.getInt()) &&
              !pinnedLiveEpochs.at(oldPhysical.getInt())
                   .contains(epoch.getInt())) {
            continue;
          }
          remappedRecords.append(oldArgs.begin() + base,
                                 oldArgs.begin() + base + kResetConfigWords);
          remappedRecords[remappedRecords.size() - kResetConfigWords] =
              builder.getI64IntegerAttr(remapPlan.physicalIndex(
                  epoch.getInt(), oldPhysical.getInt()));
          ++remappedCount;
        }
        SmallVector<Attribute> remappedArgs{
            builder.getI64IntegerAttr(remappedCount)};
        remappedArgs.append(remappedRecords);
        call.setTemplateArgsAttr(builder.getArrayAttr(remappedArgs));

        FailureOr<llvm::SmallDenseSet<int64_t, 8>> preserved =
            getPreservedPhysicalIndices(call);
        assert(succeeded(preserved) && "preserved indices were validated");
        SmallVector<Attribute> remappedPreserved;
        for (int64_t physicalIndex : *preserved) {
          remappedPreserved.push_back(builder.getI64IntegerAttr(
              remapPlan.physicalIndex(epoch.getInt(), physicalIndex)));
        }
        call->setAttr(kResetPreservedIndicesAttrName,
                      builder.getArrayAttr(remappedPreserved));
      }

      for (auto &[logicalIndex, logical] : logicalConfigs) {
        (void)logicalIndex;
        logical.physicalIndex =
            remapPlan.physicalIndex(logical.epoch, logical.physicalIndex);
      }
      module.walk([&](ttk::GetCompileArgValOp argOp) {
        auto logicalIndex =
            argOp->getAttrOfType<IntegerAttr>(kLogicalIndexAttrName);
        if (!logicalIndex) {
          return;
        }
        auto config = logicalConfigs.find(logicalIndex.getInt());
        if (config != logicalConfigs.end()) {
          argOp.setArgIndex(
              static_cast<uint32_t>(config->second.physicalIndex));
        }
      });
      for (func::FuncOp func : module.getOps<func::FuncOp>()) {
        auto thread = func->getAttrOfType<ttk::ThreadTypeAttr>(
            ttk::ThreadTypeAttr::name);
        if (!thread || thread.getValue() != ttk::ThreadType::Compute) {
          continue;
        }
        auto originalUnpackIndices =
            func->getAttrOfType<DenseI32ArrayAttr>(
                kKernelUnpackToDestFp32AttrName);
        llvm::SmallDenseSet<int64_t, 8> originalUnpackSet;
        if (originalUnpackIndices) {
          for (int32_t index : originalUnpackIndices.asArrayRef()) {
            originalUnpackSet.insert(index);
          }
        }
        llvm::SmallDenseSet<int32_t, 8> unpackIndices;
        func.walk([&](ttk::GetCompileArgValOp argOp) {
          auto logicalIndex =
              argOp->getAttrOfType<IntegerAttr>(kLogicalIndexAttrName);
          if (!logicalIndex ||
              !originalUnpackSet.contains(logicalIndex.getInt()) ||
              llvm::all_of(argOp.getResult().getUsers(),
                           isMetadataOnlyResetUse)) {
            return;
          }
          auto config = logicalConfigs.find(logicalIndex.getInt());
          if (config != logicalConfigs.end()) {
            unpackIndices.insert(
                static_cast<int32_t>(config->second.physicalIndex));
          }
        });
        if (unpackIndices.empty()) {
          func->removeAttr(kKernelUnpackToDestFp32AttrName);
          continue;
        }
        SmallVector<int32_t> sortedUnpackIndices(unpackIndices.begin(),
                                                 unpackIndices.end());
        llvm::sort(sortedUnpackIndices);
        func->setAttr(kKernelUnpackToDestFp32AttrName,
                      DenseI32ArrayAttr::get(module.getContext(),
                                             sortedUnpackIndices));
      }

      OpBuilder builder(module.getContext());
      SmallVector<Attribute> remappedConfigs;
      remappedConfigs.reserve(configsAttr.size());
      for (Attribute attr : configsAttr) {
        auto config = cast<DictionaryAttr>(attr);
        int64_t logicalIndex =
            config.getAs<IntegerAttr>("logical_index").getInt();
        NamedAttrList fields(config.getValue());
        fields.set("physical_index", builder.getI64IntegerAttr(
                                         logicalConfigs[logicalIndex]
                                             .physicalIndex));
        remappedConfigs.push_back(fields.getDictionary(module.getContext()));
      }
      configsAttr = builder.getArrayAttr(remappedConfigs);
      module->setAttr(kLogicalConfigsAttrName, configsAttr);

      SmallVector<Attribute> indexMapEntries;
      std::map<int64_t, const LogicalConfig *> compilerByPhysical;
      for (const auto &[logicalIndex, logical] : logicalConfigs) {
        if (!logical.compilerAllocated) {
          if (logical.physicalIndex != logicalIndex) {
            indexMapEntries.push_back(DictionaryAttr::get(
                module.getContext(),
                {builder.getNamedAttr(
                     "old_index", builder.getI32IntegerAttr(logicalIndex)),
                 builder.getNamedAttr(
                     "new_index",
                     builder.getI32IntegerAttr(logical.physicalIndex))}));
          }
          continue;
        }
        auto [it, inserted] =
            compilerByPhysical.try_emplace(logical.physicalIndex, &logical);
        if (!inserted) {
          const uint64_t candidateBytes =
              static_cast<uint64_t>(logical.numPages) * logical.pageBytes;
          const uint64_t currentBytes =
              static_cast<uint64_t>(it->second->numPages) *
              it->second->pageBytes;
          if (candidateBytes > currentBytes ||
              (candidateBytes == currentBytes &&
               logical.logicalIndex < it->second->logicalIndex)) {
            it->second = &logical;
          }
        }
      }
      if (indexMapEntries.empty()) {
        module->removeAttr(kDFBIndexMapAttrName);
      } else {
        module->setAttr(kDFBIndexMapAttrName,
                        builder.getArrayAttr(indexMapEntries));
      }

      SmallVector<Attribute> compilerEntries;
      for (const auto &[physicalIndex, logical] : compilerByPhysical) {
        compilerEntries.push_back(DictionaryAttr::get(
            module.getContext(),
            {builder.getNamedAttr(
                 "dfb_index", builder.getI32IntegerAttr(physicalIndex)),
             builder.getNamedAttr(
                 "num_tiles",
                 builder.getI32IntegerAttr(logical->elemsPerBlock)),
             builder.getNamedAttr("element_type",
                                  TypeAttr::get(logical->elementType)),
             builder.getNamedAttr(
                 "block_count",
                 builder.getI32IntegerAttr(logical->blockCount))}));
      }
      if (compilerEntries.empty()) {
        module->removeAttr(kCompilerAllocatedDFBsAttrName);
      } else {
        module->setAttr(kCompilerAllocatedDFBsAttrName,
                        builder.getArrayAttr(compilerEntries));
      }

      llvm::SmallDenseSet<int64_t, 8> remappedPinnedPhysicalIndices;
      std::map<int64_t, llvm::SmallDenseSet<int64_t, 8>>
          remappedPinnedLiveEpochs;
      std::map<int64_t, int64_t> remappedPinnedLogicalByPhysical;
      for (int64_t oldPhysicalIndex : pinnedPhysicalIndices) {
        int64_t newPhysicalIndex =
            remapPlan.physicalIndex(0, oldPhysicalIndex);
        remappedPinnedPhysicalIndices.insert(newPhysicalIndex);
        remappedPinnedLiveEpochs[newPhysicalIndex] =
            pinnedLiveEpochs.at(oldPhysicalIndex);
        remappedPinnedLogicalByPhysical[newPhysicalIndex] =
            pinnedLogicalByPhysical.at(oldPhysicalIndex);
      }
      pinnedPhysicalIndices = std::move(remappedPinnedPhysicalIndices);
      pinnedLiveEpochs = std::move(remappedPinnedLiveEpochs);
      pinnedLogicalByPhysical =
          std::move(remappedPinnedLogicalByPhysical);
    }

    llvm::MapVector<int64_t, PhysicalInfo> physicalInfos;
    for (const auto &[logicalIndex, logical] : logicalConfigs) {
      (void)logicalIndex;
      PhysicalInfo &physical = physicalInfos[logical.physicalIndex];
      bool logicalIsActive = logical.scope == AddressScope::Legacy;
      for (const auto &coreLogicals : logicalsByCore) {
        logicalIsActive |= coreLogicals.contains(logical.logicalIndex);
      }
      if (logicalIsActive) {
        physical.scope = joinScope(physical.scope, logical.scope);
      }
      if (logical.epoch == physical.initialEpoch &&
          logical.pageBytes != physical.initialPageBytes) {
        module.emitOpError()
            << "physical DFB " << logical.physicalIndex
            << " has incompatible page sizes in initial epoch";
        signalPassFailure();
        return;
      }
      if (std::tie(logical.epoch, logical.logicalIndex) <
          std::tie(physical.initialEpoch, physical.initialLogicalIndex)) {
        physical.initialEpoch = logical.epoch;
        physical.initialLogicalIndex = logical.logicalIndex;
        physical.initialPageBytes = logical.pageBytes;
      }
    }

    if (!remapPlan.oldSlotByPhysical.empty() &&
        module->hasAttr(kEpochPhysicalConfigsAttrName)) {
      OpBuilder physicalBuilder(module.getContext());
      int64_t physicalCount = 0;
      for (const auto &[physicalIndex, physical] : physicalInfos) {
        (void)physical;
        physicalCount = std::max(physicalCount, physicalIndex + 1);
      }
      SmallVector<const LogicalConfig *> initialConfigs(physicalCount,
                                                        nullptr);
      SmallVector<uint64_t> maxBytes(physicalCount, 0);
      llvm::SmallDenseSet<int64_t, 8> activeLogicals;
      for (const auto &coreLogicals : logicalsByCore) {
        activeLogicals.insert(coreLogicals.begin(), coreLogicals.end());
      }
      for (const auto &[logicalIndex, logical] : logicalConfigs) {
        const size_t physical = static_cast<size_t>(logical.physicalIndex);
        const LogicalConfig *initial = initialConfigs[physical];
        if (!initial ||
            std::tie(logical.epoch, logical.logicalIndex) <
                std::tie(initial->epoch, initial->logicalIndex)) {
          initialConfigs[physical] = &logical;
        }
        if (logical.scope == AddressScope::Legacy ||
            activeLogicals.contains(logicalIndex)) {
          maxBytes[physical] = std::max(
              maxBytes[physical],
              static_cast<uint64_t>(logical.numPages) * logical.pageBytes);
        }
      }
      SmallVector<Attribute> physicalConfigs;
      physicalConfigs.reserve(static_cast<size_t>(physicalCount));
      for (int64_t physical = 0; physical < physicalCount; ++physical) {
        const LogicalConfig *initial = initialConfigs[physical];
        auto tileType =
            initial ? dyn_cast<ttcore::TileType>(initial->elementType)
                    : ttcore::TileType();
        if (!initial || !tileType) {
          module.emitOpError()
              << "epoch DFB packing produced a sparse or non-tile slot "
              << physical;
          signalPassFailure();
          return;
        }
        const uint64_t totalSize = roundUpTo(
            std::max(maxBytes[physical], initial->pageBytes),
            initial->pageBytes);
        physicalConfigs.push_back(DictionaryAttr::get(
            module.getContext(),
            {physicalBuilder.getNamedAttr(
                 "dfb_index", physicalBuilder.getI32IntegerAttr(physical)),
             physicalBuilder.getNamedAttr(
                 "element_type", TypeAttr::get(initial->elementType)),
             physicalBuilder.getNamedAttr(
                 "tile_height",
                 physicalBuilder.getI32IntegerAttr(tileType.getHeight())),
             physicalBuilder.getNamedAttr(
                 "tile_width",
                 physicalBuilder.getI32IntegerAttr(tileType.getWidth())),
             physicalBuilder.getNamedAttr(
                 "total_size",
                 physicalBuilder.getI64IntegerAttr(
                     static_cast<int64_t>(totalSize)))}));
      }
      module->setAttr(kEpochPhysicalConfigsAttrName,
                      physicalBuilder.getArrayAttr(physicalConfigs));
    }

    SmallVector<std::map<int64_t, uint64_t>> bytesByCore(coreCount);
    for (size_t core = 0; core < coreCount; ++core) {
      for (int64_t logicalIndex : logicalsByCore[core]) {
        const LogicalConfig &logical = logicalConfigs[logicalIndex];
        uint64_t bytes = static_cast<uint64_t>(logical.numPages) *
                         logical.pageBytes;
        uint64_t &current = bytesByCore[core][logical.physicalIndex];
        current = std::max(current, bytes);
      }
    }

    for (const auto &[logicalIndex, logical] : logicalConfigs) {
      (void)logicalIndex;
      if (logical.scope != AddressScope::Legacy) {
        continue;
      }
      uint64_t bytes = static_cast<uint64_t>(logical.numPages) *
                       logical.pageBytes;
      for (auto &core : bytesByCore) {
        uint64_t &current = core[logical.physicalIndex];
        current = std::max(current, bytes);
      }
    }

    SmallVector<SmallVector<PhysicalConfig>> configsByCore(coreCount);
    for (size_t core = 0; core < coreCount; ++core) {
      for (auto [physicalIndex, bytes] : bytesByCore[core]) {
        const PhysicalInfo &physical = physicalInfos[physicalIndex];
        uint64_t pages =
            (bytes + physical.initialPageBytes - 1) /
            physical.initialPageBytes;
        if (pages > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          module.emitOpError() << "physical DFB " << physicalIndex
                               << " capacity exceeds i32 page count";
          signalPassFailure();
          return;
        }
        configsByCore[core].push_back(PhysicalConfig{
            physicalIndex, static_cast<int64_t>(pages), physical.scope});
      }
    }

    llvm::MapVector<int64_t, LogicalConfig> pinnedConfigs;
    for (const auto &[physicalIndex, logicalIndex] :
         pinnedLogicalByPhysical) {
      auto pinned = logicalConfigs.find(logicalIndex);
      assert(pinned != logicalConfigs.end() &&
             "preserved logical DFB was validated before packing");
      assert(pinned->second.physicalIndex == physicalIndex &&
             "preserved DFB must retain one physical slot");
      pinnedConfigs.insert({physicalIndex, pinned->second});
    }

    auto getEpochConfigs =
        [&](size_t core, int64_t epoch)
        -> FailureOr<std::map<int64_t, EpochPhysicalConfig>> {
      std::map<int64_t, EpochPhysicalConfig> result;
      auto addConfig = [&](const LogicalConfig &logical) -> LogicalResult {
        if (logical.scope != AddressScope::Legacy &&
            !logicalsByCore[core].contains(logical.logicalIndex)) {
          return success();
        }
        uint64_t bytes = static_cast<uint64_t>(logical.numPages) *
                         logical.pageBytes;
        auto [it, inserted] = result.try_emplace(
            logical.physicalIndex,
            EpochPhysicalConfig{bytes, logical.pageBytes});
        if (!inserted) {
          if (it->second.pageBytes != logical.pageBytes) {
            module.emitOpError()
                << "physical DFB " << logical.physicalIndex << " in epoch "
                << epoch << " has incompatible page sizes";
            return failure();
          }
          it->second.bytes = std::max(it->second.bytes, bytes);
        }
        return success();
      };
      for (const auto &[logicalIndex, logical] : logicalConfigs) {
        (void)logicalIndex;
        if (logical.epoch != epoch) {
          continue;
        }
        if (failed(addConfig(logical))) {
          return failure();
        }
      }
      for (const auto &[physicalIndex, logical] : pinnedConfigs) {
        auto liveEpochs = pinnedLiveEpochs.find(physicalIndex);
        assert(liveEpochs != pinnedLiveEpochs.end() &&
               "preserved DFB must have a live epoch set");
        if (logical.epoch != epoch && liveEpochs->second.contains(epoch) &&
            failed(addConfig(logical))) {
          return failure();
        }
      }
      return result;
    };

    for (ttk::OpaqueCallOp call : resetCalls) {
      auto epochAttr = call->getAttrOfType<IntegerAttr>(kResetEpochAttrName);
      if (!epochAttr || epochAttr.getInt() < 0) {
        call.emitOpError() << "is missing a valid `" << kResetEpochAttrName
                           << "`";
        signalPassFailure();
        return;
      }
      func::FuncOp func = call->getParentOfType<func::FuncOp>();
      FailureOr<SmallVector<size_t>> covered =
          getCoveredCores(func, gridX, gridY);
      if (failed(covered)) {
        func.emitOpError() << "has malformed `" << kCoreCoordAttrName << "`";
        signalPassFailure();
        return;
      }
      FailureOr<llvm::SmallDenseSet<int64_t, 8>> preserved =
          getPreservedPhysicalIndices(call);
      assert(succeeded(preserved) && "preserved indices were validated above");

      std::optional<std::map<int64_t, EpochPhysicalConfig>> desired;
      for (size_t core : *covered) {
        FailureOr<std::map<int64_t, EpochPhysicalConfig>> coreDesired =
            getEpochConfigs(core, epochAttr.getInt());
        if (failed(coreDesired)) {
          signalPassFailure();
          return;
        }
        for (const auto &[physicalIndex, epochConfig] : *coreDesired) {
          const int64_t targetPhysicalIndex = physicalIndex;
          auto allocation = llvm::find_if(
              configsByCore[core], [&](const PhysicalConfig &config) {
                return config.physicalIndex == targetPhysicalIndex;
              });
          if (allocation == configsByCore[core].end()) {
            call.emitOpError() << "retains physical DFB " << physicalIndex
                               << " without backing on core " << core;
            signalPassFailure();
            return;
          }
          const PhysicalInfo &physical = physicalInfos[physicalIndex];
          uint64_t capacity =
              static_cast<uint64_t>(allocation->numPages) *
              physical.initialPageBytes;
          if (epochConfig.bytes > capacity) {
            call.emitOpError()
                << "configures physical DFB " << physicalIndex << " for "
                << epochConfig.bytes << " bytes with only " << capacity
                << " bytes allocated on core " << core;
            signalPassFailure();
            return;
          }
        }
        for (int64_t physicalIndex : *preserved) {
          coreDesired->erase(physicalIndex);
        }
        if (desired && *desired != *coreDesired) {
          call.emitOpError()
              << "covers cores with different reset configurations; enable "
                 "core specialization for this kernel";
          signalPassFailure();
          return;
        }
        desired = std::move(*coreDesired);
      }
      assert(desired && "reset kernel must cover at least one core");

      ArrayAttr oldArgs = call.getTemplateArgsAttr();
      auto oldCount = oldArgs && !oldArgs.empty()
                          ? dyn_cast<IntegerAttr>(oldArgs[0])
                          : IntegerAttr();
      if (!oldCount || oldCount.getInt() < 0 ||
          oldArgs.size() !=
              1 + static_cast<size_t>(oldCount.getInt()) *
                      kResetConfigWords) {
        call.emitOpError() << "has malformed reset configuration arguments";
        signalPassFailure();
        return;
      }

      OpBuilder builder(call);
      SmallVector<Attribute> newArgs{
          builder.getI64IntegerAttr(static_cast<int64_t>(desired->size()))};
      llvm::SmallDenseSet<int64_t, 8> retained;
      for (int64_t record = 0; record < oldCount.getInt(); ++record) {
        size_t base = 1 + static_cast<size_t>(record) * kResetConfigWords;
        auto physicalIndex = dyn_cast<IntegerAttr>(oldArgs[base]);
        auto pageBytes = dyn_cast<IntegerAttr>(oldArgs[base + 3]);
        if (!physicalIndex || !pageBytes || pageBytes.getInt() <= 0) {
          call.emitOpError() << "has a malformed reset configuration record";
          signalPassFailure();
          return;
        }
        auto config = desired->find(physicalIndex.getInt());
        if (config == desired->end()) {
          continue;
        }
        if (config->second.pageBytes !=
                static_cast<uint64_t>(pageBytes.getInt()) ||
            config->second.bytes % config->second.pageBytes != 0 ||
            config->second.bytes >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          call.emitOpError()
              << "cannot specialize physical DFB " << physicalIndex.getInt()
              << " reset geometry";
          signalPassFailure();
          return;
        }
        retained.insert(physicalIndex.getInt());
        newArgs.push_back(oldArgs[base]);
        newArgs.push_back(builder.getI64IntegerAttr(
            static_cast<int64_t>(config->second.bytes)));
        newArgs.push_back(builder.getI64IntegerAttr(
            static_cast<int64_t>(config->second.bytes /
                                 config->second.pageBytes)));
        newArgs.append(oldArgs.begin() + base + 3,
                       oldArgs.begin() + base + kResetConfigWords);
      }
      if (retained.size() != desired->size()) {
        call.emitOpError()
            << "reset table does not contain every active physical DFB";
        signalPassFailure();
        return;
      }
      call.setTemplateArgsAttr(builder.getArrayAttr(newArgs));
      call->removeAttr(kResetEpochAttrName);
      call->removeAttr(kResetPreservedIndicesAttrName);
    }

    SmallVector<CoreGroup> groups;
    for (int64_t y = 0; y < gridY; ++y) {
      for (int64_t x = 0; x < gridX; ++x) {
        size_t core = static_cast<size_t>(y * gridX + x);
        auto group = llvm::find_if(groups, [&](const CoreGroup &candidate) {
          return candidate.configs == configsByCore[core];
        });
        if (group == groups.end()) {
          groups.push_back(CoreGroup{configsByCore[core], {}});
          group = std::prev(groups.end());
        }
        group->coords.push_back(CoreCoord{x, y});
      }
    }

    OpBuilder builder(module.getContext());
    SmallVector<Attribute> groupAttrs;
    groupAttrs.reserve(groups.size());
    for (const CoreGroup &group : groups) {
      SmallVector<Attribute> configAttrs;
      configAttrs.reserve(group.configs.size());
      for (const PhysicalConfig &config : group.configs) {
        configAttrs.push_back(DictionaryAttr::get(
            module.getContext(),
            {builder.getNamedAttr("dfb_index", builder.getI32IntegerAttr(
                                                   config.physicalIndex)),
             builder.getNamedAttr("num_pages",
                                  builder.getI32IntegerAttr(config.numPages)),
             builder.getNamedAttr("address_scope", builder.getStringAttr(
                                                       stringifyScope(
                                                           config.scope)))}));
      }
      SmallVector<Attribute> coordAttrs;
      coordAttrs.reserve(group.coords.size());
      for (CoreCoord coord : group.coords) {
        coordAttrs.push_back(builder.getI64ArrayAttr({coord.x, coord.y}));
      }
      groupAttrs.push_back(DictionaryAttr::get(
          module.getContext(),
          {builder.getNamedAttr("core_coords", builder.getArrayAttr(coordAttrs)),
           builder.getNamedAttr("configs", builder.getArrayAttr(configAttrs))}));
    }
    module->setAttr(kPerCoreConfigsAttrName, builder.getArrayAttr(groupAttrs));
  }
};

} // namespace

} // namespace mlir::tt::ttl
