// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBStateDiscard.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl {

namespace {

/// Physical DFB masks hold one bit per index.
constexpr int64_t kMaxMaskedDFBIndices = 64;

uint64_t getDFBIndexBit(int64_t physicalIndex) {
  assert(physicalIndex >= 0 && physicalIndex < kMaxMaskedDFBIndices &&
         "physical DFB index must fit the mask");
  return uint64_t{1} << static_cast<unsigned>(physicalIndex);
}

} // namespace

FailureOr<int32_t> getValidatedDFBIndex(Value dfb, Operation *op) {
  std::optional<int64_t> dfbIndex = getCBIndex(dfb);
  if (!dfbIndex) {
    return op->emitError("cannot resolve finalized DFB index");
  }
  int32_t targetMaxDFBIndices = getTargetMaxDFBIndices(op);
  if (*dfbIndex < 0 || *dfbIndex >= targetMaxDFBIndices) {
    return op->emitError("finalized DFB index ")
           << *dfbIndex << " is outside [0, " << targetMaxDFBIndices - 1
           << "] for " << getTargetDFBIndexCapacityDescription(op);
  }
  return static_cast<int32_t>(*dfbIndex);
}

FailureOr<uint64_t> getAllocatedDFBMask(ModuleOp module) {
  uint64_t allocatedMask = 0;
  WalkResult result = module.walk([&](BindCBOp bind) -> WalkResult {
    std::optional<int64_t> dfbIndex = getCBIndex(bind.getResult());
    if (!dfbIndex) {
      bind.emitOpError("requires a finalized DFB index");
      return WalkResult::interrupt();
    }
    int32_t targetMaxDFBIndices = getTargetMaxDFBIndices(bind);
    if (*dfbIndex < 0 || *dfbIndex >= targetMaxDFBIndices) {
      bind.emitOpError("finalized DFB index ")
          << *dfbIndex << " is outside [0, " << targetMaxDFBIndices - 1
          << "] for " << getTargetDFBIndexCapacityDescription(bind);
      return WalkResult::interrupt();
    }
    allocatedMask |= getDFBIndexBit(*dfbIndex);
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  return allocatedMask;
}

FailureOr<uint64_t> getSynchronizedResetDFBMask(Operation *reset,
                                                uint64_t allocatedMask) {
  if (isa<ResetAllDFBsOp>(reset)) {
    return allocatedMask;
  }
  auto selectedReset = dyn_cast<ResetDFBsOp>(reset);
  if (!selectedReset) {
    return reset->emitError("is not a synchronized DFB reset");
  }
  uint64_t resetMask = 0;
  for (Value dfb : selectedReset.getDfbs()) {
    FailureOr<int32_t> dfbIndex = getValidatedDFBIndex(dfb, reset);
    if (failed(dfbIndex)) {
      return failure();
    }
    resetMask |= getDFBIndexBit(*dfbIndex);
  }
  return resetMask;
}

FailureOr<DFBReconfigurationInstalls>
DFBReconfigurationInstalls::build(ModuleOp module) {
  DFBReconfigurationInstalls installs;
  auto plan =
      module->getAttrOfType<DictionaryAttr>(kDFBReconfigurationPlanAttrName);
  if (!plan) {
    return installs;
  }
  auto malformed = [&](llvm::StringRef requirement) {
    module.emitOpError() << kDFBReconfigurationPlanAttrName << " "
                         << requirement;
    return failure();
  };
  auto dfbEntries = plan.getAs<ArrayAttr>("dfbs");
  if (!dfbEntries) {
    return malformed("requires a dfbs array");
  }
  for (Attribute entryAttr : dfbEntries) {
    auto entry = dyn_cast<DictionaryAttr>(entryAttr);
    auto physicalIndex =
        entry ? entry.getAs<IntegerAttr>("dfb_index") : IntegerAttr();
    auto configurations =
        entry ? entry.getAs<ArrayAttr>("configurations") : ArrayAttr();
    if (!physicalIndex || physicalIndex.getInt() < 0 ||
        physicalIndex.getInt() >= kMaxMaskedDFBIndices || !configurations) {
      return malformed("dfbs entries require a dfb_index in [0, 63] and a "
                       "configurations array");
    }
    for (Attribute configurationAttr : configurations) {
      auto configuration = dyn_cast<DictionaryAttr>(configurationAttr);
      if (!configuration) {
        return malformed("configurations must be dictionaries");
      }
      // The initial configuration has no entry boundary; the runtime installs
      // it at launch.
      auto entryOrdinal =
          configuration.getAs<IntegerAttr>("entry_reconfiguration");
      if (!entryOrdinal) {
        continue;
      }
      Install install{static_cast<int32_t>(physicalIndex.getInt()),
                      std::nullopt};
      // Without storage segments the runtime installs the configuration on
      // every node, as it does for an empty segment list.
      auto segments = configuration.getAs<ArrayAttr>("storage_segments");
      if (segments && !segments.empty()) {
        std::set<LaunchNodeCoord> nodes;
        for (Attribute segmentAttr : segments) {
          auto segment = dyn_cast<DictionaryAttr>(segmentAttr);
          auto nodeArray =
              segment ? segment.getAs<ArrayAttr>("nodes") : ArrayAttr();
          if (!nodeArray) {
            return malformed("storage segments require a nodes array");
          }
          for (Attribute nodeAttr : nodeArray) {
            auto coordinates = dyn_cast<ArrayAttr>(nodeAttr);
            auto x = coordinates && coordinates.size() == 2
                         ? dyn_cast<IntegerAttr>(coordinates[0])
                         : IntegerAttr();
            auto y = coordinates && coordinates.size() == 2
                         ? dyn_cast<IntegerAttr>(coordinates[1])
                         : IntegerAttr();
            if (!x || !y) {
              return malformed("storage segment nodes must be [x, y] pairs");
            }
            nodes.insert(LaunchNodeCoord{x.getInt(), y.getInt()});
          }
        }
        install.nodes = std::move(nodes);
      }
      installs.installsByOrdinal[entryOrdinal.getInt()].push_back(
          std::move(install));
    }
  }
  return installs;
}

uint64_t
DFBReconfigurationInstalls::getInstalledDFBMask(int64_t ordinal,
                                                LaunchNodeCoord node) const {
  auto installsIt = installsByOrdinal.find(ordinal);
  if (installsIt == installsByOrdinal.end()) {
    return 0;
  }
  uint64_t installedMask = 0;
  for (const Install &install : installsIt->second) {
    if (!install.nodes || install.nodes->count(node) != 0) {
      installedMask |= getDFBIndexBit(install.physicalIndex);
    }
  }
  return installedMask;
}

uint64_t
DFBReconfigurationInstalls::getInstalledDFBMask(int64_t ordinal) const {
  auto installsIt = installsByOrdinal.find(ordinal);
  if (installsIt == installsByOrdinal.end()) {
    return 0;
  }
  uint64_t installedMask = 0;
  for (const Install &install : installsIt->second) {
    installedMask |= getDFBIndexBit(install.physicalIndex);
  }
  return installedMask;
}

} // namespace mlir::tt::ttl
