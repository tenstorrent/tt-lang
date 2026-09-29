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
    return op->emitOpError("cannot resolve finalized DFB index");
  }
  int32_t targetMaxDFBIndices = getTargetMaxDFBIndices(op);
  if (*dfbIndex < 0 || *dfbIndex >= targetMaxDFBIndices) {
    return op->emitOpError("finalized DFB index ")
           << *dfbIndex << " is outside [0, " << targetMaxDFBIndices - 1
           << "] for " << getTargetDFBIndexCapacityDescription(op);
  }
  return static_cast<int32_t>(*dfbIndex);
}

FailureOr<uint64_t> getAllocatedDFBMask(ModuleOp module) {
  uint64_t allocatedMask = 0;
  WalkResult result = module.walk([&](BindCBOp bind) -> WalkResult {
    FailureOr<int32_t> dfbIndex = getValidatedDFBIndex(bind.getResult(), bind);
    if (failed(dfbIndex)) {
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
  assert((isa<ResetDFBsOp, ResetAllDFBsOp>(reset)) &&
         "expected a synchronized DFB reset");
  auto selectedReset = dyn_cast<ResetDFBsOp>(reset);
  if (!selectedReset) {
    return allocatedMask;
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

DFBReconfigurationInstalls
DFBReconfigurationInstalls::build(ModuleOp module,
                                  const LaunchNodeDomain &launchDomain) {
  DFBReconfigurationInstalls installs;
  auto plan =
      module->getAttrOfType<DictionaryAttr>(kDFBReconfigurationPlanAttrName);
  if (!plan) {
    return installs;
  }
  for (Attribute entryAttr : plan.getAs<ArrayAttr>("dfbs")) {
    auto entry = cast<DictionaryAttr>(entryAttr);
    auto physicalIndex =
        static_cast<int32_t>(entry.getAs<IntegerAttr>("dfb_index").getInt());
    for (Attribute configurationAttr :
         entry.getAs<ArrayAttr>("configurations")) {
      auto configuration = cast<DictionaryAttr>(configurationAttr);
      // The initial configuration has no entry boundary; the runtime installs
      // it at launch.
      auto entryOrdinal =
          configuration.getAs<IntegerAttr>("entry_reconfiguration");
      if (!entryOrdinal) {
        continue;
      }
      Install install{physicalIndex, launchDomain};
      // Without storage segments the runtime installs the configuration on
      // every node, as it does for an empty segment list.
      auto segments = configuration.getAs<ArrayAttr>("storage_segments");
      if (segments && !segments.empty()) {
        install.nodes = LaunchNodeDomain{};
        for (Attribute segment : segments) {
          for (Attribute node :
               cast<DictionaryAttr>(segment).getAs<ArrayAttr>("nodes")) {
            auto coordinates = cast<ArrayAttr>(node);
            install.nodes.nodes.insert(
                {cast<IntegerAttr>(coordinates[0]).getInt(),
                 cast<IntegerAttr>(coordinates[1]).getInt()});
          }
        }
      }
      installs.installsByOrdinal[entryOrdinal.getInt()].push_back(
          std::move(install));
    }
  }
  return installs;
}

uint64_t DFBReconfigurationInstalls::getInstalledDFBMask(
    int64_t ordinal, std::optional<LaunchNodeCoord> node) const {
  uint64_t installedMask = 0;
  auto installsIt = installsByOrdinal.find(ordinal);
  if (installsIt == installsByOrdinal.end()) {
    return 0;
  }
  for (const Install &install : installsIt->second) {
    if (!node || knownLaunchNodeDomainContains(install.nodes, *node)) {
      installedMask |= getDFBIndexBit(install.physicalIndex);
    }
  }
  return installedMask;
}

} // namespace mlir::tt::ttl
