// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBPhysicalAllocationPlan.h"

#include "DFBAllocationDebugReport.h"
#include "DFBAllocationLimits.h"
#include "DFBAnalysisFailure.h"
#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/InterferenceGraphColoring.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

StringRef getDFBConflictReasonName(DFBConflictReason reason) {
  switch (reason) {
  case DFBConflictReason::DescriptorMismatch:
    return "descriptor-mismatch";
  case DFBConflictReason::StorageMismatch:
    return "storage-mismatch";
  case DFBConflictReason::UnknownLaunchNodeDomain:
    return "unknown-launch-node-domain";
  case DFBConflictReason::AccessCompletionNotProven:
    return "access-completion-not-proven";
  case DFBConflictReason::TransactionMismatch:
    return "transaction-mismatch";
  case DFBConflictReason::PointerOwnerMismatch:
    return "pointer-owner-mismatch";
  case DFBConflictReason::ConcurrentLifetime:
    return "concurrent-lifetime";
  case DFBConflictReason::ResetDomainWrite:
    return "reset-domain-write";
  case DFBConflictReason::StaticConfigurationMismatch:
    return "static-configuration-mismatch";
  }
  llvm_unreachable("unknown DFB conflict reason");
}

StringRef getDFBAllocationGroupAssumptionReasonName(
    DFBAllocationGroupAssumptionReason reason) {
  switch (reason) {
  case DFBAllocationGroupAssumptionReason::UnknownLaunchNodeDomain:
    return "unknown-launch-node-domain";
  case DFBAllocationGroupAssumptionReason::AccessCompletionNotProven:
    return "access-completion-not-proven";
  case DFBAllocationGroupAssumptionReason::PointerOwnerMismatch:
    return "pointer-owner-mismatch";
  case DFBAllocationGroupAssumptionReason::ConcurrentLifetime:
    return "concurrent-lifetime";
  case DFBAllocationGroupAssumptionReason::UnprovenCursorOrder:
    return "unproven-cursor-order";
  case DFBAllocationGroupAssumptionReason::EpochReset:
    return "epoch-reset";
  }
  llvm_unreachable("unknown DFB allocation-group assumption reason");
}

namespace {

// Preserves failed-proof evidence before using the caller-selected fallback.
static Operation *getLifetimeEvidence(const DFBPerNodeLifetime *lifetime,
                                      Operation *fallbackEvidence) {
  if (lifetime && lifetime->completionProof.evidence) {
    return lifetime->completionProof.evidence;
  }
  return fallbackEvidence;
}

static Operation *getLifetimeEvidence(const DFBPerNodeLifetime *lifetime,
                                      const DFBLogicalLifecycle &logicalDFB) {
  return getLifetimeEvidence(lifetime, logicalDFB.declarations.front());
}

static bool cursorRunsCanRepeat(ArrayRef<DFBTransactionRun> cursorRuns,
                                std::uint64_t physicalTileCount) {
  FailureOr<std::uint64_t> terminalOffset =
      advanceDFBTransactionCursor(cursorRuns, physicalTileCount);
  if (failed(terminalOffset)) {
    return false;
  }
  if (cursorRuns.empty() || *terminalOffset == 0) {
    return true;
  }

  std::uint64_t totalMovement = 0;
  for (const DFBTransactionRun &run : cursorRuns) {
    std::optional<std::uint64_t> runMovement = llvm::checkedMulUnsigned(
        run.executionCount, static_cast<std::uint64_t>(run.tilesPerExecution));
    if (!runMovement) {
      return false;
    }
    std::optional<std::uint64_t> updatedTotal =
        llvm::checkedAddUnsigned(totalMovement, *runMovement);
    if (!updatedTotal) {
      return false;
    }
    totalMovement = *updatedTotal;
  }

  // Every reachable start offset is a multiple of this value. Requiring each
  // movement and its prefix to share that alignment prevents boundary crossing
  // when the complete cursor sequence is repeated.
  std::uint64_t repeatAlignment = std::gcd(totalMovement, physicalTileCount);
  std::uint64_t prefixMovement = 0;
  for (const DFBTransactionRun &run : cursorRuns) {
    std::uint64_t tilesPerExecution = run.tilesPerExecution;
    if (repeatAlignment % tilesPerExecution != 0 ||
        prefixMovement % tilesPerExecution != 0) {
      return false;
    }
    prefixMovement += run.executionCount * tilesPerExecution;
  }
  return true;
}

static bool haveCompatibleCursorRuns(const DFBPerNodeLifetime &before,
                                     const DFBPerNodeLifetime &after,
                                     std::uint64_t physicalTileCount) {
  return before.terminalWriteCursorRuns == after.writeCursorRuns &&
         before.terminalReadCursorRuns == after.readCursorRuns &&
         cursorRunsCanRepeat(before.terminalWriteCursorRuns,
                             physicalTileCount) &&
         cursorRunsCanRepeat(before.terminalReadCursorRuns, physicalTileCount);
}

struct AllocationGroupNodeEpoch {
  unsigned logicalIndex = 0;
  unsigned epochIndex = 0;
  const DFBPerNodeLifetime *lifetime = nullptr;
  const DFBLifecycleEpoch *epoch = nullptr;
  bool possibleDomain = false;

  const DFBLifecycleCompletionProof &getCompletionProof() const {
    return epoch ? epoch->completionProof : lifetime->completionProof;
  }

  ArrayRef<DFBTransactionRun> getWriteCursorRuns() const {
    if (epoch) {
      return epoch->writeCursorRuns;
    }
    return lifetime->writeCursorRuns;
  }

  ArrayRef<DFBTransactionRun> getReadCursorRuns() const {
    if (epoch) {
      return epoch->readCursorRuns;
    }
    return lifetime->readCursorRuns;
  }

  std::optional<DFBPointerOwner> getWritePointerOwner() const {
    return epoch ? epoch->writePointerOwner : lifetime->writePointerOwner;
  }

  std::optional<DFBPointerOwner> getReadPointerOwner() const {
    return epoch ? epoch->readPointerOwner : lifetime->readPointerOwner;
  }

  std::optional<DFBPointerOwner> getTerminalWritePointerOwner() const {
    return epoch ? epoch->terminalWritePointerOwner
                 : lifetime->terminalWritePointerOwner;
  }

  std::optional<DFBPointerOwner> getTerminalReadPointerOwner() const {
    return epoch ? epoch->terminalReadPointerOwner
                 : lifetime->terminalReadPointerOwner;
  }

  bool hasCanonicalTerminalState() const {
    return epoch ? epoch->terminalStateCanonical
                 : lifetime->terminalStateCanonical;
  }

  bool isInspectionOnly() const {
    return epoch ? epoch->inspectionOnly : lifetime->inspectionOnly;
  }
};

static void appendAllocationGroupNodeEpochs(
    SmallVectorImpl<AllocationGroupNodeEpoch> &epochs, unsigned logicalIndex,
    const DFBPerNodeLifetime &lifetime, bool possibleDomain) {
  if (lifetime.epochs.empty()) {
    epochs.push_back({logicalIndex, 0, &lifetime, nullptr, possibleDomain});
    return;
  }
  for (auto [epochIndex, epoch] : llvm::enumerate(lifetime.epochs)) {
    epochs.push_back({logicalIndex, static_cast<unsigned>(epochIndex),
                      &lifetime, &epoch, possibleDomain});
  }
}

static bool isAllocationGroupEpochOrderedBefore(
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    const AllocationGroupNodeEpoch &before,
    const AllocationGroupNodeEpoch &after, LaunchNodeCoord node) {
  if (before.possibleDomain || after.possibleDomain) {
    return before.possibleDomain && after.possibleDomain &&
           liveness.isConditionallyEpochOrderedBefore(
               before.logicalIndex, before.epochIndex, after.logicalIndex,
               after.epochIndex, node);
  }
  return liveness.isEpochOrderedBefore(before.logicalIndex, before.epochIndex,
                                       after.logicalIndex, after.epochIndex,
                                       node);
}

static Operation *
getAllocationGroupEpochEvidence(const AllocationGroupNodeEpoch &epoch,
                                const DFBLogicalLifecycle &logicalDFB) {
  if (!epoch.epoch) {
    Operation *fallbackEvidence = epoch.lifetime->entryEvidence
                                      ? epoch.lifetime->entryEvidence
                                      : logicalDFB.declarations.front();
    return getLifetimeEvidence(epoch.lifetime, fallbackEvidence);
  }
  if (epoch.getCompletionProof().evidence) {
    return epoch.getCompletionProof().evidence;
  }
  assert(!epoch.epoch->accessOccurrenceIndices.empty() &&
         "active lifecycle epoch must contain an access");
  return logicalDFB.accesses[epoch.epoch->accessOccurrenceIndices.front()]
      .operation;
}

static void appendActiveConfigurationEpochs(
    const DFBLifecycleEpoch &lifecycleEpoch,
    SmallVectorImpl<std::optional<int64_t>> &configurationEpochs) {
  ArrayRef<std::optional<int64_t>> activeEpochs =
      lifecycleEpoch.activeConfigurationEpochs;
  if (activeEpochs.empty()) {
    activeEpochs = ArrayRef(lifecycleEpoch.entryReconfigurationOrdinal);
  }
  for (std::optional<int64_t> activeEpoch : activeEpochs) {
    if (!llvm::is_contained(configurationEpochs, activeEpoch)) {
      configurationEpochs.push_back(activeEpoch);
    }
  }
}

static SmallVector<std::optional<int64_t>>
getActiveConfigurationEpochs(const DFBPerNodeLifetime &lifetime) {
  SmallVector<std::optional<int64_t>> configurationEpochs;
  for (const DFBLifecycleEpoch &lifecycleEpoch : lifetime.epochs) {
    appendActiveConfigurationEpochs(lifecycleEpoch, configurationEpochs);
  }
  return configurationEpochs;
}

static SmallVector<std::optional<int64_t>>
getActiveConfigurationEpochs(const DFBLogicalLifecycle &logicalDFB) {
  SmallVector<std::optional<int64_t>> epochs;
  for (const DFBPerNodeLifetime &lifetime : logicalDFB.nodeLifetimes) {
    for (const DFBLifecycleEpoch &epoch : lifetime.epochs) {
      appendActiveConfigurationEpochs(epoch, epochs);
    }
  }
  for (const DFBPerNodeLifetime &lifetime : logicalDFB.possibleNodeLifetimes) {
    for (const DFBLifecycleEpoch &epoch : lifetime.epochs) {
      appendActiveConfigurationEpochs(epoch, epochs);
    }
  }
  return epochs;
}

static bool haveDisjointConfigurationEpochs(const DFBLogicalLifecycle &lhs,
                                            const DFBLogicalLifecycle &rhs) {
  SmallVector<std::optional<int64_t>> lhsEpochs =
      getActiveConfigurationEpochs(lhs);
  SmallVector<std::optional<int64_t>> rhsEpochs =
      getActiveConfigurationEpochs(rhs);
  return !lhsEpochs.empty() && !rhsEpochs.empty() &&
         llvm::none_of(lhsEpochs, [&](std::optional<int64_t> lhsEpoch) {
           return llvm::is_contained(rhsEpochs, lhsEpoch);
         });
}

static bool haveDisjointConfigurationEpochs(const DFBPerNodeLifetime &lhs,
                                            const DFBPerNodeLifetime &rhs) {
  SmallVector<std::optional<int64_t>> lhsEpochs =
      getActiveConfigurationEpochs(lhs);
  SmallVector<std::optional<int64_t>> rhsEpochs =
      getActiveConfigurationEpochs(rhs);
  return !lhsEpochs.empty() && !rhsEpochs.empty() &&
         llvm::none_of(lhsEpochs, [&](std::optional<int64_t> lhsEpoch) {
           return llvm::is_contained(rhsEpochs, lhsEpoch);
         });
}

static bool haveIdenticalPageFormat(Type lhs, Type rhs) {
  return cast<CircularBufferType>(lhs).getElementType() ==
         cast<CircularBufferType>(rhs).getElementType();
}

static bool canUseCapacityEnvelope(const DFBLogicalLifecycle &lhs,
                                   const DFBLogicalLifecycle &rhs) {
  return lhs.allocationGroup && lhs.allocationGroup == rhs.allocationGroup &&
         haveIdenticalPageFormat(lhs.type, rhs.type) &&
         !lhs.hasOpaqueExternalAccess && !rhs.hasOpaqueExternalAccess &&
         !lhs.tensorBacking && !rhs.tensorBacking;
}

static bool
canReconfigureDescriptorAcrossEpochs(const DFBLogicalLifecycle &lhs,
                                     const DFBLogicalLifecycle &rhs) {
  return haveIdenticalPageFormat(lhs.type, rhs.type) &&
         !lhs.hasOpaqueExternalAccess && !rhs.hasOpaqueExternalAccess &&
         haveDisjointConfigurationEpochs(lhs, rhs);
}

static bool
requiresReconfigurationStorage(const DFBLogicalLifecycle &logicalDFB) {
  // The runtime gives changed descriptors hidden tensor backing, which cannot
  // also provide static storage for a distinct physical descriptor.
  auto lifetimeRequiresStorage = [](const DFBPerNodeLifetime &lifetime) {
    return llvm::any_of(lifetime.epochs, [](const DFBLifecycleEpoch &epoch) {
      return epoch.entryReconfigurationOrdinal.has_value();
    });
  };
  return llvm::any_of(logicalDFB.nodeLifetimes, lifetimeRequiresStorage) ||
         llvm::any_of(logicalDFB.possibleNodeLifetimes,
                      lifetimeRequiresStorage);
}

} // namespace

struct DFBPairConflictRequirements {
  bool requireExactDescriptor = true;
  bool requireMatchingElementType = true;
  bool requireMatchingTransactions = true;
  bool requireMatchingPointerOwners = true;
  bool useAllocationGroupEpochs = false;
  bool allowCapacityEnvelope = false;
  bool requireStaticStorageOwnership = false;
};

class DFBPhysicalConflictModelBuilder {
public:
  static DFBPhysicalConflictModel
  build(const DFBConcurrentKernelLivenessAnalysis &liveness,
        ArrayRef<DFBStaticConfigurationConflict> staticConflicts) {
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    DFBPhysicalConflictModel model;
    model.adjacency.assign(logicalDFBs.size(),
                           llvm::BitVector(logicalDFBs.size()));
    for (unsigned lhsIndex = 0; lhsIndex < logicalDFBs.size(); ++lhsIndex) {
      for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < logicalDFBs.size();
           ++rhsIndex) {
        bool sameAllocationGroup = logicalDFBs[lhsIndex].allocationGroup &&
                                   logicalDFBs[lhsIndex].allocationGroup ==
                                       logicalDFBs[rhsIndex].allocationGroup;
        DFBPairConflictRequirements requirements;
        requirements.requireMatchingTransactions = !sameAllocationGroup;
        requirements.useAllocationGroupEpochs = sameAllocationGroup;
        requirements.allowCapacityEnvelope = sameAllocationGroup;
        addPairConflicts(model, liveness, lhsIndex, rhsIndex, requirements);
      }
    }
    addResetAllocationConflicts(model, liveness);
    DenseMap<int64_t, unsigned> logicalIndexById;
    for (auto [logicalIndex, logicalDFB] : llvm::enumerate(logicalDFBs)) {
      logicalIndexById.try_emplace(logicalDFB.logicalId, logicalIndex);
    }
    for (const DFBStaticConfigurationConflict &conflict : staticConflicts) {
      auto lhsIt = logicalIndexById.find(conflict.lhsLogicalId);
      auto rhsIt = logicalIndexById.find(conflict.rhsLogicalId);
      assert(lhsIt != logicalIndexById.end() &&
             rhsIt != logicalIndexById.end() &&
             "configuration conflicts must reference analyzed logical DFBs");
      unsigned lhsIndex = lhsIt->second;
      unsigned rhsIndex = rhsIt->second;
      if (lhsIndex == rhsIndex || model.adjacency[lhsIndex].test(rhsIndex)) {
        continue;
      }
      addEvidence(model, logicalDFBs[lhsIndex], logicalDFBs[rhsIndex], lhsIndex,
                  rhsIndex, DFBConflictReason::StaticConfigurationMismatch,
                  std::nullopt, conflict.lhsOperation, conflict.rhsOperation);
    }
    return model;
  }

  static DFBPhysicalConflictModel
  buildAllocationGroupPair(const DFBConcurrentKernelLivenessAnalysis &liveness,
                           unsigned lhsIndex, unsigned rhsIndex) {
    DFBPhysicalConflictModel model;
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    size_t logicalDFBCount = logicalDFBs.size();
    model.adjacency.assign(logicalDFBCount, llvm::BitVector(logicalDFBCount));
    const DFBLogicalLifecycle &lhs = logicalDFBs[lhsIndex];
    const DFBLogicalLifecycle &rhs = logicalDFBs[rhsIndex];
    if ((!lhs.launchDomain.known || !rhs.launchDomain.known) &&
        lhs.tensorBacking != rhs.tensorBacking) {
      addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                  DFBConflictReason::StorageMismatch, std::nullopt,
                  lhs.declarations.front(), rhs.declarations.front());
      return model;
    }
    DFBPairConflictRequirements requirements;
    requirements.requireMatchingTransactions = false;
    requirements.useAllocationGroupEpochs = true;
    requirements.allowCapacityEnvelope = true;
    addPairConflicts(model, liveness, lhsIndex, rhsIndex, requirements);
    addResetAllocationConflicts(model, liveness,
                                std::make_pair(lhsIndex, rhsIndex));
    return model;
  }

  static DFBPhysicalConflictModel
  buildStorage(const DFBConcurrentKernelLivenessAnalysis &liveness) {
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    DFBPhysicalConflictModel model;
    model.adjacency.assign(logicalDFBs.size(),
                           llvm::BitVector(logicalDFBs.size()));
    for (unsigned lhsIndex = 0; lhsIndex < logicalDFBs.size(); ++lhsIndex) {
      for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < logicalDFBs.size();
           ++rhsIndex) {
        DFBPairConflictRequirements requirements;
        requirements.requireExactDescriptor = false;
        requirements.requireMatchingElementType = false;
        requirements.requireMatchingTransactions = false;
        requirements.requireMatchingPointerOwners = false;
        requirements.requireStaticStorageOwnership = true;
        addPairConflicts(model, liveness, lhsIndex, rhsIndex, requirements);
      }
    }
    return model;
  }

private:
  static void addEvidence(DFBPhysicalConflictModel &model,
                          const DFBLogicalLifecycle &lhs,
                          const DFBLogicalLifecycle &rhs, unsigned lhsIndex,
                          unsigned rhsIndex, DFBConflictReason reason,
                          std::optional<LaunchNodeCoord> node,
                          Operation *lhsOperation, Operation *rhsOperation) {
    model.adjacency[lhsIndex].set(rhsIndex);
    model.adjacency[rhsIndex].set(lhsIndex);
    model.evidence.push_back({lhsIndex, rhsIndex, lhs.logicalId, rhs.logicalId,
                              reason, node, lhsOperation, rhsOperation});
  }

  static void addResetAllocationConflicts(
      DFBPhysicalConflictModel &model,
      const DFBConcurrentKernelLivenessAnalysis &liveness,
      std::optional<std::pair<unsigned, unsigned>> onlyPair = std::nullopt) {
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    DenseSet<std::pair<unsigned, unsigned>> recordedPairs;
    for (const DFBResetAllocationConflict &conflict :
         liveness.getResetAllocationConflicts()) {
      unsigned targetIndex = conflict.targetLogicalIndex;
      unsigned overlappingIndex = conflict.overlappingLogicalIndex;
      if (onlyPair && !((targetIndex == onlyPair->first &&
                         overlappingIndex == onlyPair->second) ||
                        (targetIndex == onlyPair->second &&
                         overlappingIndex == onlyPair->first))) {
        continue;
      }
      if (targetIndex == overlappingIndex) {
        continue;
      }
      std::pair<unsigned, unsigned> logicalPair = {
          std::min(targetIndex, overlappingIndex),
          std::max(targetIndex, overlappingIndex)};
      if (!recordedPairs.insert(logicalPair).second) {
        continue;
      }
      addEvidence(model, logicalDFBs[targetIndex],
                  logicalDFBs[overlappingIndex], targetIndex, overlappingIndex,
                  DFBConflictReason::ResetDomainWrite, conflict.node,
                  conflict.resetOperation, conflict.overlappingOperation);
    }
  }

  static void
  addPairConflicts(DFBPhysicalConflictModel &model,
                   const DFBConcurrentKernelLivenessAnalysis &liveness,
                   unsigned lhsIndex, unsigned rhsIndex,
                   const DFBPairConflictRequirements &requirements) {
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    const DFBLogicalLifecycle &lhs = logicalDFBs[lhsIndex];
    const DFBLogicalLifecycle &rhs = logicalDFBs[rhsIndex];
    bool usesCapacityEnvelope =
        requirements.allowCapacityEnvelope && canUseCapacityEnvelope(lhs, rhs);
    bool reconfiguresDescriptor =
        canReconfigureDescriptorAcrossEpochs(lhs, rhs);
    if ((requirements.requireMatchingElementType &&
         !haveIdenticalPageFormat(lhs.type, rhs.type)) ||
        (requirements.requireExactDescriptor && lhs.type != rhs.type &&
         !usesCapacityEnvelope && !reconfiguresDescriptor)) {
      addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                  DFBConflictReason::DescriptorMismatch, std::nullopt,
                  lhs.declarations.front(), rhs.declarations.front());
      return;
    }
    std::uint64_t physicalTileCount =
        cast<CircularBufferType>(lhs.type).getTotalElements();
    bool lhsInactive = lhs.launchDomain.known && lhs.launchDomain.nodes.empty();
    bool rhsInactive = rhs.launchDomain.known && rhs.launchDomain.nodes.empty();
    if (lhsInactive || rhsInactive) {
      if (lhs.tensorBacking || rhs.tensorBacking) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::StorageMismatch, std::nullopt,
                    lhs.declarations.front(), rhs.declarations.front());
      }
      return;
    }
    bool useConditionalProof =
        !lhs.launchDomain.known && !rhs.launchDomain.known &&
        lhs.conditionallyBounded && rhs.conditionallyBounded;
    if ((!lhs.launchDomain.known || !rhs.launchDomain.known) &&
        !useConditionalProof) {
      addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                  DFBConflictReason::UnknownLaunchNodeDomain, std::nullopt,
                  lhs.declarations.front(), rhs.declarations.front());
      return;
    }

    SmallVector<LaunchNodeCoord> sharedNodes;
    if (useConditionalProof) {
      llvm::append_range(sharedNodes, liveness.getLaunchNodes());
    } else {
      LaunchNodeDomain exactSharedNodes =
          lhs.launchDomain.intersectWith(rhs.launchDomain);
      llvm::append_range(sharedNodes, exactSharedNodes.nodes);
    }
    if (sharedNodes.empty()) {
      return;
    }
    if (requirements.requireStaticStorageOwnership &&
        (requiresReconfigurationStorage(lhs) ||
         requiresReconfigurationStorage(rhs))) {
      addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                  DFBConflictReason::StorageMismatch, sharedNodes.front(),
                  lhs.declarations.front(), rhs.declarations.front());
      return;
    }
    if (!lhs.accessCompletionProven || !rhs.accessCompletionProven) {
      addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                  DFBConflictReason::AccessCompletionNotProven, std::nullopt,
                  lhs.declarations.front(), rhs.declarations.front());
      return;
    }
    for (LaunchNodeCoord node : sharedNodes) {
      const DFBPerNodeLifetime *lhsLifetime =
          useConditionalProof ? lhs.findPossibleNodeLifetime(node)
                              : lhs.findNodeLifetime(node);
      const DFBPerNodeLifetime *rhsLifetime =
          useConditionalProof ? rhs.findPossibleNodeLifetime(node)
                              : rhs.findNodeLifetime(node);
      if (lhsLifetime && rhsLifetime &&
          (!lhsLifetime->mayBeActive || !rhsLifetime->mayBeActive)) {
        continue;
      }
      if (!lhsLifetime || !rhsLifetime) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::AccessCompletionNotProven, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
        continue;
      }
      if (lhs.tensorBacking != rhs.tensorBacking &&
          !haveDisjointConfigurationEpochs(*lhsLifetime, *rhsLifetime)) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::StorageMismatch, node,
                    lhs.declarations.front(), rhs.declarations.front());
        continue;
      }
      bool compareAllocationGroupEpochs =
          requirements.useAllocationGroupEpochs &&
          (!lhsLifetime->epochs.empty() || !rhsLifetime->epochs.empty());
      SmallVector<AllocationGroupNodeEpoch> lhsEpochs;
      SmallVector<AllocationGroupNodeEpoch> rhsEpochs;
      if (compareAllocationGroupEpochs) {
        appendAllocationGroupNodeEpochs(lhsEpochs, lhsIndex, *lhsLifetime,
                                        useConditionalProof);
        appendAllocationGroupNodeEpochs(rhsEpochs, rhsIndex, *rhsLifetime,
                                        useConditionalProof);
        auto unprovenLhsEpoch =
            llvm::find_if(lhsEpochs, [](const AllocationGroupNodeEpoch &epoch) {
              return !epoch.getCompletionProof().proven();
            });
        auto unprovenRhsEpoch =
            llvm::find_if(rhsEpochs, [](const AllocationGroupNodeEpoch &epoch) {
              return !epoch.getCompletionProof().proven();
            });
        if (unprovenLhsEpoch != lhsEpochs.end() ||
            unprovenRhsEpoch != rhsEpochs.end()) {
          const AllocationGroupNodeEpoch &lhsEvidence =
              unprovenLhsEpoch != lhsEpochs.end() ? *unprovenLhsEpoch
                                                  : lhsEpochs.front();
          const AllocationGroupNodeEpoch &rhsEvidence =
              unprovenRhsEpoch != rhsEpochs.end() ? *unprovenRhsEpoch
                                                  : rhsEpochs.front();
          addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                      DFBConflictReason::AccessCompletionNotProven, node,
                      getAllocationGroupEpochEvidence(lhsEvidence, lhs),
                      getAllocationGroupEpochEvidence(rhsEvidence, rhs));
          continue;
        }
      } else if (!lhsLifetime->completionProof.proven() ||
                 !rhsLifetime->completionProof.proven()) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::AccessCompletionNotProven, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
        continue;
      }
      bool disjointConfigurationEpochs =
          haveDisjointConfigurationEpochs(*lhsLifetime, *rhsLifetime);
      if (requirements.requireExactDescriptor && lhs.type != rhs.type &&
          !usesCapacityEnvelope && !disjointConfigurationEpochs) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::DescriptorMismatch, node,
                    lhs.declarations.front(), rhs.declarations.front());
        continue;
      }
      if (disjointConfigurationEpochs) {
        continue;
      }
      if (lhs.tensorBacking != rhs.tensorBacking) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::StorageMismatch, node,
                    lhs.declarations.front(), rhs.declarations.front());
        continue;
      }
      if (compareAllocationGroupEpochs) {
        for (const AllocationGroupNodeEpoch &lhsEpoch : lhsEpochs) {
          for (const AllocationGroupNodeEpoch &rhsEpoch : rhsEpochs) {
            bool lhsBeforeRhs = isAllocationGroupEpochOrderedBefore(
                liveness, lhsEpoch, rhsEpoch, node);
            bool rhsBeforeLhs = isAllocationGroupEpochOrderedBefore(
                liveness, rhsEpoch, lhsEpoch, node);
            if (lhsBeforeRhs != rhsBeforeLhs) {
              continue;
            }
            addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                        DFBConflictReason::ConcurrentLifetime, node,
                        getAllocationGroupEpochEvidence(lhsEpoch, lhs),
                        getAllocationGroupEpochEvidence(rhsEpoch, rhs));
          }
        }
        continue;
      }
      bool lhsBeforeRhs =
          useConditionalProof
              ? liveness.isConditionallyOrderedBefore(lhsIndex, rhsIndex, node)
              : liveness.isOrderedBefore(lhsIndex, rhsIndex, node);
      bool rhsBeforeLhs =
          useConditionalProof
              ? liveness.isConditionallyOrderedBefore(rhsIndex, lhsIndex, node)
              : liveness.isOrderedBefore(rhsIndex, lhsIndex, node);
      const DFBPerNodeLifetime *before =
          lhsBeforeRhs ? lhsLifetime : rhsLifetime;
      const DFBPerNodeLifetime *after =
          lhsBeforeRhs ? rhsLifetime : lhsLifetime;
      bool terminalStateCompatible = false;
      bool pointerOwnersCompatible = false;
      if (lhsBeforeRhs || rhsBeforeLhs) {
        bool hasIdentityTransition =
            before->inspectionOnly || after->inspectionOnly;
        terminalStateCompatible =
            hasIdentityTransition || before->terminalStateCanonical ||
            !requirements.requireMatchingTransactions ||
            haveCompatibleCursorRuns(*before, *after, physicalTileCount);
        pointerOwnersCompatible =
            !requirements.requireMatchingPointerOwners ||
            hasIdentityTransition || before->terminalStateCanonical ||
            (before->terminalWritePointerOwner == after->writePointerOwner &&
             before->terminalReadPointerOwner == after->readPointerOwner);
      } else {
        // Preserve the more specific state diagnosis when lifetimes are also
        // unordered; ordering alone must not obscure a protocol mismatch.
        bool hasIdentityTransition =
            lhsLifetime->inspectionOnly || rhsLifetime->inspectionOnly;
        terminalStateCompatible =
            hasIdentityTransition ||
            !requirements.requireMatchingTransactions ||
            (haveCompatibleCursorRuns(*lhsLifetime, *rhsLifetime,
                                      physicalTileCount) &&
             haveCompatibleCursorRuns(*rhsLifetime, *lhsLifetime,
                                      physicalTileCount));
        pointerOwnersCompatible =
            !requirements.requireMatchingPointerOwners ||
            hasIdentityTransition ||
            (lhsLifetime->writePointerOwner == rhsLifetime->writePointerOwner &&
             lhsLifetime->readPointerOwner == rhsLifetime->readPointerOwner);
      }
      if (!terminalStateCompatible) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::TransactionMismatch, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
        continue;
      }
      if (!pointerOwnersCompatible) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::PointerOwnerMismatch, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
        continue;
      }
      if (!lhsBeforeRhs && !rhsBeforeLhs) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::ConcurrentLifetime, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
      }
    }
  }
};

namespace {

static bool hasAllocationGroups(ArrayRef<DFBLogicalLifecycle> logicalDFBs) {
  return llvm::any_of(logicalDFBs, [](const DFBLogicalLifecycle &logicalDFB) {
    return static_cast<bool>(logicalDFB.allocationGroup);
  });
}

static void printAllocationGroup(raw_ostream &os,
                                 DFBAllocationGroupAttr allocationGroup,
                                 ArrayRef<unsigned> members,
                                 ArrayRef<DFBLogicalLifecycle> logicalDFBs) {
  os << allocationGroup << " members=[";
  llvm::interleaveComma(members, os, [&](unsigned member) {
    os << logicalDFBs[member].logicalId;
  });
  os << ']';
}

static LogicalResult validateAllocationGroupTypes(
    const DFBLogicalLifecycle &lhs, const DFBLogicalLifecycle &rhs,
    ArrayRef<unsigned> members, ArrayRef<DFBLogicalLifecycle> logicalDFBs,
    DFBAnalysisFailure &analysisFailure) {
  auto lhsType = cast<CircularBufferType>(lhs.type);
  auto rhsType = cast<CircularBufferType>(rhs.type);
  if (lhsType.getElementType() != rhsType.getElementType()) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "DFB allocation group ";
    printAllocationGroup(messageStream, lhs.allocationGroup, members,
                         logicalDFBs);
    messageStream << " has incompatible element types for logical DFBs "
                  << lhs.logicalId << " and " << rhs.logicalId << ": "
                  << lhsType.getElementType() << " versus "
                  << rhsType.getElementType();
    analysisFailure.set(rhs.declarations.front(), messageStream.str());
    return failure();
  }
  if (lhs.type != rhs.type && (lhs.tensorBacking || rhs.tensorBacking)) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "DFB allocation group ";
    printAllocationGroup(messageStream, lhs.allocationGroup, members,
                         logicalDFBs);
    messageStream << " cannot use a static capacity envelope for tensor-backed "
                     "logical DFBs "
                  << lhs.logicalId << " and " << rhs.logicalId;
    analysisFailure.set(rhs.declarations.front(), messageStream.str());
    return failure();
  }
  return success();
}

static void
printAllocationGroupNodeEpoch(raw_ostream &output,
                              const AllocationGroupNodeEpoch &epoch,
                              ArrayRef<DFBLogicalLifecycle> logicalDFBs) {
  output << "logical DFB " << logicalDFBs[epoch.logicalIndex].logicalId;
  if (epoch.epoch) {
    output << " epoch " << epoch.epochIndex;
  }
}

static bool hasAllocationGroupEpochInconsistentOrder(
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    const AllocationGroupNodeEpoch &lhs, const AllocationGroupNodeEpoch &rhs,
    LaunchNodeCoord node) {
  if (lhs.possibleDomain || rhs.possibleDomain) {
    return liveness.hasConditionallyInconsistentEpochOrder(
        lhs.logicalIndex, lhs.epochIndex, rhs.logicalIndex, rhs.epochIndex,
        node);
  }
  return liveness.hasInconsistentEpochOrder(
      lhs.logicalIndex, lhs.epochIndex, rhs.logicalIndex, rhs.epochIndex, node);
}

struct AllocationGroupCursorState {
  std::uint64_t writePointerOffset = 0;
  std::uint64_t readPointerOffset = 0;
};

enum class AllocationGroupCursorFailure {
  None,
  RingBoundary,
  UnequalOffsets,
};

static LogicalResult advanceAllocationGroupCursorRuns(
    ArrayRef<DFBTransactionRun> writeCursorRuns,
    ArrayRef<DFBTransactionRun> readCursorRuns, std::uint64_t physicalTileCount,
    AllocationGroupCursorState &cursorState, bool terminalStateCanonical,
    AllocationGroupCursorFailure *failureReason = nullptr) {
  FailureOr<std::uint64_t> writePointerOffset = advanceDFBTransactionCursor(
      writeCursorRuns, physicalTileCount, cursorState.writePointerOffset);
  FailureOr<std::uint64_t> readPointerOffset = advanceDFBTransactionCursor(
      readCursorRuns, physicalTileCount, cursorState.readPointerOffset);
  if (failed(writePointerOffset) || failed(readPointerOffset)) {
    if (failureReason) {
      *failureReason = AllocationGroupCursorFailure::RingBoundary;
    }
    return failure();
  }
  cursorState = {*writePointerOffset, *readPointerOffset};
  if (terminalStateCanonical) {
    cursorState = {};
    if (failureReason) {
      *failureReason = AllocationGroupCursorFailure::None;
    }
    return success();
  }
  if (*writePointerOffset != *readPointerOffset) {
    if (failureReason) {
      *failureReason = AllocationGroupCursorFailure::UnequalOffsets;
    }
    return failure();
  }
  if (failureReason) {
    *failureReason = AllocationGroupCursorFailure::None;
  }
  return success();
}

static FailureOr<AllocationGroupCursorState> advanceAllocationGroupMemberCursor(
    const DFBPerNodeLifetime &lifetime, std::uint64_t physicalTileCount,
    AllocationGroupCursorState cursorState = {}) {
  if (lifetime.epochs.empty()) {
    if (failed(advanceAllocationGroupCursorRuns(
            lifetime.writeCursorRuns, lifetime.readCursorRuns,
            physicalTileCount, cursorState,
            /*terminalStateCanonical=*/false))) {
      return failure();
    }
  } else {
    for (const DFBLifecycleEpoch &epoch : lifetime.epochs) {
      if (failed(advanceAllocationGroupCursorRuns(
              epoch.writeCursorRuns, epoch.readCursorRuns, physicalTileCount,
              cursorState, epoch.terminalStateCanonical))) {
        return failure();
      }
    }
  }
  return cursorState;
}

static FailureOr<AllocationGroupCursorState> advanceAllocationGroupEpochCursor(
    const AllocationGroupNodeEpoch &epoch, std::uint64_t physicalTileCount,
    AllocationGroupCursorState cursorState = {},
    AllocationGroupCursorFailure *failureReason = nullptr) {
  if (failed(advanceAllocationGroupCursorRuns(
          epoch.getWriteCursorRuns(), epoch.getReadCursorRuns(),
          physicalTileCount, cursorState, epoch.hasCanonicalTerminalState(),
          failureReason))) {
    return failure();
  }
  return cursorState;
}

static void addAllocationGroupAssumption(
    SmallVectorImpl<DFBAllocationGroupAssumption> &assumptions,
    DFBAllocationGroupAssumptionReason reason, int64_t lhsLogicalId,
    std::optional<int64_t> rhsLogicalId = std::nullopt) {
  bool exists = llvm::any_of(
      assumptions, [&](const DFBAllocationGroupAssumption &assumption) {
        return assumption.reason == reason &&
               assumption.lhsLogicalId == lhsLogicalId &&
               assumption.rhsLogicalId == rhsLogicalId;
      });
  if (!exists) {
    assumptions.push_back({reason, lhsLogicalId, rhsLogicalId});
  }
}

static std::optional<DFBAllocationGroupAssumptionReason>
getAllocationGroupAssumptionReason(DFBConflictReason reason) {
  switch (reason) {
  case DFBConflictReason::UnknownLaunchNodeDomain:
    return DFBAllocationGroupAssumptionReason::UnknownLaunchNodeDomain;
  case DFBConflictReason::AccessCompletionNotProven:
    return DFBAllocationGroupAssumptionReason::AccessCompletionNotProven;
  case DFBConflictReason::PointerOwnerMismatch:
    return DFBAllocationGroupAssumptionReason::PointerOwnerMismatch;
  case DFBConflictReason::ConcurrentLifetime:
    return DFBAllocationGroupAssumptionReason::ConcurrentLifetime;
  case DFBConflictReason::DescriptorMismatch:
  case DFBConflictReason::StorageMismatch:
  case DFBConflictReason::ResetDomainWrite:
  case DFBConflictReason::StaticConfigurationMismatch:
  case DFBConflictReason::TransactionMismatch:
    return std::nullopt;
  }
  llvm_unreachable("unknown DFB conflict reason");
}

static LogicalResult validateAllocationGroupCursor(
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    ArrayRef<unsigned> members, DFBAllocationGroupAttr allocationGroup,
    std::uint64_t physicalTileCount, bool unsafeAssumeAllocationGroups,
    SmallVectorImpl<DFBAllocationGroupAssumption> &assumptions,
    DFBAnalysisFailure &analysisFailure) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  for (LaunchNodeCoord node : liveness.getLaunchNodes()) {
    SmallVector<AllocationGroupNodeEpoch> activeEpochs;
    for (unsigned logicalIndex : members) {
      const DFBLogicalLifecycle &logicalDFB = logicalDFBs[logicalIndex];
      bool possibleDomain = !logicalDFB.launchDomain.known;
      const DFBPerNodeLifetime *lifetime =
          possibleDomain ? logicalDFB.findPossibleNodeLifetime(node)
                         : logicalDFB.findNodeLifetime(node);
      if (!lifetime || !lifetime->mayBeActive) {
        continue;
      }
      appendAllocationGroupNodeEpochs(activeEpochs, logicalIndex, *lifetime,
                                      possibleDomain);
    }

    bool hasUnprovenOrder = false;
    for (const AllocationGroupNodeEpoch &epoch : activeEpochs) {
      if (hasAllocationGroupEpochInconsistentOrder(liveness, epoch, epoch,
                                                   node)) {
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "DFB allocation group ";
        printAllocationGroup(messageStream, allocationGroup, members,
                             logicalDFBs);
        messageStream << " has contradictory cursor order involving ";
        printAllocationGroupNodeEpoch(messageStream, epoch, logicalDFBs);
        messageStream << " on launch node (" << node.x << ',' << node.y << ')';
        analysisFailure.set(getAllocationGroupEpochEvidence(
                                epoch, logicalDFBs[epoch.logicalIndex]),
                            messageStream.str());
        return failure();
      }
      if (epoch.getCompletionProof().proven() || unsafeAssumeAllocationGroups) {
        continue;
      }
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB allocation group ";
      printAllocationGroup(messageStream, allocationGroup, members,
                           logicalDFBs);
      messageStream << " does not have proven lifecycle completion for ";
      printAllocationGroupNodeEpoch(messageStream, epoch, logicalDFBs);
      messageStream << " on launch node (" << node.x << ',' << node.y << ')';
      analysisFailure.set(getAllocationGroupEpochEvidence(
                              epoch, logicalDFBs[epoch.logicalIndex]),
                          messageStream.str());
      return failure();
    }

    SmallVector<unsigned> predecessorCounts(activeEpochs.size());
    for (auto [lhsPosition, lhs] : llvm::enumerate(activeEpochs)) {
      for (unsigned rhsPosition = lhsPosition + 1;
           rhsPosition < activeEpochs.size(); ++rhsPosition) {
        const AllocationGroupNodeEpoch &rhs = activeEpochs[rhsPosition];
        bool lhsBeforeRhs =
            isAllocationGroupEpochOrderedBefore(liveness, lhs, rhs, node);
        bool rhsBeforeLhs =
            isAllocationGroupEpochOrderedBefore(liveness, rhs, lhs, node);
        if (lhsBeforeRhs != rhsBeforeLhs) {
          unsigned afterPosition = lhsBeforeRhs ? rhsPosition : lhsPosition;
          ++predecessorCounts[afterPosition];
          continue;
        }
        bool inconsistentOrder =
            hasAllocationGroupEpochInconsistentOrder(liveness, lhs, rhs, node);
        if (!inconsistentOrder && unsafeAssumeAllocationGroups) {
          addAllocationGroupAssumption(
              assumptions,
              DFBAllocationGroupAssumptionReason::UnprovenCursorOrder,
              logicalDFBs[lhs.logicalIndex].logicalId,
              logicalDFBs[rhs.logicalIndex].logicalId);
          hasUnprovenOrder = true;
          continue;
        }
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "DFB allocation group ";
        printAllocationGroup(messageStream, allocationGroup, members,
                             logicalDFBs);
        messageStream << (inconsistentOrder
                              ? " has inconsistent cursor order for "
                              : " has no proven cursor order for ")
                      << "epochs ";
        printAllocationGroupNodeEpoch(messageStream, lhs, logicalDFBs);
        messageStream << " and ";
        printAllocationGroupNodeEpoch(messageStream, rhs, logicalDFBs);
        messageStream << " on launch node (" << node.x << ',' << node.y << ')';
        analysisFailure.set(
            getAllocationGroupEpochEvidence(rhs, logicalDFBs[rhs.logicalIndex]),
            messageStream.str());
        return failure();
      }
    }
    if (hasUnprovenOrder) {
      DenseSet<unsigned> validatedLogicalIndices;
      for (const AllocationGroupNodeEpoch &epoch : activeEpochs) {
        if (!validatedLogicalIndices.insert(epoch.logicalIndex).second ||
            succeeded(advanceAllocationGroupMemberCursor(*epoch.lifetime,
                                                         physicalTileCount))) {
          continue;
        }
        const DFBLogicalLifecycle &logicalDFB = logicalDFBs[epoch.logicalIndex];
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "DFB allocation group ";
        printAllocationGroup(messageStream, allocationGroup, members,
                             logicalDFBs);
        messageStream << " physical envelope of " << physicalTileCount
                      << " tiles makes logical DFB " << logicalDFB.logicalId
                      << " cross the ring boundary on launch node (" << node.x
                      << ',' << node.y << ')';
        analysisFailure.set(getAllocationGroupEpochEvidence(epoch, logicalDFB),
                            messageStream.str());
        return failure();
      }
      continue;
    }
    SmallVector<AllocationGroupNodeEpoch> orderedEpochs(activeEpochs.size());
    llvm::BitVector occupiedRanks(activeEpochs.size());
    for (auto [epochPosition, epoch] : llvm::enumerate(activeEpochs)) {
      unsigned rank = predecessorCounts[epochPosition];
      if (rank < orderedEpochs.size() && !occupiedRanks.test(rank)) {
        orderedEpochs[rank] = epoch;
        occupiedRanks.set(rank);
        continue;
      }
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB allocation group ";
      printAllocationGroup(messageStream, allocationGroup, members,
                           logicalDFBs);
      messageStream << " has inconsistent cursor order for ";
      printAllocationGroupNodeEpoch(messageStream, epoch, logicalDFBs);
      messageStream << " on launch node (" << node.x << ',' << node.y << ')';
      analysisFailure.set(getAllocationGroupEpochEvidence(
                              epoch, logicalDFBs[epoch.logicalIndex]),
                          messageStream.str());
      return failure();
    }
    activeEpochs = std::move(orderedEpochs);

    LLVM_DEBUG({
      llvm::dbgs() << "DFB allocation group " << allocationGroup
                   << " launch_node=(" << node.x << ',' << node.y
                   << ") epoch_order=[";
      llvm::interleaveComma(activeEpochs, llvm::dbgs(),
                            [&](const AllocationGroupNodeEpoch &epoch) {
                              llvm::dbgs()
                                  << logicalDFBs[epoch.logicalIndex].logicalId
                                  << ':' << epoch.epochIndex;
                            });
      llvm::dbgs() << "]\n";
    });

    AllocationGroupCursorState cursorState;
    const AllocationGroupNodeEpoch *previousEpoch = nullptr;
    for (const AllocationGroupNodeEpoch &epoch : activeEpochs) {
      if (epoch.isInspectionOnly()) {
        if (epoch.hasCanonicalTerminalState()) {
          cursorState = {};
          previousEpoch = &epoch;
        }
        continue;
      }
      bool pointerOwnersCompatible =
          !previousEpoch || previousEpoch->hasCanonicalTerminalState() ||
          (previousEpoch->getTerminalWritePointerOwner() ==
               epoch.getWritePointerOwner() &&
           previousEpoch->getTerminalReadPointerOwner() ==
               epoch.getReadPointerOwner());
      AllocationGroupCursorFailure cursorFailure =
          AllocationGroupCursorFailure::None;
      FailureOr<AllocationGroupCursorState> nextState = failure();
      if (pointerOwnersCompatible) {
        nextState = advanceAllocationGroupEpochCursor(
            epoch, physicalTileCount, cursorState, &cursorFailure);
      }
      if (succeeded(nextState)) {
        cursorState = *nextState;
        previousEpoch = &epoch;
        continue;
      }

      bool differentLogicalMember =
          previousEpoch && previousEpoch->logicalIndex != epoch.logicalIndex;
      if (unsafeAssumeAllocationGroups && differentLogicalMember) {
        AllocationGroupCursorFailure resetFailure =
            AllocationGroupCursorFailure::None;
        FailureOr<AllocationGroupCursorState> resetState =
            advanceAllocationGroupEpochCursor(epoch, physicalTileCount, {},
                                              &resetFailure);
        if (succeeded(resetState)) {
          const DFBLogicalLifecycle &logicalDFB =
              logicalDFBs[epoch.logicalIndex];
          addAllocationGroupAssumption(
              assumptions, DFBAllocationGroupAssumptionReason::EpochReset,
              logicalDFB.logicalId);
          cursorState = *resetState;
          previousEpoch = &epoch;
          continue;
        }
        cursorFailure = resetFailure;
      }

      const DFBLogicalLifecycle &logicalDFB = logicalDFBs[epoch.logicalIndex];
      if (!pointerOwnersCompatible &&
          cursorFailure == AllocationGroupCursorFailure::None) {
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "DFB allocation group ";
        printAllocationGroup(messageStream, allocationGroup, members,
                             logicalDFBs);
        messageStream << " cannot alias logical DFBs "
                      << logicalDFBs[previousEpoch->logicalIndex].logicalId
                      << " and " << logicalDFB.logicalId << ": "
                      << getDFBConflictReasonName(
                             DFBConflictReason::PointerOwnerMismatch);
        analysisFailure.set(getAllocationGroupEpochEvidence(epoch, logicalDFB),
                            messageStream.str());
        return failure();
      }

      if (cursorFailure == AllocationGroupCursorFailure::UnequalOffsets) {
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "DFB allocation group ";
        printAllocationGroup(messageStream, allocationGroup, members,
                             logicalDFBs);
        messageStream << " leaves unequal write and read offsets after ";
        printAllocationGroupNodeEpoch(messageStream, epoch, logicalDFBs);
        messageStream << " on launch node (" << node.x << ',' << node.y << ')';
        analysisFailure.set(getAllocationGroupEpochEvidence(epoch, logicalDFB),
                            messageStream.str());
        return failure();
      }

      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB allocation group ";
      printAllocationGroup(messageStream, allocationGroup, members,
                           logicalDFBs);
      messageStream << " physical envelope of " << physicalTileCount
                    << " tiles makes logical DFB " << logicalDFB.logicalId;
      if (epoch.epoch) {
        messageStream << " epoch " << epoch.epochIndex;
      }
      messageStream << " cross the ring boundary on launch node (" << node.x
                    << ',' << node.y << ')';
      analysisFailure.set(getAllocationGroupEpochEvidence(epoch, logicalDFB),
                          messageStream.str());
      return failure();
    }
  }
  return success();
}

static LogicalResult validateAllocationGroups(
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    ArrayRef<DFBStaticConfigurationConflict> staticConflicts,
    bool unsafeAssumeAllocationGroups,
    SmallVectorImpl<DFBAssumedAllocationGroup> &assumedAllocationGroups,
    DFBAnalysisFailure &analysisFailure) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  DenseMap<int64_t, unsigned> groupIndexByOrdinal;
  SmallVector<std::pair<int64_t, SmallVector<unsigned>>> groups;
  for (auto [logicalIndex, logicalDFB] : llvm::enumerate(logicalDFBs)) {
    if (!logicalDFB.allocationGroup) {
      continue;
    }
    int64_t ordinal = logicalDFB.allocationGroup.getOrdinal();
    auto [groupIt, inserted] =
        groupIndexByOrdinal.try_emplace(ordinal, groups.size());
    if (inserted) {
      groups.push_back({ordinal, {}});
    }
    groups[groupIt->second].second.push_back(logicalIndex);
  }

  for (const auto &[ordinal, members] : groups) {
    SmallVector<DFBAllocationGroupAssumption> assumptions;
    uint64_t envelopeBytes = 0;
    std::uint64_t envelopeTiles = 0;
    bool collectAllocationDiagnostics = false;
    LLVM_DEBUG(collectAllocationDiagnostics = true);
    SmallVector<std::pair<int64_t, int64_t>> removedDescriptorConflicts;
    for (unsigned logicalIndex : members) {
      auto memberType =
          cast<CircularBufferType>(logicalDFBs[logicalIndex].type);
      std::string failureReason;
      FailureOr<uint64_t> memberBytes =
          getDFBAllocationSizeBytes(memberType, failureReason);
      if (failed(memberBytes)) {
        analysisFailure.set(logicalDFBs[logicalIndex].declarations.front(),
                            failureReason);
        return failure();
      }
      if (*memberBytes > envelopeBytes) {
        envelopeBytes = *memberBytes;
        envelopeTiles = memberType.getTotalElements();
      }
    }

    for (unsigned lhsPosition = 0; lhsPosition < members.size();
         ++lhsPosition) {
      unsigned lhsIndex = members[lhsPosition];
      const DFBLogicalLifecycle &lhs = logicalDFBs[lhsIndex];
      for (unsigned rhsPosition = lhsPosition + 1; rhsPosition < members.size();
           ++rhsPosition) {
        unsigned rhsIndex = members[rhsPosition];
        const DFBLogicalLifecycle &rhs = logicalDFBs[rhsIndex];
        if (failed(validateAllocationGroupTypes(lhs, rhs, members, logicalDFBs,
                                                analysisFailure))) {
          return failure();
        }
        if (collectAllocationDiagnostics && lhs.type != rhs.type) {
          removedDescriptorConflicts.push_back({lhs.logicalId, rhs.logicalId});
        }

        auto staticConflictIt =
            llvm::find_if(staticConflicts,
                          [&](const DFBStaticConfigurationConflict &conflict) {
                            return (conflict.lhsLogicalId == lhs.logicalId &&
                                    conflict.rhsLogicalId == rhs.logicalId) ||
                                   (conflict.lhsLogicalId == rhs.logicalId &&
                                    conflict.rhsLogicalId == lhs.logicalId);
                          });
        if (staticConflictIt != staticConflicts.end()) {
          std::string message;
          llvm::raw_string_ostream messageStream(message);
          messageStream << "DFB allocation group ";
          printAllocationGroup(messageStream, lhs.allocationGroup, members,
                               logicalDFBs);
          messageStream << " cannot alias logical DFBs " << lhs.logicalId
                        << " and " << rhs.logicalId << ": "
                        << getDFBConflictReasonName(
                               DFBConflictReason::StaticConfigurationMismatch);
          analysisFailure.set(staticConflictIt->rhsOperation,
                              messageStream.str());
          return failure();
        }

        DFBPhysicalConflictModel groupPair =
            DFBPhysicalConflictModelBuilder::buildAllocationGroupPair(
                liveness, lhsIndex, rhsIndex);
        if (groupPair.conflicts(lhsIndex, rhsIndex)) {
          const DFBConflictEvidence *failureEvidence = nullptr;
          if (unsafeAssumeAllocationGroups) {
            for (const DFBConflictEvidence &evidence :
                 groupPair.getEvidence()) {
              std::optional<DFBAllocationGroupAssumptionReason> reason =
                  getAllocationGroupAssumptionReason(evidence.reason);
              if (!reason) {
                failureEvidence = &evidence;
                break;
              }
              addAllocationGroupAssumption(assumptions, *reason,
                                           evidence.lhsLogicalId,
                                           evidence.rhsLogicalId);
            }
          } else {
            failureEvidence = &groupPair.getEvidence().front();
          }
          if (!failureEvidence) {
            continue;
          }
          std::string message;
          llvm::raw_string_ostream messageStream(message);
          messageStream << "DFB allocation group ";
          printAllocationGroup(messageStream, lhs.allocationGroup, members,
                               logicalDFBs);
          messageStream << " cannot alias logical DFBs " << lhs.logicalId
                        << " and " << rhs.logicalId << ": "
                        << getDFBConflictReasonName(failureEvidence->reason);
          analysisFailure.set(failureEvidence->rhsOperation,
                              messageStream.str());
          return failure();
        }
      }
    }

    DFBAllocationGroupAttr allocationGroup =
        logicalDFBs[members.front()].allocationGroup;
    if (failed(validateAllocationGroupCursor(
            liveness, members, allocationGroup, envelopeTiles,
            unsafeAssumeAllocationGroups, assumptions, analysisFailure))) {
      return failure();
    }

    bool handoffAssumed = !assumptions.empty();
    if (handoffAssumed) {
      DFBAssumedAllocationGroup assumedGroup;
      assumedGroup.allocationGroup = allocationGroup;
      assumedGroup.operation =
          logicalDFBs[members.front()].declarations.front();
      for (unsigned member : members) {
        assumedGroup.logicalIds.push_back(logicalDFBs[member].logicalId);
      }
      assumedGroup.assumptions = std::move(assumptions);
      assumedAllocationGroups.push_back(std::move(assumedGroup));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "DFB allocation group #ttl.dfb_allocation_group<"
                   << ordinal << "> members=[";
      llvm::interleaveComma(members, llvm::dbgs(), [&](unsigned member) {
        llvm::dbgs() << logicalDFBs[member].logicalId;
      });
      llvm::dbgs() << "] envelope_bytes=" << envelopeBytes
                   << " handoff=" << (handoffAssumed ? "assumed" : "proven")
                   << " removed_conflicts=[";
      llvm::interleaveComma(removedDescriptorConflicts, llvm::dbgs(),
                            [](const std::pair<int64_t, int64_t> &conflict) {
                              llvm::dbgs()
                                  << "descriptor-mismatch(" << conflict.first
                                  << ',' << conflict.second << ')';
                            });
      llvm::dbgs() << "]\n";
    });
  }
  return success();
}

/// Physical-index assignment and its proven bounds for selected logical DFBs.
struct ConcurrentAssignmentResult {
  SmallVector<int32_t> assignments;
  unsigned colorCount = 0;
  unsigned provenColorLowerBound = 0;
  bool minimumProven = false;
  bool exactSearchLimitReached = false;
  std::uint64_t exactSearchStateCount = 0;
};

/// Assigns indices by mapping each logical DFB to a graph vertex, each conflict
/// to an edge, and each available physical index to a graph color.
///
/// First-fit processes DFBs in immutable declaration order and chooses the
/// lowest index not used by a conflicting DFB. Its assignment is accepted when
/// it fits `availableIndices` and the optional allocation-byte limit. Otherwise
/// bounded exhaustive searches decide whether another assignment satisfies the
/// failed limit. `firstPhysicalIndex` reserves lower index values without
/// changing which DFB pairs may share.
static FailureOr<ConcurrentAssignmentResult> computeConcurrentAssignments(
    ModuleOp moduleOp, ArrayRef<unsigned> candidateIndices,
    int32_t firstPhysicalIndex, const DFBPhysicalConflictModel &conflictModel,
    ArrayRef<DFBLogicalLifecycle> logicalDFBs, unsigned availableIndices,
    std::uint64_t exactColoringSearchStateLimit,
    DFBAnalysisFailure &analysisFailure,
    ArrayRef<uint64_t> allocationBytesByLogicalIndex,
    std::optional<uint64_t> allocationByteLimit,
    std::optional<uint64_t> minimumSearchTriggerBytes) {
  SmallVector<unsigned> logicalIndices(candidateIndices.begin(),
                                       candidateIndices.end());

  SmallVector<unsigned> allocationVertexByCandidate;
  DenseMap<int64_t, unsigned> vertexByAllocationGroup;
  unsigned allocationVertexCount = 0;
  allocationVertexByCandidate.reserve(logicalIndices.size());
  for (unsigned logicalIndex : logicalIndices) {
    DFBAllocationGroupAttr allocationGroup =
        logicalDFBs[logicalIndex].allocationGroup;
    if (!allocationGroup) {
      allocationVertexByCandidate.push_back(allocationVertexCount++);
      continue;
    }
    auto [groupIt, inserted] = vertexByAllocationGroup.try_emplace(
        allocationGroup.getOrdinal(), allocationVertexCount);
    if (inserted) {
      ++allocationVertexCount;
    }
    allocationVertexByCandidate.push_back(groupIt->second);
  }

  InterferenceGraph interferenceGraph(allocationVertexCount);
  for (unsigned lhsVertex = 0; lhsVertex < logicalIndices.size(); ++lhsVertex) {
    for (unsigned rhsVertex = lhsVertex + 1; rhsVertex < logicalIndices.size();
         ++rhsVertex) {
      unsigned lhsIndex = logicalIndices[lhsVertex];
      unsigned rhsIndex = logicalIndices[rhsVertex];
      unsigned lhsAllocationVertex = allocationVertexByCandidate[lhsVertex];
      unsigned rhsAllocationVertex = allocationVertexByCandidate[rhsVertex];
      if (lhsAllocationVertex != rhsAllocationVertex &&
          conflictModel.conflicts(lhsIndex, rhsIndex)) {
        interferenceGraph.addInterference(lhsAllocationVertex,
                                          rhsAllocationVertex);
      }
    }
  }

  InterferenceGraphColoringBounds bounds =
      computeInterferenceGraphColoringBounds(interferenceGraph);
  SmallVector<unsigned> selectedColors = bounds.colors;
  unsigned colorCount = bounds.colorCount;
  unsigned provenColorLowerBound = bounds.pairwiseConflictLowerBound;
  bool minimumProven = bounds.provesMinimum();
  bool exactSearchLimitReached = false;
  std::uint64_t exactSearchStateCount = 0;
  if (colorCount > availableIndices &&
      provenColorLowerBound <= availableIndices) {
    InterferenceGraphColorLimitResult fitResult =
        colorInterferenceGraphWithColorLimitExactly(
            interferenceGraph, availableIndices, exactColoringSearchStateLimit);
    exactSearchStateCount = fitResult.exploredStateCount;
    if (fitResult.status == InterferenceGraphColorLimitStatus::Feasible) {
      selectedColors = std::move(fitResult.colors);
      colorCount = fitResult.colorCount;
      minimumProven = colorCount == provenColorLowerBound;
    } else if (fitResult.status ==
               InterferenceGraphColorLimitStatus::Infeasible) {
      provenColorLowerBound = availableIndices + 1;
    } else {
      exactSearchLimitReached = true;
    }
  }

  if ((allocationByteLimit || minimumSearchTriggerBytes) &&
      !exactSearchLimitReached && colorCount <= availableIndices) {
    SmallVector<uint64_t> vertexWeights(allocationVertexCount, 0);
    for (auto [candidateIndex, logicalIndex] :
         llvm::enumerate(logicalIndices)) {
      assert(logicalIndex < allocationBytesByLogicalIndex.size());
      unsigned allocationVertex = allocationVertexByCandidate[candidateIndex];
      vertexWeights[allocationVertex] =
          std::max(vertexWeights[allocationVertex],
                   allocationBytesByLogicalIndex[logicalIndex]);
    }
    SmallVector<uint64_t> maximumWeightByColor(colorCount);
    for (auto [vertex, color] : llvm::enumerate(selectedColors)) {
      maximumWeightByColor[color] =
          std::max(maximumWeightByColor[color], vertexWeights[vertex]);
    }
    uint64_t allocationBytes = 0;
    for (uint64_t colorWeight : maximumWeightByColor) {
      std::optional<uint64_t> updatedBytes =
          llvm::checkedAddUnsigned(allocationBytes, colorWeight);
      if (!updatedBytes) {
        allocationBytes = std::numeric_limits<uint64_t>::max();
        break;
      }
      allocationBytes = *updatedBytes;
    }
    if (minimumSearchTriggerBytes &&
        allocationBytes > *minimumSearchTriggerBytes) {
      uint64_t remainingSearchStates =
          exactSearchStateCount >= exactColoringSearchStateLimit
              ? 0
              : exactColoringSearchStateLimit - exactSearchStateCount;
      ExactInterferenceGraphWeightColoring minimum =
          colorInterferenceGraphMinimumWeightExactly(
              interferenceGraph, vertexWeights, availableIndices,
              selectedColors, remainingSearchStates);
      exactSearchStateCount += minimum.exploredStateCount;
      if (minimum.isOptimal()) {
        selectedColors = std::move(minimum.colors);
        colorCount = minimum.colorCount;
        minimumProven = false;
      } else if (minimum.status ==
                 ExactInterferenceGraphWeightStatus::SearchLimitReached) {
        if (allocationByteLimit && allocationBytes > *allocationByteLimit) {
          exactSearchLimitReached = true;
        }
      } else {
        analysisFailure.set(moduleOp,
                            "DFB allocation size is not representable");
        return failure();
      }
    } else if (allocationByteLimit && allocationBytes > *allocationByteLimit) {
      uint64_t remainingSearchStates =
          exactSearchStateCount >= exactColoringSearchStateLimit
              ? 0
              : exactColoringSearchStateLimit - exactSearchStateCount;
      InterferenceGraphWeightLimitResult fitResult =
          colorInterferenceGraphWithinWeightLimitExactly(
              interferenceGraph, vertexWeights, availableIndices,
              *allocationByteLimit, remainingSearchStates);
      exactSearchStateCount += fitResult.exploredStateCount;
      if (fitResult.status == InterferenceGraphColorLimitStatus::Feasible) {
        selectedColors = std::move(fitResult.colors);
        colorCount = fitResult.colorCount;
        minimumProven = colorCount == provenColorLowerBound;
      } else if (fitResult.status ==
                 InterferenceGraphColorLimitStatus::Infeasible) {
        // Preserve first-fit for the precise final L1 diagnostic.
      } else {
        exactSearchLimitReached = true;
      }
    }
  }
  ArrayRef<unsigned> colors = selectedColors;
  assert(colors.size() == allocationVertexCount);

  DenseMap<unsigned, int32_t> assignmentByLogicalIndex;
  for (auto [candidateIndex, logicalIndex] : llvm::enumerate(logicalIndices)) {
    unsigned allocationVertex = allocationVertexByCandidate[candidateIndex];
    assignmentByLogicalIndex[logicalIndex] =
        firstPhysicalIndex + static_cast<int32_t>(colors[allocationVertex]);
  }

  for (unsigned lhsVertex = 0; lhsVertex < logicalIndices.size(); ++lhsVertex) {
    for (unsigned rhsVertex = lhsVertex + 1; rhsVertex < logicalIndices.size();
         ++rhsVertex) {
      unsigned lhsAllocationVertex = allocationVertexByCandidate[lhsVertex];
      unsigned rhsAllocationVertex = allocationVertexByCandidate[rhsVertex];
      if (lhsAllocationVertex == rhsAllocationVertex ||
          colors[lhsAllocationVertex] != colors[rhsAllocationVertex]) {
        continue;
      }
      if (interferenceGraph.interferes(lhsAllocationVertex,
                                       rhsAllocationVertex)) {
        analysisFailure.set(
            moduleOp, "internal DFB allocation assigned one physical index to "
                      "conflicting logical DFBs");
        return failure();
      }
    }
  }

  ConcurrentAssignmentResult result;
  result.assignments.reserve(candidateIndices.size());
  for (unsigned logicalIndex : candidateIndices) {
    result.assignments.push_back(assignmentByLogicalIndex.lookup(logicalIndex));
  }
  result.colorCount = colorCount;
  result.provenColorLowerBound = provenColorLowerBound;
  result.minimumProven = minimumProven;
  result.exactSearchLimitReached = exactSearchLimitReached;
  result.exactSearchStateCount = exactSearchStateCount;
  return result;
}

/// Diagnoses an inconclusive bounded search without reporting hardware or L1
/// infeasibility that the search did not prove.
static void
setExactSearchLimitFailure(ModuleOp moduleOp, unsigned firstFitCount,
                           std::uint64_t exactSearchStateCount,
                           std::uint64_t exactColoringSearchStateLimit,
                           StringRef constrainedResource,
                           DFBAnalysisFailure &analysisFailure) {
  std::string message;
  llvm::raw_string_ostream messageStream(message);
  messageStream << "deterministic first-fit uses " << firstFitCount
                << " physical DFB indices; exact allocation search explored "
                << exactSearchStateCount << " states and reached the "
                << exactColoringSearchStateLimit
                << "-state limit without proving whether the allocation fits "
                << constrainedResource
                << "; increase `exact-coloring-search-limit`";
  analysisFailure.set(moduleOp, messageStream.str());
}

/// Maps provisional user indices to a dense physical index range.
///
/// Frontend indices identify allocation groups before finalization but may
/// contain gaps when an uncaptured DFB has no declarations. Compacting the
/// distinct values preserves existing sharing without making logical identity
/// depend on provisional numbering.
static DenseMap<int64_t, int32_t>
computeCompactedUserIndices(ArrayRef<DFBLogicalLifecycle> logicalDFBs) {
  DenseSet<int64_t> provisionalIndices;
  for (const DFBLogicalLifecycle &logicalDFB : logicalDFBs) {
    for (BindCBOp declaration : logicalDFB.declarations) {
      if (!declaration->hasAttr(kCompilerAllocatedAttrName)) {
        provisionalIndices.insert(declaration.getCbIndex().getSExtValue());
      }
    }
  }

  SmallVector<int64_t> sortedIndices(provisionalIndices.begin(),
                                     provisionalIndices.end());
  llvm::sort(sortedIndices);

  DenseMap<int64_t, int32_t> compactedIndices;
  for (auto [physicalIndex, provisionalIndex] :
       llvm::enumerate(sortedIndices)) {
    compactedIndices[provisionalIndex] = static_cast<int32_t>(physicalIndex);
  }
  return compactedIndices;
}

/// Complete assignment candidate before descriptor construction.
struct PhysicalAllocationCandidate {
  SmallVector<DFBPhysicalIndexAssignment> assignments;
  int32_t physicalDFBCount = 0;
  bool minimumProven = false;
  bool exactSearchLimitReached = false;
  std::uint64_t exactSearchStateCount = 0;
};

/// Compacts user indices and assigns compiler-created DFBs after that range.
///
/// User declarations retain sharing already expressed by equal provisional
/// indices. Compiler-created DFBs use the same concurrent lifetime proof as
/// the all-DFB allocator, so the option changes allocation policy without
/// selecting a second lifetime model.
static FailureOr<PhysicalAllocationCandidate> computeDistinctUserAllocation(
    ModuleOp moduleOp, const DFBConcurrentKernelLivenessAnalysis &liveness,
    const DFBPhysicalConflictModel &conflictModel,
    const TargetDFBIndexCapacity &targetCapacity,
    DFBAnalysisFailure &analysisFailure,
    std::uint64_t exactColoringSearchStateLimit,
    ArrayRef<uint64_t> allocationBytesByLogicalIndex,
    std::optional<uint64_t> allocationByteLimit,
    std::optional<uint64_t> minimumSearchTriggerBytes) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  DenseMap<int64_t, int32_t> compactedUserIndices =
      computeCompactedUserIndices(logicalDFBs);
  int32_t firstCompilerIndex =
      static_cast<int32_t>(compactedUserIndices.size());
  int32_t targetMaxDFBIndices = targetCapacity.indexCount;
  SmallVector<unsigned> compilerLogicalIndices;
  for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
    if (indexedLogicalDFB.value().compilerCreated) {
      compilerLogicalIndices.push_back(indexedLogicalDFB.index());
    }
  }

  DenseMap<unsigned, int32_t> userIndexByLogicalIndex;
  DFBAllocationFootprint fixedUserFootprint;
  for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
    unsigned logicalIndex = indexedLogicalDFB.index();
    const DFBLogicalLifecycle &logicalDFB = indexedLogicalDFB.value();
    if (logicalDFB.compilerCreated) {
      continue;
    }
    std::optional<int32_t> logicalPhysicalIndex;
    for (BindCBOp declaration : logicalDFB.declarations) {
      if (declaration->hasAttr(kCompilerAllocatedAttrName)) {
        continue;
      }
      int64_t provisionalIndex = declaration.getCbIndex().getSExtValue();
      auto physicalIndexIt = compactedUserIndices.find(provisionalIndex);
      assert(physicalIndexIt != compactedUserIndices.end() &&
             "every user DFB must have a compacted physical index");
      if (!logicalPhysicalIndex) {
        logicalPhysicalIndex = physicalIndexIt->second;
      } else if (*logicalPhysicalIndex != physicalIndexIt->second) {
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "logical DFB " << logicalDFB.logicalId
                      << " has inconsistent physical indices "
                      << *logicalPhysicalIndex << " and "
                      << physicalIndexIt->second;
        analysisFailure.set(declaration, messageStream.str());
        return failure();
      }
    }
    assert(logicalPhysicalIndex &&
           "a user logical DFB must have a user declaration");
    userIndexByLogicalIndex[logicalIndex] = *logicalPhysicalIndex;
    if (!logicalDFB.tensorBacking) {
      std::string failureReason;
      if (failed(fixedUserFootprint.add(
              moduleOp, *logicalPhysicalIndex,
              cast<CircularBufferType>(logicalDFB.type), failureReason))) {
        analysisFailure.set(logicalDFB.declarations.front(), failureReason);
        return failure();
      }
    }
  }
  FailureOr<uint64_t> fixedUserBytes = fixedUserFootprint.getTotalBytes();
  if (failed(fixedUserBytes)) {
    analysisFailure.set(moduleOp,
                        "DFB allocation size is not representable as uint64_t");
    return failure();
  }
  std::optional<uint64_t> compilerAllocationByteLimit;
  if (allocationByteLimit) {
    compilerAllocationByteLimit = *fixedUserBytes > *allocationByteLimit
                                      ? 0
                                      : *allocationByteLimit - *fixedUserBytes;
  }
  std::optional<uint64_t> compilerMinimumSearchTriggerBytes;
  if (minimumSearchTriggerBytes) {
    compilerMinimumSearchTriggerBytes =
        *fixedUserBytes > *minimumSearchTriggerBytes
            ? 0
            : *minimumSearchTriggerBytes - *fixedUserBytes;
  }

  unsigned availableCompilerIndices =
      firstCompilerIndex >= targetMaxDFBIndices
          ? 0
          : static_cast<unsigned>(targetMaxDFBIndices - firstCompilerIndex);
  FailureOr<ConcurrentAssignmentResult> compilerAssignment =
      computeConcurrentAssignments(
          moduleOp, compilerLogicalIndices, firstCompilerIndex, conflictModel,
          logicalDFBs, availableCompilerIndices, exactColoringSearchStateLimit,
          analysisFailure, allocationBytesByLogicalIndex,
          compilerAllocationByteLimit, compilerMinimumSearchTriggerBytes);
  if (failed(compilerAssignment)) {
    return failure();
  }
  DenseMap<unsigned, int32_t> compilerIndexByLogicalIndex;
  for (auto indexedCompilerIndex : llvm::enumerate(compilerLogicalIndices)) {
    compilerIndexByLogicalIndex[indexedCompilerIndex.value()] =
        compilerAssignment->assignments[indexedCompilerIndex.index()];
  }

  PhysicalAllocationCandidate allocation;
  allocation.minimumProven = compilerAssignment->minimumProven;
  allocation.exactSearchLimitReached =
      compilerAssignment->exactSearchLimitReached;
  allocation.exactSearchStateCount = compilerAssignment->exactSearchStateCount;
  for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
    unsigned logicalIndex = indexedLogicalDFB.index();
    const DFBLogicalLifecycle &logicalDFB = indexedLogicalDFB.value();
    int32_t physicalIndex;
    if (logicalDFB.compilerCreated) {
      auto physicalIndexIt = compilerIndexByLogicalIndex.find(logicalIndex);
      assert(physicalIndexIt != compilerIndexByLogicalIndex.end() &&
             "every compiler-created DFB must have a physical index");
      physicalIndex = physicalIndexIt->second;
    } else {
      physicalIndex = userIndexByLogicalIndex.lookup(logicalIndex);
    }

    allocation.assignments.push_back(
        {logicalDFB.logicalId, physicalIndex, physicalIndex, logicalDFB.type,
         logicalDFB.tensorBacking, logicalDFB.allocationGroup,
         logicalDFB.launchDomain, logicalDFB.declarations,
         logicalDFB.bounded || logicalDFB.conditionallyBounded});
    allocation.physicalDFBCount =
        std::max(allocation.physicalDFBCount, physicalIndex + 1);
  }

  for (unsigned lhsIndex = 0; lhsIndex < logicalDFBs.size(); ++lhsIndex) {
    if (logicalDFBs[lhsIndex].compilerCreated) {
      continue;
    }
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < logicalDFBs.size();
         ++rhsIndex) {
      if (logicalDFBs[rhsIndex].compilerCreated ||
          allocation.assignments[lhsIndex].physicalIndex !=
              allocation.assignments[rhsIndex].physicalIndex ||
          !conflictModel.conflicts(lhsIndex, rhsIndex)) {
        continue;
      }
      int32_t physicalIndex = allocation.assignments[rhsIndex].physicalIndex;
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "provisional physical DFB index " << physicalIndex
                    << " aliases conflicting logical DFBs "
                    << logicalDFBs[lhsIndex].logicalId << " and "
                    << logicalDFBs[rhsIndex].logicalId;
      analysisFailure.set(logicalDFBs[rhsIndex].declarations.front(),
                          messageStream.str());
      return failure();
    }
  }

  int32_t compilerSlotCount = allocation.physicalDFBCount - firstCompilerIndex;
  if (allocation.physicalDFBCount > targetMaxDFBIndices) {
    if (allocation.exactSearchLimitReached) {
      setExactSearchLimitFailure(
          moduleOp, allocation.physicalDFBCount,
          allocation.exactSearchStateCount, exactColoringSearchStateLimit,
          targetCapacity.getDescription(), analysisFailure);
      return failure();
    }
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    if (allocation.minimumProven) {
      messageStream << "need " << allocation.physicalDFBCount;
    } else {
      messageStream << "need at least "
                    << firstCompilerIndex +
                           static_cast<int32_t>(
                               compilerAssignment->provenColorLowerBound);
    }
    messageStream << " unspilled DFB indices, exceeding "
                  << targetCapacity.getDescription() << " ("
                  << compilerSlotCount
                  << " compiler-allocated after proven reuse)";
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }
  return allocation;
}

/// Assigns every logical DFB together so user and compiler-created lifetimes
/// may share physical indices under the same conflict model.
static FailureOr<PhysicalAllocationCandidate>
computeReuseAllocation(ModuleOp moduleOp,
                       const DFBConcurrentKernelLivenessAnalysis &liveness,
                       const DFBPhysicalConflictModel &conflictModel,
                       const TargetDFBIndexCapacity &targetCapacity,
                       DFBAnalysisFailure &analysisFailure,
                       std::uint64_t exactColoringSearchStateLimit,
                       ArrayRef<uint64_t> allocationBytesByLogicalIndex,
                       std::optional<uint64_t> allocationByteLimit,
                       std::optional<uint64_t> minimumSearchTriggerBytes) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  SmallVector<unsigned> logicalIndices =
      llvm::to_vector(llvm::seq<unsigned>(0, logicalDFBs.size()));
  int32_t targetMaxDFBIndices = targetCapacity.indexCount;
  FailureOr<ConcurrentAssignmentResult> assignment =
      computeConcurrentAssignments(
          moduleOp, logicalIndices, /*firstPhysicalIndex=*/0, conflictModel,
          logicalDFBs, targetMaxDFBIndices, exactColoringSearchStateLimit,
          analysisFailure, allocationBytesByLogicalIndex, allocationByteLimit,
          minimumSearchTriggerBytes);
  if (failed(assignment)) {
    return failure();
  }

  PhysicalAllocationCandidate allocation;
  allocation.minimumProven = assignment->minimumProven;
  allocation.exactSearchLimitReached = assignment->exactSearchLimitReached;
  allocation.exactSearchStateCount = assignment->exactSearchStateCount;
  for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
    const DFBLogicalLifecycle &logicalDFB = indexedLogicalDFB.value();
    int32_t physicalIndex = assignment->assignments[indexedLogicalDFB.index()];
    allocation.physicalDFBCount =
        std::max(allocation.physicalDFBCount, physicalIndex + 1);
    allocation.assignments.push_back(
        {logicalDFB.logicalId, physicalIndex, physicalIndex, logicalDFB.type,
         logicalDFB.tensorBacking, logicalDFB.allocationGroup,
         logicalDFB.launchDomain, logicalDFB.declarations,
         logicalDFB.bounded || logicalDFB.conditionallyBounded});
  }

  if (allocation.physicalDFBCount <= targetMaxDFBIndices) {
    return allocation;
  }
  if (allocation.exactSearchLimitReached) {
    setExactSearchLimitFailure(
        moduleOp, allocation.physicalDFBCount, allocation.exactSearchStateCount,
        exactColoringSearchStateLimit, targetCapacity.getDescription(),
        analysisFailure);
    return failure();
  }

  std::string message;
  llvm::raw_string_ostream messageStream(message);
  if (allocation.minimumProven) {
    messageStream << "DFB allocation needs " << allocation.physicalDFBCount;
  } else {
    messageStream << "DFB allocation needs at least "
                  << assignment->provenColorLowerBound;
  }
  messageStream << " unspilled physical indices, exceeding "
                << targetCapacity.getDescription();
  analysisFailure.set(moduleOp, messageStream.str());
  return failure();
}

static void setInvalidDFBPageSizeFailure(CircularBufferType dfbType,
                                         Operation *operation,
                                         DFBAnalysisFailure &analysisFailure) {
  std::string message;
  llvm::raw_string_ostream messageStream(message);
  messageStream << "DFB element type must occupy a positive whole number of "
                   "bytes, got "
                << dfbType.getElementType();
  analysisFailure.set(operation, messageStream.str());
}

/// Returns the L1 bytes required by the unique physical assignments.
static FailureOr<uint64_t> computeAllocationBytes(
    ModuleOp moduleOp, ArrayRef<DFBPhysicalIndexAssignment> assignments,
    ArrayRef<LaunchNodeCoord> launchNodes, std::string &failureReason) {
  SmallVector<DFBStorageFootprint> footprints(
      launchNodes.empty() ? 1 : launchNodes.size());
  for (const DFBPhysicalIndexAssignment &assignment : assignments) {
    if (assignment.tensorBacking) {
      continue;
    }
    for (auto indexedFootprint : llvm::enumerate(footprints)) {
      bool active = launchNodes.empty();
      if (!active) {
        const std::set<LaunchNodeCoord> *possibleNodes =
            assignment.launchDomain.getUpperBoundNodes();
        active = !possibleNodes ||
                 possibleNodes->find(launchNodes[indexedFootprint.index()]) !=
                     possibleNodes->end();
      }
      if (!active) {
        continue;
      }
      if (failed(indexedFootprint.value().add(
              assignment.storageIndex,
              cast<CircularBufferType>(assignment.type), failureReason))) {
        return failure();
      }
    }
  }
  uint64_t peakBytes = 0;
  for (const DFBStorageFootprint &footprint : footprints) {
    FailureOr<uint64_t> nodeBytes = footprint.getL1AllocationBytes(moduleOp);
    if (failed(nodeBytes)) {
      failureReason = "DFB storage allocation size is not representable";
      return failure();
    }
    peakBytes = std::max(peakBytes, *nodeBytes);
  }
  return peakBytes;
}

static LogicalResult assignPhysicalStorageIndices(
    ModuleOp moduleOp, PhysicalAllocationCandidate &allocation,
    const DFBPhysicalConflictModel &storageConflictModel,
    ArrayRef<LaunchNodeCoord> launchNodes, DFBAnalysisFailure &analysisFailure,
    bool reuseUserDFBs) {
  if (!reuseUserDFBs) {
    for (DFBPhysicalIndexAssignment &assignment : allocation.assignments) {
      assignment.storageIndex = assignment.physicalIndex;
    }
    return success();
  }

  SmallVector<SmallVector<unsigned>> logicalIndicesByPhysicalIndex(
      allocation.physicalDFBCount);
  SmallVector<uint64_t> bytesByPhysicalIndex(allocation.physicalDFBCount, 0);
  SmallVector<uint64_t> l1BytesByPhysicalIndex(allocation.physicalDFBCount, 0);
  SmallVector<uint64_t> pageSizeByPhysicalIndex(allocation.physicalDFBCount, 1);
  SmallVector<LaunchNodeDomain> domainByPhysicalIndex(
      allocation.physicalDFBCount);
  llvm::BitVector tensorBacked(allocation.physicalDFBCount);
  for (auto indexedAssignment : llvm::enumerate(allocation.assignments)) {
    const DFBPhysicalIndexAssignment &assignment = indexedAssignment.value();
    logicalIndicesByPhysicalIndex[assignment.physicalIndex].push_back(
        indexedAssignment.index());
    domainByPhysicalIndex[assignment.physicalIndex] =
        domainByPhysicalIndex[assignment.physicalIndex].unionWith(
            assignment.launchDomain);
    if (assignment.tensorBacking) {
      tensorBacked.set(assignment.physicalIndex);
      continue;
    }
    std::string failureReason;
    FailureOr<uint64_t> assignmentBytes = getDFBAllocationSizeBytes(
        cast<CircularBufferType>(assignment.type), failureReason);
    FailureOr<uint64_t> pageSize =
        getDFBPageSizeBytes(cast<CircularBufferType>(assignment.type));
    if (failed(assignmentBytes) || failed(pageSize)) {
      if (failureReason.empty()) {
        failureReason = "DFB page size is not representable";
      }
      analysisFailure.set(assignment.declarations.front(), failureReason);
      return failure();
    }
    bytesByPhysicalIndex[assignment.physicalIndex] = std::max(
        bytesByPhysicalIndex[assignment.physicalIndex], *assignmentBytes);
    pageSizeByPhysicalIndex[assignment.physicalIndex] = *pageSize;
  }
  auto failStorageAllocation = [&](unsigned physicalIndex) {
    analysisFailure.set(
        allocation
            .assignments[logicalIndicesByPhysicalIndex[physicalIndex].front()]
            .declarations.front(),
        "DFB storage allocation size is not representable");
    return failure();
  };
  auto replaceAllocationContribution =
      [&](uint64_t totalBytes, uint64_t previousPayloadBytes,
          uint64_t replacementPayloadBytes) -> FailureOr<uint64_t> {
    FailureOr<uint64_t> previousAllocationBytes =
        getL1AllocationSizeBytes(moduleOp, previousPayloadBytes);
    FailureOr<uint64_t> replacementAllocationBytes =
        getL1AllocationSizeBytes(moduleOp, replacementPayloadBytes);
    if (failed(previousAllocationBytes) || failed(replacementAllocationBytes) ||
        totalBytes < *previousAllocationBytes) {
      return failure();
    }
    std::optional<uint64_t> updatedBytes = llvm::checkedAddUnsigned(
        totalBytes - *previousAllocationBytes, *replacementAllocationBytes);
    if (!updatedBytes) {
      return failure();
    }
    return *updatedBytes;
  };
  for (int32_t physicalIndex = 0; physicalIndex < allocation.physicalDFBCount;
       ++physicalIndex) {
    FailureOr<uint64_t> allocationBytes =
        getL1AllocationSizeBytes(moduleOp, bytesByPhysicalIndex[physicalIndex]);
    if (failed(allocationBytes)) {
      return failStorageAllocation(physicalIndex);
    }
    l1BytesByPhysicalIndex[physicalIndex] = *allocationBytes;
  }

  InterferenceGraph storageInterference(allocation.physicalDFBCount);
  for (int32_t lhsPhysicalIndex = 0;
       lhsPhysicalIndex < allocation.physicalDFBCount; ++lhsPhysicalIndex) {
    for (int32_t rhsPhysicalIndex = lhsPhysicalIndex + 1;
         rhsPhysicalIndex < allocation.physicalDFBCount; ++rhsPhysicalIndex) {
      bool conflicts = tensorBacked.test(lhsPhysicalIndex) ||
                       tensorBacked.test(rhsPhysicalIndex);
      for (unsigned lhsLogicalIndex :
           logicalIndicesByPhysicalIndex[lhsPhysicalIndex]) {
        for (unsigned rhsLogicalIndex :
             logicalIndicesByPhysicalIndex[rhsPhysicalIndex]) {
          conflicts |=
              storageConflictModel.conflicts(lhsLogicalIndex, rhsLogicalIndex);
        }
      }
      if (conflicts) {
        storageInterference.addInterference(lhsPhysicalIndex, rhsPhysicalIndex);
      }
    }
  }

  SmallVector<unsigned> physicalIndices;
  llvm::append_range(physicalIndices,
                     llvm::seq<unsigned>(0, allocation.physicalDFBCount));
  llvm::sort(physicalIndices, [&](unsigned lhs, unsigned rhs) {
    if (l1BytesByPhysicalIndex[lhs] != l1BytesByPhysicalIndex[rhs]) {
      return l1BytesByPhysicalIndex[lhs] > l1BytesByPhysicalIndex[rhs];
    }
    if (bytesByPhysicalIndex[lhs] != bytesByPhysicalIndex[rhs]) {
      return bytesByPhysicalIndex[lhs] > bytesByPhysicalIndex[rhs];
    }
    return lhs < rhs;
  });

  struct StorageSlot {
    DFBStorageLayout layout;
    SmallVector<DFBStorageLayout> layoutsByNode;
    SmallVector<unsigned> physicalIndices;
  };
  SmallVector<StorageSlot> slots;
  SmallVector<uint64_t> totalBytesByNode(launchNodes.size(), 0);
  uint64_t totalGlobalBytes = 0;
  SmallVector<int32_t> storageIndexByPhysicalIndex(allocation.physicalDFBCount,
                                                   -1);
  for (unsigned physicalIndex : physicalIndices) {
    std::string standaloneFailureReason;
    FailureOr<DFBStorageLayout> standaloneLayout = mergeDFBStorageLayout(
        {}, bytesByPhysicalIndex[physicalIndex],
        pageSizeByPhysicalIndex[physicalIndex], standaloneFailureReason);
    if (failed(standaloneLayout)) {
      analysisFailure.set(
          allocation
              .assignments[logicalIndicesByPhysicalIndex[physicalIndex].front()]
              .declarations.front(),
          standaloneFailureReason);
      return failure();
    }
    struct StoragePlacement {
      unsigned slotIndex = 0;
      uint64_t peakNodeBytes = 0;
      uint64_t globalBytes = 0;
      bool createsSlot = false;
      DFBStorageLayout layout;
      SmallVector<DFBStorageLayout> layoutsByNode;
      SmallVector<uint64_t> totalBytesByNode;
    };
    std::optional<StoragePlacement> selectedPlacement;
    for (unsigned slotIndex = 0; slotIndex <= slots.size(); ++slotIndex) {
      bool createsSlot = slotIndex == slots.size();
      const StorageSlot *slot = createsSlot ? nullptr : &slots[slotIndex];
      bool conflicts =
          llvm::any_of(slot ? slot->physicalIndices : ArrayRef<unsigned>{},
                       [&](unsigned existingPhysicalIndex) {
                         return storageInterference.interferes(
                             physicalIndex, existingPhysicalIndex);
                       });
      if (conflicts) {
        continue;
      }
      std::string failureReason;
      FailureOr<DFBStorageLayout> mergedLayout = mergeDFBStorageLayout(
          slot ? slot->layout : DFBStorageLayout{},
          bytesByPhysicalIndex[physicalIndex],
          pageSizeByPhysicalIndex[physicalIndex], failureReason);
      if (failed(mergedLayout)) {
        continue;
      }
      SmallVector<DFBStorageLayout> candidateLayoutsByNode =
          slot ? slot->layoutsByNode
               : SmallVector<DFBStorageLayout>(launchNodes.size());
      SmallVector<uint64_t> candidateTotalBytesByNode = totalBytesByNode;
      uint64_t peakNodeBytes = 0;
      const std::set<LaunchNodeCoord> *possibleNodes =
          domainByPhysicalIndex[physicalIndex].getUpperBoundNodes();
      for (auto indexedNode : llvm::enumerate(launchNodes)) {
        DFBStorageLayout &nodeLayout =
            candidateLayoutsByNode[indexedNode.index()];
        uint64_t previousCapacity = nodeLayout.capacityBytes;
        if (!possibleNodes ||
            possibleNodes->find(indexedNode.value()) != possibleNodes->end()) {
          FailureOr<DFBStorageLayout> mergedNodeLayout = mergeDFBStorageLayout(
              nodeLayout, bytesByPhysicalIndex[physicalIndex],
              pageSizeByPhysicalIndex[physicalIndex], failureReason);
          if (failed(mergedNodeLayout)) {
            candidateLayoutsByNode.clear();
            break;
          }
          nodeLayout = *mergedNodeLayout;
        }
        FailureOr<uint64_t> candidateNodeBytes = replaceAllocationContribution(
            totalBytesByNode[indexedNode.index()], previousCapacity,
            nodeLayout.capacityBytes);
        if (failed(candidateNodeBytes)) {
          return failStorageAllocation(physicalIndex);
        }
        peakNodeBytes = std::max(peakNodeBytes, *candidateNodeBytes);
        candidateTotalBytesByNode[indexedNode.index()] = *candidateNodeBytes;
      }
      if (candidateLayoutsByNode.empty() && !launchNodes.empty()) {
        continue;
      }
      FailureOr<uint64_t> candidateGlobalBytes = replaceAllocationContribution(
          totalGlobalBytes, slot ? slot->layout.capacityBytes : 0,
          mergedLayout->capacityBytes);
      if (failed(candidateGlobalBytes)) {
        return failStorageAllocation(physicalIndex);
      }
      if (launchNodes.empty()) {
        peakNodeBytes = *candidateGlobalBytes;
      }
      StoragePlacement placement{slotIndex,
                                 peakNodeBytes,
                                 *candidateGlobalBytes,
                                 createsSlot,
                                 *mergedLayout,
                                 std::move(candidateLayoutsByNode),
                                 std::move(candidateTotalBytesByNode)};
      auto isBetterPlacement = [](const StoragePlacement &candidate,
                                  const StoragePlacement &selected) {
        if (candidate.peakNodeBytes != selected.peakNodeBytes) {
          return candidate.peakNodeBytes < selected.peakNodeBytes;
        }
        if (candidate.globalBytes != selected.globalBytes) {
          return candidate.globalBytes < selected.globalBytes;
        }
        if (candidate.createsSlot != selected.createsSlot) {
          return !candidate.createsSlot;
        }
        return candidate.slotIndex < selected.slotIndex;
      };
      if (!selectedPlacement ||
          isBetterPlacement(placement, *selectedPlacement)) {
        selectedPlacement = std::move(placement);
      }
    }
    if (!selectedPlacement) {
      analysisFailure.set(
          allocation
              .assignments[logicalIndicesByPhysicalIndex[physicalIndex].front()]
              .declarations.front(),
          "DFB storage allocation size is not representable");
      return failure();
    }
    if (selectedPlacement->createsSlot) {
      slots.push_back({});
    }
    StorageSlot &slot = slots[selectedPlacement->slotIndex];
    slot.layout = selectedPlacement->layout;
    slot.layoutsByNode = std::move(selectedPlacement->layoutsByNode);
    totalBytesByNode = std::move(selectedPlacement->totalBytesByNode);
    slot.physicalIndices.push_back(physicalIndex);
    storageIndexByPhysicalIndex[physicalIndex] = selectedPlacement->slotIndex;
    totalGlobalBytes = selectedPlacement->globalBytes;
  }

  for (DFBPhysicalIndexAssignment &assignment : allocation.assignments) {
    assignment.storageIndex =
        storageIndexByPhysicalIndex[assignment.physicalIndex];
  }
  return success();
}

/// Selects an assignment that fits both the target index count and the L1
/// allocation limit. A conservative PipeNet reservation may trigger a stricter
/// search, but only the authoritative DFB-plus-fixed-state budget can reject an
/// assignment. Conversion validates the selected assignment against exact
/// PipeNet resources.
static FailureOr<PhysicalAllocationCandidate> computeAllocationWithinL1(
    ModuleOp moduleOp, std::uint64_t exactColoringSearchStateLimit,
    std::optional<uint64_t> l1BudgetOverride,
    DFBAnalysisFailure &analysisFailure,
    const DFBPhysicalConflictModel &storageConflictModel,
    ArrayRef<LaunchNodeCoord> launchNodes, bool reuseUserDFBs,
    llvm::function_ref<FailureOr<PhysicalAllocationCandidate>(
        std::optional<uint64_t>, std::optional<uint64_t>)>
        computeAllocation) {
  std::string allocationSizeFailureReason;
  FailureOr<uint64_t> resetStateBytes =
      getSynchronizedDFBResetStateAllocationBytes(moduleOp);
  if (failed(resetStateBytes)) {
    analysisFailure.set(moduleOp,
                        "failed to compute synchronized-reset scratch size");
    return failure();
  }
  FailureOr<uint64_t> reconfigurationStateBytes =
      getDFBReconfigurationStateAllocationBytes(moduleOp);
  if (failed(reconfigurationStateBytes)) {
    analysisFailure.set(moduleOp,
                        "failed to compute DFB reconfiguration state size");
    return failure();
  }
  std::optional<uint64_t> fixedStateBytes =
      llvm::checkedAddUnsigned(*resetStateBytes, *reconfigurationStateBytes);
  if (!fixedStateBytes) {
    analysisFailure.set(moduleOp,
                        "combined DFB fixed-state size is not representable");
    return failure();
  }
  uint64_t l1BudgetBytes = getUsableDFBL1Bytes(moduleOp, l1BudgetOverride);
  if (*fixedStateBytes > l1BudgetBytes) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "DFB fixed state requires " << *fixedStateBytes
                  << " L1 bytes but the budget is " << l1BudgetBytes
                  << " (reset scratch=" << *resetStateBytes
                  << ", reconfiguration state=" << *reconfigurationStateBytes
                  << ")";
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }
  uint64_t dfbBudgetBytes = l1BudgetBytes - *fixedStateBytes;
  std::optional<uint64_t> minimumSearchTriggerBytes;
  if (auto reservation = moduleOp->getAttrOfType<IntegerAttr>(
          kPipeConservativeL1BytesAttrName)) {
    if (reservation.getValue().isNegative()) {
      analysisFailure.set(moduleOp,
                          "conservative PipeNet L1 reservation is negative");
      return failure();
    }
    uint64_t reservationBytes = reservation.getValue().getZExtValue();
    if (reservationBytes != 0) {
      std::optional<uint64_t> fixedBytes =
          llvm::checkedAddUnsigned(*fixedStateBytes, reservationBytes);
      minimumSearchTriggerBytes =
          !fixedBytes || *fixedBytes > l1BudgetBytes
              ? std::optional<uint64_t>(0)
              : std::optional<uint64_t>(l1BudgetBytes - *fixedBytes);
    }
  }
  std::optional<uint64_t> initialAllocationByteLimit =
      reuseUserDFBs ? std::nullopt : std::optional<uint64_t>(dfbBudgetBytes);
  FailureOr<PhysicalAllocationCandidate> allocation =
      computeAllocation(initialAllocationByteLimit, minimumSearchTriggerBytes);
  if (failed(allocation)) {
    return failure();
  }
  if (failed(assignPhysicalStorageIndices(moduleOp, *allocation,
                                          storageConflictModel, launchNodes,
                                          analysisFailure, reuseUserDFBs))) {
    return failure();
  }
  FailureOr<uint64_t> allocationBytes =
      computeAllocationBytes(moduleOp, allocation->assignments, launchNodes,
                             allocationSizeFailureReason);
  if (failed(allocationBytes)) {
    analysisFailure.set(moduleOp, allocationSizeFailureReason);
    return failure();
  }
  if (*allocationBytes > dfbBudgetBytes && reuseUserDFBs) {
    allocation = computeAllocation(dfbBudgetBytes, std::optional<uint64_t>(0));
    if (failed(allocation)) {
      return failure();
    }
    if (failed(assignPhysicalStorageIndices(moduleOp, *allocation,
                                            storageConflictModel, launchNodes,
                                            analysisFailure, reuseUserDFBs))) {
      return failure();
    }
    allocationBytes =
        computeAllocationBytes(moduleOp, allocation->assignments, launchNodes,
                               allocationSizeFailureReason);
    if (failed(allocationBytes)) {
      analysisFailure.set(moduleOp, allocationSizeFailureReason);
      return failure();
    }
  }
  if (allocation->exactSearchLimitReached &&
      *allocationBytes > dfbBudgetBytes) {
    setExactSearchLimitFailure(moduleOp, allocation->physicalDFBCount,
                               allocation->exactSearchStateCount,
                               exactColoringSearchStateLimit,
                               "the target L1 budget", analysisFailure);
    return failure();
  }
  if (*allocationBytes > dfbBudgetBytes) {
    std::optional<uint64_t> combinedBytes =
        llvm::checkedAddUnsigned(*allocationBytes, *fixedStateBytes);
    if (!combinedBytes) {
      analysisFailure.set(moduleOp,
                          "combined DFB and fixed-state allocation is not "
                          "representable");
      return failure();
    }
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "selected DFB and fixed-state allocation uses "
                  << *combinedBytes << " L1 bytes, exceeding the "
                  << l1BudgetBytes << "-byte budget (DFB=" << *allocationBytes
                  << ", reset scratch=" << *resetStateBytes
                  << ", reconfiguration state=" << *reconfigurationStateBytes
                  << ")";
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }
  return allocation;
}

/// Builds the dense runtime descriptor table without modifying IR.
static FailureOr<DFBPhysicalAllocationDescriptorList>
buildDescriptors(ArrayRef<DFBPhysicalIndexAssignment> assignments,
                 const DFBConcurrentKernelLivenessAnalysis &liveness,
                 DFBAnalysisFailure &analysisFailure) {
  DenseMap<int64_t, unsigned> reconfigurationOrder;
  for (auto [position, ordinal] :
       llvm::enumerate(liveness.getReconfigurationBoundaryOrdinals())) {
    reconfigurationOrder[ordinal] = position;
  }
  DenseMap<int64_t, const DFBLogicalLifecycle *> lifecycleByLogicalId;
  for (const DFBLogicalLifecycle &logicalDFB :
       liveness.getLogicalDFBLifecycles()) {
    lifecycleByLogicalId.try_emplace(logicalDFB.logicalId, &logicalDFB);
  }
  llvm::DenseMap<int32_t, const DFBPhysicalIndexAssignment *> uniqueByIndex;
  llvm::DenseMap<int32_t, LaunchNodeDomain> allocationDomainByIndex;
  llvm::DenseMap<int32_t, SmallVector<const DFBPhysicalIndexAssignment *, 0>>
      assignmentsByIndex;
  auto getRuntimeAllocationDomain = [&](const LaunchNodeDomain &domain) {
    return liveness.hasExactLaunchGrid() ? domain : LaunchNodeDomain::unknown();
  };
  for (const DFBPhysicalIndexAssignment &assignment : assignments) {
    assignmentsByIndex[assignment.physicalIndex].push_back(&assignment);
    LaunchNodeDomain &allocationDomain =
        allocationDomainByIndex[assignment.physicalIndex];
    allocationDomain = allocationDomain.unionWith(
        getRuntimeAllocationDomain(assignment.launchDomain));
    auto [existingIt, inserted] =
        uniqueByIndex.try_emplace(assignment.physicalIndex, &assignment);
    if (inserted || existingIt->second->type == assignment.type) {
      continue;
    }
    const DFBPhysicalIndexAssignment *existing = existingIt->second;
    auto existingType = cast<CircularBufferType>(existing->type);
    auto assignmentType = cast<CircularBufferType>(assignment.type);
    const DFBLogicalLifecycle *existingLifecycle =
        lifecycleByLogicalId.lookup(existing->logicalId);
    const DFBLogicalLifecycle *assignmentLifecycle =
        lifecycleByLogicalId.lookup(assignment.logicalId);
    assert(existingLifecycle && assignmentLifecycle &&
           "every assignment must have a logical lifecycle");
    bool usesCapacityEnvelope =
        canUseCapacityEnvelope(*existingLifecycle, *assignmentLifecycle);
    bool reconfiguresDescriptor = canReconfigureDescriptorAcrossEpochs(
        *existingLifecycle, *assignmentLifecycle);
    if (!usesCapacityEnvelope && !reconfiguresDescriptor) {
      BindCBOp declaration = assignment.declarations.front();
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "physical DFB index " << assignment.physicalIndex
                    << " has inconsistent CircularBufferType values "
                    << existingIt->second->type << " and " << assignment.type;
      analysisFailure.set(declaration, messageStream.str());
      return failure();
    }
    if (!usesCapacityEnvelope) {
      continue;
    }
    std::string failureReason;
    FailureOr<uint64_t> existingBytes =
        getDFBAllocationSizeBytes(existingType, failureReason);
    FailureOr<uint64_t> assignmentBytes =
        getDFBAllocationSizeBytes(assignmentType, failureReason);
    if (failed(existingBytes) || failed(assignmentBytes)) {
      analysisFailure.set(assignment.declarations.front(), failureReason);
      return failure();
    }
    if (*assignmentBytes > *existingBytes) {
      existingIt->second = &assignment;
    }
  }

  SmallVector<std::pair<int32_t, const DFBPhysicalIndexAssignment *>> sorted(
      uniqueByIndex.begin(), uniqueByIndex.end());
  llvm::sort(sorted, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  DFBPhysicalAllocationDescriptorList descriptors;
  descriptors.reserve(sorted.size());
  for (auto [expectedIndex, indexedAssignment] : llvm::enumerate(sorted)) {
    int32_t physicalIndex = indexedAssignment.first;
    const DFBPhysicalIndexAssignment *assignment = indexedAssignment.second;
    if (physicalIndex != static_cast<int32_t>(expectedIndex)) {
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "physical DFB allocation plan is not dense: expected "
                    << expectedIndex << " but found " << physicalIndex;
      analysisFailure.set(assignment->declarations.front(),
                          messageStream.str());
      return failure();
    }
    DFBPhysicalAllocationDescriptor descriptor;
    descriptor.physicalIndex = physicalIndex;
    descriptor.storageIndex = assignment->storageIndex;
    descriptor.allocationDomain = allocationDomainByIndex.lookup(physicalIndex);
    SmallVector<const DFBPhysicalIndexAssignment *>
        configurationRepresentatives;
    auto addConfiguration =
        [&](const DFBPhysicalIndexAssignment &candidate,
            std::optional<int64_t> entryReconfigurationOrdinal,
            LaunchNodeDomain activeDomain) -> LogicalResult {
      activeDomain = getRuntimeAllocationDomain(activeDomain);
      auto dfbType = cast<CircularBufferType>(candidate.type);
      FailureOr<uint64_t> pagesPerBlock = getDFBPagesPerBlock(dfbType);
      FailureOr<uint64_t> pageSizeBytes = getDFBPageSizeBytes(dfbType);
      if (failed(pageSizeBytes)) {
        setInvalidDFBPageSizeFailure(dfbType, candidate.declarations.front(),
                                     analysisFailure);
        return failure();
      }
      if (failed(pagesPerBlock) ||
          *pagesPerBlock > std::numeric_limits<int32_t>::max() ||
          dfbType.getBlockCount() > std::numeric_limits<int32_t>::max()) {
        analysisFailure.set(candidate.declarations.front(),
                            "DFB dimensions do not fit runtime metadata");
        return failure();
      }
      if (*pageSizeBytes > std::numeric_limits<int32_t>::max()) {
        analysisFailure.set(candidate.declarations.front(),
                            "DFB page size does not fit runtime metadata");
        return failure();
      }

      int32_t numTiles = static_cast<int32_t>(*pagesPerBlock);
      int32_t pageSize = static_cast<int32_t>(*pageSizeBytes);
      int32_t blockCount = static_cast<int32_t>(dfbType.getBlockCount());
      auto configurationIt = llvm::find_if(
          descriptor.epochConfigurations,
          [&](const DFBConfigurationEpochDescriptor &configuration) {
            return configuration.entryReconfigurationOrdinal ==
                   entryReconfigurationOrdinal;
          });
      if (configurationIt == descriptor.epochConfigurations.end()) {
        descriptor.epochConfigurations.push_back({entryReconfigurationOrdinal,
                                                  numTiles,
                                                  dfbType.getElementType(),
                                                  pageSize,
                                                  blockCount,
                                                  {}});
        configurationRepresentatives.push_back(&candidate);
        configurationIt = std::prev(descriptor.epochConfigurations.end());
      } else if (configurationIt->numTiles != numTiles ||
                 configurationIt->elementType != dfbType.getElementType() ||
                 configurationIt->pageSize != pageSize ||
                 configurationIt->blockCount != blockCount) {
        unsigned configurationIndex = std::distance(
            descriptor.epochConfigurations.begin(), configurationIt);
        const DFBPhysicalIndexAssignment *representative =
            configurationRepresentatives[configurationIndex];
        const DFBLogicalLifecycle *representativeLifecycle =
            lifecycleByLogicalId.lookup(representative->logicalId);
        const DFBLogicalLifecycle *candidateLifecycle =
            lifecycleByLogicalId.lookup(candidate.logicalId);
        assert(representativeLifecycle && candidateLifecycle &&
               "every assignment must have a logical lifecycle");
        if (!canUseCapacityEnvelope(*representativeLifecycle,
                                    *candidateLifecycle)) {
          analysisFailure.set(
              candidate.declarations.front(),
              "one physical DFB has inconsistent configurations in one epoch");
          return failure();
        }
        std::string failureReason;
        FailureOr<uint64_t> representativeBytes = getDFBAllocationSizeBytes(
            cast<CircularBufferType>(representative->type), failureReason);
        FailureOr<uint64_t> candidateBytes =
            getDFBAllocationSizeBytes(dfbType, failureReason);
        if (failed(representativeBytes) || failed(candidateBytes)) {
          analysisFailure.set(candidate.declarations.front(), failureReason);
          return failure();
        }
        if (*candidateBytes > *representativeBytes) {
          configurationIt->numTiles = numTiles;
          configurationIt->elementType = dfbType.getElementType();
          configurationIt->pageSize = pageSize;
          configurationIt->blockCount = blockCount;
          configurationRepresentatives[configurationIndex] = &candidate;
        }
      }

      if (!activeDomain.known || activeDomain.nodes.empty()) {
        if (candidate.tensorBacking) {
          analysisFailure.set(
              candidate.declarations.front(),
              "tensor-backed physical DFB requires an exact non-empty "
              "launch-node domain");
          return failure();
        }
        return success();
      }
      auto segmentIt = llvm::find_if(
          configurationIt->storageSegments,
          [&](const DFBPhysicalStorageSegment &segment) {
            return segment.tensorBacking == candidate.tensorBacking;
          });
      if (segmentIt == configurationIt->storageSegments.end()) {
        configurationIt->storageSegments.push_back(
            {LaunchNodeDomain{}, candidate.tensorBacking});
        segmentIt = std::prev(configurationIt->storageSegments.end());
      }
      segmentIt->launchDomain = segmentIt->launchDomain.unionWith(activeDomain);
      return success();
    };

    auto assignmentsIt = assignmentsByIndex.find(physicalIndex);
    assert(assignmentsIt != assignmentsByIndex.end() &&
           "every physical index must have an assignment");
    for (const DFBPhysicalIndexAssignment *indexedCandidate :
         assignmentsIt->second) {
      const DFBPhysicalIndexAssignment &candidate = *indexedCandidate;
      const DFBLogicalLifecycle *lifecycle =
          lifecycleByLogicalId.lookup(candidate.logicalId);
      assert(lifecycle && "every assignment must have a logical lifecycle");
      bool addedConfigurationEpoch = false;
      auto addLifetimeEpochs = [&](const DFBPerNodeLifetime &lifetime) {
        for (const DFBLifecycleEpoch &epoch : lifetime.epochs) {
          LaunchNodeDomain nodeDomain;
          nodeDomain.nodes.insert(lifetime.node);
          if (failed(addConfiguration(
                  candidate, epoch.entryReconfigurationOrdinal, nodeDomain))) {
            return failure();
          }
          addedConfigurationEpoch = true;
        }
        return success();
      };
      for (const DFBPerNodeLifetime &lifetime : lifecycle->nodeLifetimes) {
        if (failed(addLifetimeEpochs(lifetime))) {
          return failure();
        }
      }
      for (const DFBPerNodeLifetime &lifetime :
           lifecycle->possibleNodeLifetimes) {
        if (!lifetime.mayBeActive || failed(addLifetimeEpochs(lifetime))) {
          if (lifetime.mayBeActive) {
            return failure();
          }
        }
      }
      if (!addedConfigurationEpoch &&
          failed(addConfiguration(candidate, std::nullopt,
                                  candidate.launchDomain))) {
        return failure();
      }
    }

    llvm::sort(
        descriptor.epochConfigurations,
        [&](const DFBConfigurationEpochDescriptor &lhs,
            const DFBConfigurationEpochDescriptor &rhs) {
          if (!lhs.entryReconfigurationOrdinal) {
            return rhs.entryReconfigurationOrdinal.has_value();
          }
          if (!rhs.entryReconfigurationOrdinal) {
            return false;
          }
          return reconfigurationOrder.lookup(*lhs.entryReconfigurationOrdinal) <
                 reconfigurationOrder.lookup(*rhs.entryReconfigurationOrdinal);
        });
    assert(!descriptor.epochConfigurations.empty() &&
           "every physical DFB must have one configuration");
    auto compareStorageSegments = [](const DFBPhysicalStorageSegment &lhs,
                                     const DFBPhysicalStorageSegment &rhs) {
      return *lhs.launchDomain.nodes.begin() < *rhs.launchDomain.nodes.begin();
    };
    for (DFBConfigurationEpochDescriptor &configuration :
         descriptor.epochConfigurations) {
      llvm::sort(configuration.storageSegments, compareStorageSegments);
    }
    const DFBConfigurationEpochDescriptor &initialConfiguration =
        descriptor.epochConfigurations.front();
    descriptor.numTiles = initialConfiguration.numTiles;
    descriptor.elementType = initialConfiguration.elementType;
    descriptor.pageSize = initialConfiguration.pageSize;
    descriptor.blockCount = initialConfiguration.blockCount;
    bool hasTensorBacking =
        llvm::any_of(initialConfiguration.storageSegments,
                     [](const DFBPhysicalStorageSegment &segment) {
                       return static_cast<bool>(segment.tensorBacking);
                     });
    if (hasTensorBacking) {
      descriptor.storageSegments = initialConfiguration.storageSegments;
      LaunchNodeDomain initiallyCovered;
      for (const DFBPhysicalStorageSegment &segment :
           initialConfiguration.storageSegments) {
        initiallyCovered = initiallyCovered.unionWith(segment.launchDomain);
      }
      LaunchNodeDomain eventuallyUsed;
      for (const DFBConfigurationEpochDescriptor &configuration :
           descriptor.epochConfigurations) {
        if (configuration.storageSegments.empty()) {
          eventuallyUsed.nodes.insert(liveness.getLaunchNodes().begin(),
                                      liveness.getLaunchNodes().end());
          continue;
        }
        for (const DFBPhysicalStorageSegment &segment :
             configuration.storageSegments) {
          eventuallyUsed = eventuallyUsed.unionWith(segment.launchDomain);
        }
      }
      // Static descriptors must define the index on later-active cores.
      // Scratch placeholders avoid installing future tensor aliases early.
      LaunchNodeDomain placeholderDomain =
          eventuallyUsed.subtract(initiallyCovered);
      if (!placeholderDomain.nodes.empty()) {
        auto placeholderIt =
            llvm::find_if(descriptor.storageSegments,
                          [](const DFBPhysicalStorageSegment &segment) {
                            return !segment.tensorBacking;
                          });
        if (placeholderIt == descriptor.storageSegments.end()) {
          descriptor.storageSegments.push_back(
              {std::move(placeholderDomain), {}});
        } else {
          placeholderIt->launchDomain =
              placeholderIt->launchDomain.unionWith(placeholderDomain);
        }
      }
      llvm::sort(descriptor.storageSegments, compareStorageSegments);
    }
    descriptors.push_back(std::move(descriptor));
  }
  return descriptors;
}

/// Rejects storage ranges whose physical aliasing is not represented by one
/// shared physical DFB index and an exact backing identity.
static LogicalResult
validateTensorBackingRanges(ArrayRef<DFBPhysicalIndexAssignment> assignments,
                            DFBAnalysisFailure &analysisFailure) {
  for (unsigned lhsIndex = 0; lhsIndex < assignments.size(); ++lhsIndex) {
    const DFBPhysicalIndexAssignment &lhs = assignments[lhsIndex];
    if (!lhs.tensorBacking) {
      continue;
    }
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < assignments.size();
         ++rhsIndex) {
      const DFBPhysicalIndexAssignment &rhs = assignments[rhsIndex];
      if (!rhs.tensorBacking ||
          lhs.tensorBacking.getTensorIndex() !=
              rhs.tensorBacking.getTensorIndex() ||
          !launchNodeDomainsOverlap(lhs.launchDomain, rhs.launchDomain)) {
        continue;
      }

      int64_t lhsStart = lhs.tensorBacking.getByteOffset();
      int64_t lhsEnd = lhsStart + lhs.tensorBacking.getByteSize();
      int64_t rhsStart = rhs.tensorBacking.getByteOffset();
      int64_t rhsEnd = rhsStart + rhs.tensorBacking.getByteSize();
      if (lhsStart >= rhsEnd || rhsStart >= lhsEnd) {
        continue;
      }
      if (lhs.physicalIndex == rhs.physicalIndex) {
        continue;
      }
      if (lhs.tensorBacking != rhs.tensorBacking) {
        analysisFailure.set(
            rhs.declarations.front(),
            "tensor-backed DFB byte ranges partially overlap on a shared "
            "launch node");
        return failure();
      }
      analysisFailure.set(
          rhs.declarations.front(),
          "identical tensor-backed DFB ranges require one proven shared "
          "physical index on a shared launch node");
      return failure();
    }
  }
  return success();
}

} // namespace

DFBPhysicalAllocationPlanner::DFBPhysicalAllocationPlanner(
    Operation *operation, bool reuseUserDFBs, bool unsafeAssumeAllocationGroups,
    std::uint64_t exactColoringSearchStateLimit,
    std::optional<uint64_t> l1BudgetOverride,
    ArrayRef<DFBStaticConfigurationConflict> staticConfigurationConflicts,
    AnalysisManager analysisManager) {
  ModuleOp moduleOp = cast<ModuleOp>(operation);
  std::string targetFailureReason;
  FailureOr<TargetDFBIndexCapacity> targetCapacity =
      resolveTargetDFBIndexCapacity(moduleOp, targetFailureReason);
  if (failed(targetCapacity)) {
    errorOperation = moduleOp;
    errorMessage = std::move(targetFailureReason);
    return;
  }
  const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis =
      analysisManager.getAnalysis<DFBLogicalIdentityAnalysis>();
  if (!logicalIdentityAnalysis.succeeded()) {
    errorOperation = logicalIdentityAnalysis.getErrorOperation();
    errorMessage = logicalIdentityAnalysis.getErrorMessage().str();
    return;
  }

  const DFBConcurrentKernelLivenessAnalysis &liveness =
      analysisManager.getAnalysis<DFBConcurrentKernelLivenessAnalysis>();
  if (!liveness.succeeded()) {
    errorOperation = liveness.getErrorOperation();
    errorMessage = liveness.getErrorMessage().str();
    return;
  }
  DFBAnalysisFailure analysisFailure;
  plan.reconfigurationBoundaryOrdinals.assign(
      liveness.getReconfigurationBoundaryOrdinals().begin(),
      liveness.getReconfigurationBoundaryOrdinals().end());
  if (!reuseUserDFBs &&
      hasAllocationGroups(liveness.getLogicalDFBLifecycles())) {
    auto groupedDFB =
        llvm::find_if(liveness.getLogicalDFBLifecycles(),
                      [](const DFBLogicalLifecycle &logicalDFB) {
                        return static_cast<bool>(logicalDFB.allocationGroup);
                      });
    errorOperation = groupedDFB->declarations.front();
    errorMessage = "DFB allocation groups require user DFB reuse to be enabled";
    return;
  }
  if (failed(validateAllocationGroups(
          liveness, staticConfigurationConflicts, unsafeAssumeAllocationGroups,
          plan.assumedAllocationGroups, analysisFailure))) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  plan.conflictModel = DFBPhysicalConflictModelBuilder::build(
      liveness, staticConfigurationConflicts);
  DFBPhysicalConflictModel storageConflictModel =
      DFBPhysicalConflictModelBuilder::buildStorage(liveness);
  LLVM_DEBUG({
    printDFBAllocationDebugReport(llvm::dbgs(), liveness, plan.conflictModel);
    printDFBStorageConflictDebugReport(llvm::dbgs(), storageConflictModel);
  });

  SmallVector<uint64_t> allocationBytesByLogicalIndex;
  allocationBytesByLogicalIndex.reserve(
      liveness.getLogicalDFBLifecycles().size());
  for (const DFBLogicalLifecycle &logicalDFB :
       liveness.getLogicalDFBLifecycles()) {
    if (logicalDFB.tensorBacking) {
      allocationBytesByLogicalIndex.push_back(0);
      continue;
    }
    std::string allocationFailureReason;
    FailureOr<uint64_t> allocationBytes = getDFBL1AllocationSizeBytes(
        moduleOp, cast<CircularBufferType>(logicalDFB.type),
        allocationFailureReason);
    if (failed(allocationBytes)) {
      errorOperation = logicalDFB.declarations.front();
      errorMessage = std::move(allocationFailureReason);
      return;
    }
    allocationBytesByLogicalIndex.push_back(*allocationBytes);
  }

  auto computeAllocation =
      [&](std::optional<uint64_t> allocationByteLimit,
          std::optional<uint64_t> minimumSearchTriggerBytes)
      -> FailureOr<PhysicalAllocationCandidate> {
    if (reuseUserDFBs) {
      return computeReuseAllocation(
          moduleOp, liveness, plan.conflictModel, *targetCapacity,
          analysisFailure, exactColoringSearchStateLimit,
          allocationBytesByLogicalIndex, allocationByteLimit,
          minimumSearchTriggerBytes);
    }
    return computeDistinctUserAllocation(
        moduleOp, liveness, plan.conflictModel, *targetCapacity,
        analysisFailure, exactColoringSearchStateLimit,
        allocationBytesByLogicalIndex, allocationByteLimit,
        minimumSearchTriggerBytes);
  };
  FailureOr<PhysicalAllocationCandidate> allocation = computeAllocationWithinL1(
      moduleOp, exactColoringSearchStateLimit, l1BudgetOverride,
      analysisFailure, storageConflictModel, liveness.getLaunchNodes(),
      reuseUserDFBs, computeAllocation);
  if (failed(allocation)) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  plan.assignments = std::move(allocation->assignments);
  plan.physicalDFBCount = allocation->physicalDFBCount;

  if (failed(validateTensorBackingRanges(plan.assignments, analysisFailure))) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }

  FailureOr<DFBPhysicalAllocationDescriptorList> descriptors =
      buildDescriptors(plan.assignments, liveness, analysisFailure);
  if (failed(descriptors)) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  plan.descriptors = std::move(*descriptors);

  int32_t kernelBaseIndex = plan.physicalDFBCount;
  if (!liveness.getReconfigurationBoundaryOrdinals().empty()) {
    ++kernelBaseIndex;
  }
  if (kernelBaseIndex > 0) {
    for (func::FuncOp kernel : moduleOp.getOps<func::FuncOp>()) {
      if (kernel->hasAttr(kBaseCTAIndexAttrName)) {
        plan.kernelBaseIndices.push_back({kernel, kernelBaseIndex});
      }
    }
  }
}

} // namespace mlir::tt::ttl
