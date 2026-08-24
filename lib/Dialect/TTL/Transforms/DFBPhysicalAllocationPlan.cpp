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

/// Preserves the operation that made a proof conservative, falling back to the
/// declaration only when no more specific evidence exists.
static Operation *getLifetimeEvidence(const DFBPerNodeLifetime *lifetime,
                                      const DFBLogicalLifecycle &logicalDFB) {
  if (lifetime && lifetime->quiescence.evidence) {
    return lifetime->quiescence.evidence;
  }
  return logicalDFB.declarations.front();
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

} // namespace

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
        DFBAllocationGroupAttr lhsGroup = logicalDFBs[lhsIndex].allocationGroup;
        DFBAllocationGroupAttr rhsGroup = logicalDFBs[rhsIndex].allocationGroup;
        bool sameAllocationGroup = lhsGroup && lhsGroup == rhsGroup;
        addPairConflicts(model, liveness, lhsIndex, rhsIndex,
                         /*requireExactDescriptor=*/!sameAllocationGroup,
                         /*requireMatchingTransactions=*/!sameAllocationGroup);
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
    addPairConflicts(model, liveness, lhsIndex, rhsIndex,
                     /*requireExactDescriptor=*/false,
                     /*requireMatchingTransactions=*/false);
    addResetAllocationConflicts(model, liveness,
                                std::make_pair(lhsIndex, rhsIndex));
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
                   bool requireExactDescriptor = true,
                   bool requireMatchingTransactions = true) {
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    const DFBLogicalLifecycle &lhs = logicalDFBs[lhsIndex];
    const DFBLogicalLifecycle &rhs = logicalDFBs[rhsIndex];
    if (requireExactDescriptor && lhs.type != rhs.type) {
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
      if (lhs.tensorBacking != rhs.tensorBacking) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::StorageMismatch, node,
                    lhs.declarations.front(), rhs.declarations.front());
        continue;
      }
      if (!lhsLifetime || !rhsLifetime || !lhsLifetime->quiescence.proven() ||
          !rhsLifetime->quiescence.proven()) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::AccessCompletionNotProven, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
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
        terminalStateCompatible =
            before->terminalStateCanonical || !requireMatchingTransactions ||
            haveCompatibleCursorRuns(*before, *after, physicalTileCount);
        pointerOwnersCompatible =
            before->terminalStateCanonical ||
            (before->terminalWritePointerOwner == after->writePointerOwner &&
             before->terminalReadPointerOwner == after->readPointerOwner);
      } else {
        // Preserve the more specific state diagnosis when lifetimes are also
        // unordered; ordering alone must not obscure a protocol mismatch.
        terminalStateCompatible =
            !requireMatchingTransactions ||
            (haveCompatibleCursorRuns(*lhsLifetime, *rhsLifetime,
                                      physicalTileCount) &&
             haveCompatibleCursorRuns(*rhsLifetime, *lhsLifetime,
                                      physicalTileCount));
        pointerOwnersCompatible =
            lhsLifetime->writePointerOwner == rhsLifetime->writePointerOwner &&
            lhsLifetime->readPointerOwner == rhsLifetime->readPointerOwner;
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

struct AllocationGroupNodeMember {
  unsigned logicalIndex = 0;
  const DFBPerNodeLifetime *lifetime = nullptr;
  bool possibleDomain = false;
};

static bool isAllocationGroupMemberOrderedBefore(
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    const AllocationGroupNodeMember &lhs, const AllocationGroupNodeMember &rhs,
    LaunchNodeCoord node) {
  if (lhs.possibleDomain || rhs.possibleDomain) {
    return lhs.possibleDomain && rhs.possibleDomain &&
           liveness.isConditionallyOrderedBefore(lhs.logicalIndex,
                                                 rhs.logicalIndex, node);
  }
  return liveness.isOrderedBefore(lhs.logicalIndex, rhs.logicalIndex, node);
}

static bool hasAllocationGroupMemberInconsistentOrder(
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    const AllocationGroupNodeMember &lhs, const AllocationGroupNodeMember &rhs,
    LaunchNodeCoord node) {
  if (lhs.possibleDomain || rhs.possibleDomain) {
    return liveness.hasConditionallyInconsistentOrder(lhs.logicalIndex,
                                                      rhs.logicalIndex, node);
  }
  return liveness.hasInconsistentOrder(lhs.logicalIndex, rhs.logicalIndex,
                                       node);
}

struct AllocationGroupCursorState {
  std::uint64_t writePointerOffset = 0;
  std::uint64_t readPointerOffset = 0;
};

static FailureOr<AllocationGroupCursorState> advanceAllocationGroupMemberCursor(
    const DFBPerNodeLifetime &lifetime, std::uint64_t physicalTileCount,
    AllocationGroupCursorState cursorState = {}) {
  auto advanceRuns = [&](ArrayRef<DFBTransactionRun> transactionRuns,
                         std::uint64_t &pointerOffset) {
    FailureOr<std::uint64_t> nextOffset = advanceDFBTransactionCursor(
        transactionRuns, physicalTileCount, pointerOffset);
    if (succeeded(nextOffset)) {
      pointerOffset = *nextOffset;
    }
    return nextOffset;
  };
  if (lifetime.resetEpochs.empty()) {
    if (failed(advanceRuns(lifetime.writeCursorRuns,
                           cursorState.writePointerOffset)) ||
        failed(advanceRuns(lifetime.readCursorRuns,
                           cursorState.readPointerOffset))) {
      return failure();
    }
  } else {
    for (const DFBLifecycleEpoch &epoch : lifetime.resetEpochs) {
      if (failed(advanceRuns(epoch.writeCursorRuns,
                             cursorState.writePointerOffset)) ||
          failed(advanceRuns(epoch.readCursorRuns,
                             cursorState.readPointerOffset))) {
        return failure();
      }
      if (epoch.terminalStateCanonical) {
        cursorState = {};
      }
    }
  }
  if (cursorState.writePointerOffset != cursorState.readPointerOffset) {
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
    SmallVector<AllocationGroupNodeMember> activeMembers;
    for (unsigned logicalIndex : members) {
      const DFBLogicalLifecycle &logicalDFB = logicalDFBs[logicalIndex];
      bool possibleDomain = !logicalDFB.launchDomain.known;
      const DFBPerNodeLifetime *lifetime =
          possibleDomain ? logicalDFB.findPossibleNodeLifetime(node)
                         : logicalDFB.findNodeLifetime(node);
      if (!lifetime || !lifetime->mayBeActive) {
        continue;
      }
      activeMembers.push_back({logicalIndex, lifetime, possibleDomain});
    }

    for (const AllocationGroupNodeMember &member : activeMembers) {
      if (!hasAllocationGroupMemberInconsistentOrder(liveness, member, member,
                                                     node)) {
        continue;
      }
      const DFBLogicalLifecycle &logicalDFB = logicalDFBs[member.logicalIndex];
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB allocation group ";
      printAllocationGroup(messageStream, allocationGroup, members,
                           logicalDFBs);
      messageStream << " has contradictory cursor order involving logical DFB "
                    << logicalDFB.logicalId << " on launch node (" << node.x
                    << ',' << node.y << ')';
      analysisFailure.set(logicalDFB.declarations.front(), messageStream.str());
      return failure();
    }

    bool hasUnprovenOrder = false;
    SmallVector<unsigned> predecessorCounts(activeMembers.size());
    for (auto [lhsPosition, lhs] : llvm::enumerate(activeMembers)) {
      for (unsigned rhsPosition = lhsPosition + 1;
           rhsPosition < activeMembers.size(); ++rhsPosition) {
        const AllocationGroupNodeMember &rhs = activeMembers[rhsPosition];
        bool lhsBeforeRhs =
            isAllocationGroupMemberOrderedBefore(liveness, lhs, rhs, node);
        bool rhsBeforeLhs =
            isAllocationGroupMemberOrderedBefore(liveness, rhs, lhs, node);
        if (lhsBeforeRhs != rhsBeforeLhs) {
          unsigned afterPosition = lhsBeforeRhs ? rhsPosition : lhsPosition;
          ++predecessorCounts[afterPosition];
          continue;
        }
        bool inconsistentOrder =
            hasAllocationGroupMemberInconsistentOrder(liveness, lhs, rhs, node);
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
                      << "logical DFBs "
                      << logicalDFBs[lhs.logicalIndex].logicalId << " and "
                      << logicalDFBs[rhs.logicalIndex].logicalId
                      << " on launch node (" << node.x << ',' << node.y << ')';
        analysisFailure.set(logicalDFBs[rhs.logicalIndex].declarations.front(),
                            messageStream.str());
        return failure();
      }
    }

    if (hasUnprovenOrder) {
      for (const AllocationGroupNodeMember &member : activeMembers) {
        if (succeeded(advanceAllocationGroupMemberCursor(*member.lifetime,
                                                         physicalTileCount))) {
          continue;
        }
        const DFBLogicalLifecycle &logicalDFB =
            logicalDFBs[member.logicalIndex];
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "DFB allocation group ";
        printAllocationGroup(messageStream, allocationGroup, members,
                             logicalDFBs);
        messageStream << " physical envelope of " << physicalTileCount
                      << " tiles makes logical DFB " << logicalDFB.logicalId
                      << " cross the ring boundary on launch node (" << node.x
                      << ',' << node.y << ')';
        analysisFailure.set(logicalDFB.declarations.front(),
                            messageStream.str());
        return failure();
      }
      continue;
    }
    SmallVector<AllocationGroupNodeMember> orderedMembers(activeMembers.size());
    llvm::BitVector occupiedRanks(activeMembers.size());
    for (auto [memberPosition, member] : llvm::enumerate(activeMembers)) {
      unsigned rank = predecessorCounts[memberPosition];
      if (rank < orderedMembers.size() && !occupiedRanks.test(rank)) {
        orderedMembers[rank] = member;
        occupiedRanks.set(rank);
        continue;
      }
      const DFBLogicalLifecycle &logicalDFB = logicalDFBs[member.logicalIndex];
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB allocation group ";
      printAllocationGroup(messageStream, allocationGroup, members,
                           logicalDFBs);
      messageStream << " has inconsistent cursor order for logical DFB "
                    << logicalDFB.logicalId << " on launch node (" << node.x
                    << ',' << node.y << ')';
      analysisFailure.set(logicalDFB.declarations.front(), messageStream.str());
      return failure();
    }
    activeMembers = std::move(orderedMembers);

    AllocationGroupCursorState cursorState;
    for (const AllocationGroupNodeMember &member : activeMembers) {
      const DFBPerNodeLifetime &lifetime = *member.lifetime;
      FailureOr<AllocationGroupCursorState> nextOffset =
          advanceAllocationGroupMemberCursor(lifetime, physicalTileCount,
                                             cursorState);
      if (succeeded(nextOffset)) {
        cursorState = *nextOffset;
        continue;
      }

      const DFBLogicalLifecycle &logicalDFB = logicalDFBs[member.logicalIndex];
      FailureOr<AllocationGroupCursorState> resetState =
          advanceAllocationGroupMemberCursor(lifetime, physicalTileCount);
      if (unsafeAssumeAllocationGroups && succeeded(resetState)) {
        addAllocationGroupAssumption(
            assumptions, DFBAllocationGroupAssumptionReason::EpochReset,
            logicalDFB.logicalId);
        cursorState = *resetState;
        continue;
      }
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB allocation group ";
      printAllocationGroup(messageStream, allocationGroup, members,
                           logicalDFBs);
      messageStream << " physical envelope of " << physicalTileCount
                    << " tiles makes logical DFB " << logicalDFB.logicalId
                    << " cross the ring boundary on launch node (" << node.x
                    << ',' << node.y << ')';
      analysisFailure.set(logicalDFB.declarations.front(), messageStream.str());
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
/// it fits `availableIndices`. Otherwise one exhaustive fixed-limit search
/// decides whether some assignment fits. A minimum physical-index-count search
/// runs only for an L1-budget decision. `firstPhysicalIndex` reserves lower
/// index values without changing which DFB pairs may share.
static FailureOr<ConcurrentAssignmentResult> computeConcurrentAssignments(
    ModuleOp moduleOp, ArrayRef<unsigned> candidateIndices,
    int32_t firstPhysicalIndex, const DFBPhysicalConflictModel &conflictModel,
    ArrayRef<DFBLogicalLifecycle> logicalDFBs, unsigned availableIndices,
    std::uint64_t exactColoringSearchStateLimit,
    DFBAnalysisFailure &analysisFailure, bool requireMinimum = false) {
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
  if (!minimumProven && requireMinimum) {
    ExactInterferenceGraphColoring exactColoring =
        colorInterferenceGraphExactly(interferenceGraph,
                                      exactColoringSearchStateLimit);
    exactSearchStateCount = exactColoring.exploredStateCount;
    if (exactColoring.isOptimal()) {
      selectedColors = std::move(exactColoring.colors);
      colorCount = exactColoring.colorCount;
      minimumProven = true;
      provenColorLowerBound = colorCount;
    } else {
      exactSearchLimitReached = true;
    }
  } else if (colorCount > availableIndices &&
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
    std::uint64_t exactColoringSearchStateLimit, bool requireMinimum = false) {
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

  unsigned availableCompilerIndices =
      firstCompilerIndex >= targetMaxDFBIndices
          ? 0
          : static_cast<unsigned>(targetMaxDFBIndices - firstCompilerIndex);
  FailureOr<ConcurrentAssignmentResult> compilerAssignment =
      computeConcurrentAssignments(
          moduleOp, compilerLogicalIndices, firstCompilerIndex, conflictModel,
          logicalDFBs, availableCompilerIndices, exactColoringSearchStateLimit,
          analysisFailure, requireMinimum);
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
      std::optional<int32_t> logicalPhysicalIndex;
      for (BindCBOp declaration : logicalDFB.declarations) {
        if (declaration->hasAttr(kCompilerAllocatedAttrName)) {
          continue;
        }
        int64_t provisionalIndex = declaration.getCbIndex().getSExtValue();
        auto physicalIndexIt = compactedUserIndices.find(provisionalIndex);
        assert(physicalIndexIt != compactedUserIndices.end() &&
               "every user DFB must have a compacted physical index");
        if (!logicalPhysicalIndex.has_value()) {
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
      assert(logicalPhysicalIndex.has_value() &&
             "a user logical DFB must have a user declaration");
      physicalIndex = *logicalPhysicalIndex;
    }

    allocation.assignments.push_back(
        {logicalDFB.logicalId, physicalIndex, logicalDFB.type,
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
static FailureOr<PhysicalAllocationCandidate> computeReuseAllocation(
    ModuleOp moduleOp, const DFBConcurrentKernelLivenessAnalysis &liveness,
    const DFBPhysicalConflictModel &conflictModel,
    const TargetDFBIndexCapacity &targetCapacity,
    DFBAnalysisFailure &analysisFailure,
    std::uint64_t exactColoringSearchStateLimit, bool requireMinimum) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  SmallVector<unsigned> logicalIndices =
      llvm::to_vector(llvm::seq<unsigned>(0, logicalDFBs.size()));
  int32_t targetMaxDFBIndices = targetCapacity.indexCount;
  FailureOr<ConcurrentAssignmentResult> assignment =
      computeConcurrentAssignments(
          moduleOp, logicalIndices, /*firstPhysicalIndex=*/0, conflictModel,
          logicalDFBs, targetMaxDFBIndices, exactColoringSearchStateLimit,
          analysisFailure, requireMinimum);
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
        {logicalDFB.logicalId, physicalIndex, logicalDFB.type,
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
static FailureOr<uint64_t>
computeAllocationBytes(ModuleOp moduleOp,
                       ArrayRef<DFBPhysicalIndexAssignment> assignments,
                       std::string &failureReason) {
  DFBAllocationFootprint footprint;
  for (const DFBPhysicalIndexAssignment &assignment : assignments) {
    if (assignment.tensorBacking) {
      continue;
    }
    if (failed(footprint.add(moduleOp, assignment.physicalIndex,
                             cast<CircularBufferType>(assignment.type),
                             failureReason))) {
      return failure();
    }
  }
  return footprint.getTotalBytes();
}

/// Recomputes an assignment with the minimum physical-index count when a valid
/// first-fit assignment exceeds either the authoritative DFB-plus-reset budget
/// or the provisional threshold after a conservative PipeNet reservation. The
/// reservation only triggers search; finalization rejects against the
/// authoritative budget, and conversion validates exact PipeNet resources.
static FailureOr<PhysicalAllocationCandidate> computeAllocationWithinL1(
    ModuleOp moduleOp, std::uint64_t exactColoringSearchStateLimit,
    std::optional<uint64_t> l1BudgetOverride,
    DFBAnalysisFailure &analysisFailure,
    llvm::function_ref<FailureOr<PhysicalAllocationCandidate>(bool)>
        computeAllocation) {
  FailureOr<PhysicalAllocationCandidate> allocation =
      computeAllocation(/*requireMinimum=*/false);
  if (failed(allocation)) {
    return failure();
  }

  std::string allocationSizeFailureReason;
  FailureOr<uint64_t> allocationBytes = computeAllocationBytes(
      moduleOp, allocation->assignments, allocationSizeFailureReason);
  if (failed(allocationBytes)) {
    analysisFailure.set(moduleOp, allocationSizeFailureReason);
    return failure();
  }
  FailureOr<uint64_t> resetStateBytes =
      getSynchronizedDFBResetStateAllocationBytes(moduleOp);
  if (failed(resetStateBytes)) {
    analysisFailure.set(moduleOp,
                        "failed to compute synchronized-reset scratch size");
    return failure();
  }
  uint64_t l1BudgetBytes = getUsableDFBL1Bytes(moduleOp, l1BudgetOverride);
  if (*resetStateBytes > l1BudgetBytes) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "synchronized-reset scratch requires " << *resetStateBytes
                  << " L1 bytes but the budget is " << l1BudgetBytes;
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }
  uint64_t dfbBudgetBytes = l1BudgetBytes - *resetStateBytes;
  uint64_t minimumSearchTriggerBytes = dfbBudgetBytes;
  if (auto pipeReservation = moduleOp->getAttrOfType<IntegerAttr>(
          kPipeConservativeL1BytesAttrName)) {
    if (pipeReservation.getValue().isNegative()) {
      analysisFailure.set(moduleOp,
                          "conservative PipeNet L1 reservation is negative");
      return failure();
    }
    uint64_t pipeBytes = pipeReservation.getValue().getZExtValue();
    minimumSearchTriggerBytes = pipeBytes > minimumSearchTriggerBytes
                                    ? 0
                                    : minimumSearchTriggerBytes - pipeBytes;
  }
  if (*allocationBytes > minimumSearchTriggerBytes &&
      !allocation->minimumProven) {
    allocation = computeAllocation(/*requireMinimum=*/true);
    if (failed(allocation)) {
      return failure();
    }
    allocationBytes = computeAllocationBytes(moduleOp, allocation->assignments,
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
        llvm::checkedAddUnsigned(*allocationBytes, *resetStateBytes);
    if (!combinedBytes) {
      analysisFailure.set(
          moduleOp, "combined DFB and reset allocation is not representable");
      return failure();
    }
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "DFB and synchronized-reset allocation requires "
                  << *combinedBytes << " L1 bytes but the budget is "
                  << l1BudgetBytes << " (DFB=" << *allocationBytes
                  << ", reset scratch=" << *resetStateBytes << ")";
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }
  return allocation;
}

/// Builds the dense runtime descriptor table without modifying IR.
static FailureOr<SmallVector<DFBPhysicalAllocationDescriptor>>
buildDescriptors(ArrayRef<DFBPhysicalIndexAssignment> assignments,
                 DFBAnalysisFailure &analysisFailure) {
  llvm::DenseMap<int32_t, const DFBPhysicalIndexAssignment *> uniqueByIndex;
  for (const DFBPhysicalIndexAssignment &assignment : assignments) {
    auto [existingIt, inserted] =
        uniqueByIndex.try_emplace(assignment.physicalIndex, &assignment);
    if (inserted || existingIt->second->type == assignment.type) {
      continue;
    }
    const DFBPhysicalIndexAssignment *existing = existingIt->second;
    auto existingType = cast<CircularBufferType>(existing->type);
    auto assignmentType = cast<CircularBufferType>(assignment.type);
    if (!existing->allocationGroup ||
        existing->allocationGroup != assignment.allocationGroup ||
        existingType.getElementType() != assignmentType.getElementType() ||
        existing->tensorBacking || assignment.tensorBacking) {
      BindCBOp declaration = assignment.declarations.front();
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "physical DFB index " << assignment.physicalIndex
                    << " has inconsistent CircularBufferType values "
                    << existingIt->second->type << " and " << assignment.type;
      analysisFailure.set(declaration, messageStream.str());
      return failure();
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

  SmallVector<DFBPhysicalAllocationDescriptor> descriptors;
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
    auto dfbType = cast<CircularBufferType>(assignment->type);
    FailureOr<uint64_t> pagesPerBlock = getDFBPagesPerBlock(dfbType);
    FailureOr<uint64_t> pageSizeBytes = getDFBPageSizeBytes(dfbType);
    if (failed(pageSizeBytes)) {
      setInvalidDFBPageSizeFailure(dfbType, assignment->declarations.front(),
                                   analysisFailure);
      return failure();
    }
    if (failed(pagesPerBlock) ||
        *pagesPerBlock > std::numeric_limits<int32_t>::max() ||
        dfbType.getBlockCount() > std::numeric_limits<int32_t>::max()) {
      analysisFailure.set(assignment->declarations.front(),
                          "DFB dimensions do not fit runtime metadata");
      return failure();
    }
    if (*pageSizeBytes > std::numeric_limits<int32_t>::max()) {
      analysisFailure.set(assignment->declarations.front(),
                          "DFB page size does not fit runtime metadata");
      return failure();
    }
    DFBPhysicalAllocationDescriptor descriptor{
        physicalIndex,
        static_cast<int32_t>(*pagesPerBlock),
        dfbType.getElementType(),
        static_cast<int32_t>(*pageSizeBytes),
        static_cast<int32_t>(dfbType.getBlockCount()),
        {}};

    bool hasTensorBacking = llvm::any_of(
        assignments, [&](const DFBPhysicalIndexAssignment &candidate) {
          return candidate.physicalIndex == physicalIndex &&
                 static_cast<bool>(candidate.tensorBacking);
        });
    if (hasTensorBacking) {
      for (const DFBPhysicalIndexAssignment &candidate : assignments) {
        if (candidate.physicalIndex != physicalIndex) {
          continue;
        }
        // TODO(#813): Represent empty and unknown launch domains without
        // selecting scratch storage.
        if (!candidate.launchDomain.known ||
            candidate.launchDomain.nodes.empty()) {
          analysisFailure.set(
              candidate.declarations.front(),
              "tensor-backed physical DFB requires an exact non-empty "
              "launch-node domain");
          return failure();
        }
        auto segmentIt = llvm::find_if(
            descriptor.storageSegments,
            [&](const DFBPhysicalStorageSegment &segment) {
              return segment.tensorBacking == candidate.tensorBacking;
            });
        if (segmentIt == descriptor.storageSegments.end()) {
          descriptor.storageSegments.push_back(
              {LaunchNodeDomain{}, candidate.tensorBacking});
          segmentIt = std::prev(descriptor.storageSegments.end());
        }
        segmentIt->launchDomain =
            segmentIt->launchDomain.unionWith(candidate.launchDomain);
      }
      llvm::sort(descriptor.storageSegments,
                 [](const DFBPhysicalStorageSegment &lhs,
                    const DFBPhysicalStorageSegment &rhs) {
                   return *lhs.launchDomain.nodes.begin() <
                          *rhs.launchDomain.nodes.begin();
                 });
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
      if (lhs.tensorBacking != rhs.tensorBacking) {
        analysisFailure.set(
            rhs.declarations.front(),
            "tensor-backed DFB byte ranges partially overlap on a shared "
            "launch node");
        return failure();
      }
      if (lhs.physicalIndex != rhs.physicalIndex) {
        analysisFailure.set(
            rhs.declarations.front(),
            "identical tensor-backed DFB ranges require one proven shared "
            "physical index on a shared launch node");
        return failure();
      }
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
  LLVM_DEBUG(printDFBAllocationDebugReport(llvm::dbgs(), liveness,
                                           plan.conflictModel));

  auto computeAllocation =
      [&](bool requireMinimum) -> FailureOr<PhysicalAllocationCandidate> {
    if (reuseUserDFBs) {
      return computeReuseAllocation(
          moduleOp, liveness, plan.conflictModel, *targetCapacity,
          analysisFailure, exactColoringSearchStateLimit, requireMinimum);
    }
    return computeDistinctUserAllocation(
        moduleOp, liveness, plan.conflictModel, *targetCapacity,
        analysisFailure, exactColoringSearchStateLimit, requireMinimum);
  };
  FailureOr<PhysicalAllocationCandidate> allocation = computeAllocationWithinL1(
      moduleOp, exactColoringSearchStateLimit, l1BudgetOverride,
      analysisFailure, computeAllocation);
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

  FailureOr<SmallVector<DFBPhysicalAllocationDescriptor>> descriptors =
      buildDescriptors(plan.assignments, analysisFailure);
  if (failed(descriptors)) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  plan.descriptors = std::move(*descriptors);

  if (plan.physicalDFBCount > 0) {
    for (func::FuncOp kernel : moduleOp.getOps<func::FuncOp>()) {
      if (kernel->hasAttr(kBaseCTAIndexAttrName)) {
        plan.kernelBaseIndices.push_back({kernel, plan.physicalDFBCount});
      }
    }
  }
}

} // namespace mlir::tt::ttl
