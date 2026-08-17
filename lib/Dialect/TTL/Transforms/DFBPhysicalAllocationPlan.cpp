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
#include <optional>
#include <string>

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

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
        addPairConflicts(model, liveness, lhsIndex, rhsIndex);
      }
    }
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

  static void
  addPairConflicts(DFBPhysicalConflictModel &model,
                   const DFBConcurrentKernelLivenessAnalysis &liveness,
                   unsigned lhsIndex, unsigned rhsIndex) {
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    const DFBLogicalLifecycle &lhs = logicalDFBs[lhsIndex];
    const DFBLogicalLifecycle &rhs = logicalDFBs[rhsIndex];
    auto lhsType = cast<CircularBufferType>(lhs.type);
    auto rhsType = cast<CircularBufferType>(rhs.type);
    if (lhsType.getElementType() != rhsType.getElementType() ||
        (lhs.type != rhs.type && !haveDisjointConfigurationEpochs(lhs, rhs))) {
      addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                  DFBConflictReason::DescriptorMismatch, std::nullopt,
                  lhs.declarations.front(), rhs.declarations.front());
      return;
    }
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
      if (!lhsLifetime || !rhsLifetime || !lhsLifetime->quiescence.proven() ||
          !rhsLifetime->quiescence.proven()) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::UnprovenQuiescence, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
        continue;
      }
      if (haveDisjointConfigurationEpochs(*lhsLifetime, *rhsLifetime)) {
        continue;
      }
      if (lhs.type != rhs.type) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::DescriptorMismatch, node,
                    lhs.declarations.front(), rhs.declarations.front());
        continue;
      }
      if (lhs.tensorBacking != rhs.tensorBacking) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::StorageMismatch, node,
                    lhs.declarations.front(), rhs.declarations.front());
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
            before->terminalStateCanonical ||
            before->terminalTransactionRuns == after->transactionRuns;
        pointerOwnersCompatible =
            before->terminalStateCanonical ||
            (before->terminalWritePointerOwner == after->writePointerOwner &&
             before->terminalReadPointerOwner == after->readPointerOwner);
      } else {
        // Preserve the more specific state diagnosis when lifetimes are also
        // unordered; ordering alone must not obscure a protocol mismatch.
        terminalStateCompatible =
            lhsLifetime->transactionRuns == rhsLifetime->transactionRuns;
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
    unsigned availableIndices, std::uint64_t exactColoringSearchStateLimit,
    DFBAnalysisFailure &analysisFailure,
    ArrayRef<uint64_t> allocationBytesByLogicalIndex,
    std::optional<uint64_t> allocationByteLimit) {
  SmallVector<unsigned> logicalIndices(candidateIndices.begin(),
                                       candidateIndices.end());

  InterferenceGraph interferenceGraph(logicalIndices.size());
  for (unsigned lhsVertex = 0; lhsVertex < logicalIndices.size(); ++lhsVertex) {
    for (unsigned rhsVertex = lhsVertex + 1; rhsVertex < logicalIndices.size();
         ++rhsVertex) {
      unsigned lhsIndex = logicalIndices[lhsVertex];
      unsigned rhsIndex = logicalIndices[rhsVertex];
      if (conflictModel.conflicts(lhsIndex, rhsIndex)) {
        interferenceGraph.addInterference(lhsVertex, rhsVertex);
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

  if (allocationByteLimit && !exactSearchLimitReached &&
      colorCount <= availableIndices) {
    SmallVector<uint64_t> vertexWeights;
    vertexWeights.reserve(logicalIndices.size());
    for (unsigned logicalIndex : logicalIndices) {
      assert(logicalIndex < allocationBytesByLogicalIndex.size());
      vertexWeights.push_back(allocationBytesByLogicalIndex[logicalIndex]);
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
    if (allocationBytes > *allocationByteLimit) {
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
  assert(colors.size() == logicalIndices.size());

  DenseMap<unsigned, int32_t> assignmentByLogicalIndex;
  for (auto [vertex, color] : llvm::enumerate(colors)) {
    assignmentByLogicalIndex[logicalIndices[vertex]] =
        firstPhysicalIndex + static_cast<int32_t>(color);
  }

  for (unsigned lhsVertex = 0; lhsVertex < logicalIndices.size(); ++lhsVertex) {
    for (unsigned rhsVertex = lhsVertex + 1; rhsVertex < logicalIndices.size();
         ++rhsVertex) {
      if (colors[lhsVertex] != colors[rhsVertex]) {
        continue;
      }
      if (interferenceGraph.interferes(lhsVertex, rhsVertex)) {
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
    std::optional<uint64_t> allocationByteLimit) {
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
              *logicalPhysicalIndex, cast<CircularBufferType>(logicalDFB.type),
              failureReason))) {
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
    compilerAllocationByteLimit =
        *fixedUserBytes > *allocationByteLimit
            ? 0
            : *allocationByteLimit - *fixedUserBytes;
  }

  unsigned availableCompilerIndices =
      firstCompilerIndex >= targetMaxDFBIndices
          ? 0
          : static_cast<unsigned>(targetMaxDFBIndices - firstCompilerIndex);
  FailureOr<ConcurrentAssignmentResult> compilerAssignment =
      computeConcurrentAssignments(
          moduleOp, compilerLogicalIndices, firstCompilerIndex, conflictModel,
          availableCompilerIndices, exactColoringSearchStateLimit,
          analysisFailure, allocationBytesByLogicalIndex,
          compilerAllocationByteLimit);
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
        {logicalDFB.logicalId, physicalIndex, logicalDFB.type,
         logicalDFB.tensorBacking, logicalDFB.launchDomain,
         logicalDFB.declarations,
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
    std::uint64_t exactColoringSearchStateLimit,
    ArrayRef<uint64_t> allocationBytesByLogicalIndex,
    std::optional<uint64_t> allocationByteLimit) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  SmallVector<unsigned> logicalIndices =
      llvm::to_vector(llvm::seq<unsigned>(0, logicalDFBs.size()));
  int32_t targetMaxDFBIndices = targetCapacity.indexCount;
  FailureOr<ConcurrentAssignmentResult> assignment =
      computeConcurrentAssignments(
          moduleOp, logicalIndices, /*firstPhysicalIndex=*/0, conflictModel,
          targetMaxDFBIndices, exactColoringSearchStateLimit, analysisFailure,
          allocationBytesByLogicalIndex, allocationByteLimit);
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
         logicalDFB.tensorBacking, logicalDFB.launchDomain,
         logicalDFB.declarations,
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
computeAllocationBytes(ArrayRef<DFBPhysicalIndexAssignment> assignments,
                       std::string &failureReason) {
  DFBAllocationFootprint footprint;
  for (const DFBPhysicalIndexAssignment &assignment : assignments) {
    if (assignment.tensorBacking) {
      continue;
    }
    if (failed(footprint.add(assignment.physicalIndex,
                             cast<CircularBufferType>(assignment.type),
                             failureReason))) {
      return failure();
    }
  }
  return footprint.getTotalBytes();
}

static FailureOr<uint64_t>
computeRequiredL1Bytes(ArrayRef<DFBPhysicalIndexAssignment> assignments,
                       uint64_t reconfigurationStateBytes,
                       std::string &failureReason) {
  FailureOr<uint64_t> allocationBytes =
      computeAllocationBytes(assignments, failureReason);
  if (failed(allocationBytes)) {
    return failure();
  }
  std::optional<uint64_t> requiredBytes =
      llvm::checkedAddUnsigned(*allocationBytes, reconfigurationStateBytes);
  if (!requiredBytes) {
    failureReason = "DFB allocation size is not representable";
    return failure();
  }
  return *requiredBytes;
}

/// Selects an assignment that fits both the target index count and the L1
/// allocation limit. Both user-reuse policies share identical search and
/// diagnostic behavior.
static FailureOr<PhysicalAllocationCandidate> computeAllocationWithinL1(
    ModuleOp moduleOp, std::uint64_t exactColoringSearchStateLimit,
    std::optional<std::uint64_t> l1BudgetOverride,
    DFBAnalysisFailure &analysisFailure,
    llvm::function_ref<FailureOr<PhysicalAllocationCandidate>(
        std::optional<uint64_t>)>
        computeAllocation) {
  std::string allocationSizeFailureReason;
  FailureOr<uint64_t> reconfigurationStateBytes =
      getDFBReconfigurationStateBytes(moduleOp);
  if (failed(reconfigurationStateBytes)) {
    analysisFailure.set(moduleOp,
                        "DFB reconfiguration state size is not representable");
    return failure();
  }
  uint64_t l1BudgetBytes = getUsableDFBL1Bytes(moduleOp, l1BudgetOverride);
  std::optional<uint64_t> allocationByteLimit =
      *reconfigurationStateBytes > l1BudgetBytes
          ? std::optional<uint64_t>(0)
          : std::optional<uint64_t>(l1BudgetBytes -
                                    *reconfigurationStateBytes);
  FailureOr<PhysicalAllocationCandidate> allocation =
      computeAllocation(allocationByteLimit);
  if (failed(allocation)) {
    return failure();
  }
  FailureOr<uint64_t> requiredL1Bytes = computeRequiredL1Bytes(
      allocation->assignments, *reconfigurationStateBytes,
      allocationSizeFailureReason);
  if (failed(requiredL1Bytes)) {
    analysisFailure.set(moduleOp, allocationSizeFailureReason);
    return failure();
  }
  if (allocation->exactSearchLimitReached) {
    setExactSearchLimitFailure(moduleOp, allocation->physicalDFBCount,
                               allocation->exactSearchStateCount,
                               exactColoringSearchStateLimit,
                               "the target L1 budget", analysisFailure);
    return failure();
  }
  if (*requiredL1Bytes > l1BudgetBytes) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "DFB allocation";
    if (*reconfigurationStateBytes > 0) {
      messageStream << " plus reconfiguration state";
    }
    messageStream << " requires " << *requiredL1Bytes
                  << " L1 bytes but the target supports " << l1BudgetBytes;
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }
  return allocation;
}

/// Builds the dense runtime descriptor table without modifying IR.
static FailureOr<SmallVector<DFBPhysicalAllocationDescriptor>>
buildDescriptors(ArrayRef<DFBPhysicalIndexAssignment> assignments,
                 const DFBConcurrentKernelLivenessAnalysis &liveness,
                 DFBAnalysisFailure &analysisFailure) {
  llvm::DenseMap<int32_t, const DFBPhysicalIndexAssignment *> uniqueByIndex;
  for (const DFBPhysicalIndexAssignment &assignment : assignments) {
    uniqueByIndex.try_emplace(assignment.physicalIndex, &assignment);
  }

  DenseMap<int64_t, const DFBLogicalLifecycle *> lifecycleByLogicalId;
  for (const DFBLogicalLifecycle &logicalDFB :
       liveness.getLogicalDFBLifecycles()) {
    lifecycleByLogicalId.try_emplace(logicalDFB.logicalId, &logicalDFB);
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
    DFBPhysicalAllocationDescriptor descriptor;
    descriptor.physicalIndex = physicalIndex;
    auto addConfiguration =
        [&](const DFBPhysicalIndexAssignment &candidate,
            std::optional<int64_t> entryReconfigurationOrdinal,
            LaunchNodeDomain activeDomain) -> LogicalResult {
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
        configurationIt = std::prev(descriptor.epochConfigurations.end());
      } else if (configurationIt->numTiles != numTiles ||
                 configurationIt->elementType != dfbType.getElementType() ||
                 configurationIt->pageSize != pageSize ||
                 configurationIt->blockCount != blockCount) {
        analysisFailure.set(
            candidate.declarations.front(),
            "one physical DFB has inconsistent configurations in one epoch");
        return failure();
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

    for (const DFBPhysicalIndexAssignment &candidate : assignments) {
      if (candidate.physicalIndex != physicalIndex) {
        continue;
      }
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

    llvm::sort(descriptor.epochConfigurations,
               [](const DFBConfigurationEpochDescriptor &lhs,
                  const DFBConfigurationEpochDescriptor &rhs) {
                 if (!lhs.entryReconfigurationOrdinal) {
                   return rhs.entryReconfigurationOrdinal.has_value();
                 }
                 return rhs.entryReconfigurationOrdinal &&
                        *lhs.entryReconfigurationOrdinal <
                            *rhs.entryReconfigurationOrdinal;
               });
    assert(!descriptor.epochConfigurations.empty() &&
           "every physical DFB must have one configuration");
    for (DFBConfigurationEpochDescriptor &configuration :
         descriptor.epochConfigurations) {
      llvm::sort(configuration.storageSegments,
                 [](const DFBPhysicalStorageSegment &lhs,
                    const DFBPhysicalStorageSegment &rhs) {
                   return *lhs.launchDomain.nodes.begin() <
                          *rhs.launchDomain.nodes.begin();
                 });
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
    Operation *operation, bool reuseUserDFBs,
    std::uint64_t exactColoringSearchStateLimit,
    std::optional<std::uint64_t> l1BudgetOverride,
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
  plan.conflictModel = DFBPhysicalConflictModelBuilder::build(
      liveness, staticConfigurationConflicts);
  LLVM_DEBUG(printDFBAllocationDebugReport(llvm::dbgs(), liveness,
                                           plan.conflictModel));

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
    FailureOr<uint64_t> allocationBytes = getDFBAllocationSizeBytes(
        cast<CircularBufferType>(logicalDFB.type), allocationFailureReason);
    if (failed(allocationBytes)) {
      errorOperation = logicalDFB.declarations.front();
      errorMessage = std::move(allocationFailureReason);
      return;
    }
    allocationBytesByLogicalIndex.push_back(*allocationBytes);
  }

  auto computeAllocation = [&](std::optional<uint64_t> allocationByteLimit)
      -> FailureOr<PhysicalAllocationCandidate> {
    if (reuseUserDFBs) {
      return computeReuseAllocation(
          moduleOp, liveness, plan.conflictModel, *targetCapacity,
          analysisFailure, exactColoringSearchStateLimit,
          allocationBytesByLogicalIndex, allocationByteLimit);
    }
    return computeDistinctUserAllocation(
        moduleOp, liveness, plan.conflictModel, *targetCapacity,
        analysisFailure, exactColoringSearchStateLimit,
        allocationBytesByLogicalIndex, allocationByteLimit);
  };
  FailureOr<PhysicalAllocationCandidate> allocation =
      computeAllocationWithinL1(moduleOp, exactColoringSearchStateLimit,
                                l1BudgetOverride, analysisFailure,
                                computeAllocation);
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
      buildDescriptors(plan.assignments, liveness, analysisFailure);
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
