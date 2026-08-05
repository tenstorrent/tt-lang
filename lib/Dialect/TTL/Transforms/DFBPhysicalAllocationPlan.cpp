// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBPhysicalAllocationPlan.h"

#include "DFBAllocationLimits.h"
#include "DFBAnalysisFailure.h"
#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/InterferenceGraphColoring.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

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

} // namespace

class DFBPhysicalConflictModelBuilder {
public:
  static DFBPhysicalConflictModel
  build(const DFBConcurrentKernelLivenessAnalysis &liveness) {
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
    if (lhs.type != rhs.type) {
      addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                  DFBConflictReason::DescriptorMismatch, std::nullopt,
                  lhs.declarations.front(), rhs.declarations.front());
      return;
    }
    if (!lhs.launchDomain.known || !rhs.launchDomain.known) {
      addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                  DFBConflictReason::UnknownLaunchNodeDomain, std::nullopt,
                  lhs.declarations.front(), rhs.declarations.front());
      return;
    }

    LaunchNodeDomain sharedNodes =
        lhs.launchDomain.intersectWith(rhs.launchDomain);
    for (LaunchNodeCoord node : sharedNodes.nodes) {
      const DFBPerNodeLifetime *lhsLifetime = lhs.findNodeLifetime(node);
      const DFBPerNodeLifetime *rhsLifetime = rhs.findNodeLifetime(node);
      if (!lhsLifetime || !rhsLifetime || !lhsLifetime->quiescence.proven() ||
          !rhsLifetime->quiescence.proven()) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::UnprovenQuiescence, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
        continue;
      }
      if (lhsLifetime->transactionTileCount !=
          rhsLifetime->transactionTileCount) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::TransactionMismatch, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
        continue;
      }
      if (lhsLifetime->writePointerOwner != rhsLifetime->writePointerOwner ||
          lhsLifetime->readPointerOwner != rhsLifetime->readPointerOwner) {
        addEvidence(model, lhs, rhs, lhsIndex, rhsIndex,
                    DFBConflictReason::PointerOwnerMismatch, node,
                    getLifetimeEvidence(lhsLifetime, lhs),
                    getLifetimeEvidence(rhsLifetime, rhs));
        continue;
      }
      if (!liveness.isOrderedBefore(lhsIndex, rhsIndex, node) &&
          !liveness.isOrderedBefore(rhsIndex, lhsIndex, node)) {
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
/// it fits `availableIndices`. Otherwise one exhaustive fixed-limit search
/// decides whether some assignment fits. A minimum physical-index-count search
/// runs only for an L1-budget decision. `firstPhysicalIndex` reserves lower
/// index values without changing which DFB pairs may share.
static FailureOr<ConcurrentAssignmentResult> computeConcurrentAssignments(
    ModuleOp moduleOp, ArrayRef<unsigned> candidateIndices,
    int32_t firstPhysicalIndex, const DFBPhysicalConflictModel &conflictModel,
    unsigned availableIndices, std::uint64_t exactColoringSearchStateLimit,
    DFBAnalysisFailure &analysisFailure, bool requireMinimum = false) {
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

/// Derives capacity text from the enforced constant so diagnostics cannot
/// report a stale limit.
static std::string getPhysicalIndexLimitDescription() {
  return "the " + std::to_string(kMaxCircularBuffers) + "-index hardware limit";
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
    DFBAnalysisFailure &analysisFailure,
    std::uint64_t exactColoringSearchStateLimit, bool requireMinimum = false) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  DenseMap<int64_t, int32_t> compactedUserIndices =
      computeCompactedUserIndices(logicalDFBs);
  int32_t firstCompilerIndex =
      static_cast<int32_t>(compactedUserIndices.size());
  SmallVector<unsigned> compilerLogicalIndices;
  for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
    if (indexedLogicalDFB.value().compilerCreated) {
      compilerLogicalIndices.push_back(indexedLogicalDFB.index());
    }
  }

  unsigned availableCompilerIndices =
      firstCompilerIndex >= kMaxCircularBuffers
          ? 0
          : static_cast<unsigned>(kMaxCircularBuffers - firstCompilerIndex);
  FailureOr<ConcurrentAssignmentResult> compilerAssignment =
      computeConcurrentAssignments(
          moduleOp, compilerLogicalIndices, firstCompilerIndex, conflictModel,
          availableCompilerIndices, exactColoringSearchStateLimit,
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

    allocation.assignments.push_back({logicalDFB.logicalId, physicalIndex,
                                      logicalDFB.type, logicalDFB.declarations,
                                      logicalDFB.bounded});
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
  if (allocation.physicalDFBCount > kMaxCircularBuffers) {
    if (allocation.exactSearchLimitReached) {
      setExactSearchLimitFailure(
          moduleOp, allocation.physicalDFBCount,
          allocation.exactSearchStateCount, exactColoringSearchStateLimit,
          getPhysicalIndexLimitDescription(), analysisFailure);
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
    messageStream << " unspilled DFB indices but hardware supports at most "
                  << kMaxCircularBuffers << " (" << compilerSlotCount
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
    DFBAnalysisFailure &analysisFailure,
    std::uint64_t exactColoringSearchStateLimit, bool requireMinimum) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  SmallVector<unsigned> logicalIndices =
      llvm::to_vector(llvm::seq<unsigned>(0, logicalDFBs.size()));
  FailureOr<ConcurrentAssignmentResult> assignment =
      computeConcurrentAssignments(
          moduleOp, logicalIndices, /*firstPhysicalIndex=*/0, conflictModel,
          kMaxCircularBuffers, exactColoringSearchStateLimit, analysisFailure,
          requireMinimum);
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
    allocation.assignments.push_back({logicalDFB.logicalId, physicalIndex,
                                      logicalDFB.type, logicalDFB.declarations,
                                      logicalDFB.bounded});
  }

  if (allocation.physicalDFBCount <= kMaxCircularBuffers) {
    return allocation;
  }
  if (allocation.exactSearchLimitReached) {
    setExactSearchLimitFailure(
        moduleOp, allocation.physicalDFBCount, allocation.exactSearchStateCount,
        exactColoringSearchStateLimit, getPhysicalIndexLimitDescription(),
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
  messageStream << " unspilled physical indices but hardware supports at most "
                << kMaxCircularBuffers;
  analysisFailure.set(moduleOp, messageStream.str());
  return failure();
}

/// Returns the L1 bytes required by the unique physical assignments.
static FailureOr<uint64_t>
computeAllocationBytes(ArrayRef<DFBPhysicalIndexAssignment> assignments) {
  DFBAllocationFootprint footprint;
  for (const DFBPhysicalIndexAssignment &assignment : assignments) {
    if (failed(footprint.add(assignment.physicalIndex,
                             cast<CircularBufferType>(assignment.type)))) {
      return failure();
    }
  }
  return footprint.getTotalBytes();
}

/// Recomputes an assignment with the minimum physical-index count only when a
/// valid first-fit assignment exceeds the L1 budget. Both user-reuse policies
/// therefore share identical search and diagnostic behavior.
static FailureOr<PhysicalAllocationCandidate> computeAllocationWithinL1(
    ModuleOp moduleOp, std::uint64_t exactColoringSearchStateLimit,
    DFBAnalysisFailure &analysisFailure,
    llvm::function_ref<FailureOr<PhysicalAllocationCandidate>(bool)>
        computeAllocation) {
  FailureOr<PhysicalAllocationCandidate> allocation =
      computeAllocation(/*requireMinimum=*/false);
  if (failed(allocation)) {
    return failure();
  }

  FailureOr<uint64_t> allocationBytes =
      computeAllocationBytes(allocation->assignments);
  uint64_t l1BudgetBytes = getUsableDFBL1Bytes(moduleOp);
  if (succeeded(allocationBytes) && *allocationBytes > l1BudgetBytes &&
      !allocation->minimumProven) {
    allocation = computeAllocation(/*requireMinimum=*/true);
    if (failed(allocation)) {
      return failure();
    }
    allocationBytes = computeAllocationBytes(allocation->assignments);
  }
  if (allocation->exactSearchLimitReached) {
    setExactSearchLimitFailure(moduleOp, allocation->physicalDFBCount,
                               allocation->exactSearchStateCount,
                               exactColoringSearchStateLimit,
                               "the target L1 budget", analysisFailure);
    return failure();
  }
  if (failed(allocationBytes)) {
    analysisFailure.set(moduleOp,
                        "DFB allocation has an invalid negative element count");
    return failure();
  }
  if (*allocationBytes > l1BudgetBytes) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "DFB allocation requires " << *allocationBytes
                  << " L1 bytes but the target supports " << l1BudgetBytes;
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
    if (!inserted && existingIt->second->type != assignment.type) {
      BindCBOp declaration = assignment.declarations.front();
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "physical DFB index " << assignment.physicalIndex
                    << " has inconsistent CircularBufferType values "
                    << existingIt->second->type << " and " << assignment.type;
      analysisFailure.set(declaration, messageStream.str());
      return failure();
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
    auto [physicalIndex, assignment] = indexedAssignment;
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
    descriptors.push_back(
        {physicalIndex, static_cast<int32_t>(dfbType.getElementsPerBlock()),
         dfbType.getElementType(),
         static_cast<int32_t>(
             ttcore::getElementSizeBytes(dfbType.getElementType())),
         static_cast<int32_t>(dfbType.getBlockCount())});
  }
  return descriptors;
}

} // namespace

DFBPhysicalAllocationPlanner::DFBPhysicalAllocationPlanner(
    Operation *operation, bool reuseUserDFBs,
    std::uint64_t exactColoringSearchStateLimit,
    AnalysisManager analysisManager) {
  ModuleOp moduleOp = cast<ModuleOp>(operation);
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
  plan.conflictModel = DFBPhysicalConflictModelBuilder::build(liveness);

  auto computeAllocation =
      [&](bool requireMinimum) -> FailureOr<PhysicalAllocationCandidate> {
    if (reuseUserDFBs) {
      return computeReuseAllocation(
          moduleOp, liveness, plan.conflictModel, analysisFailure,
          exactColoringSearchStateLimit, requireMinimum);
    }
    return computeDistinctUserAllocation(
        moduleOp, liveness, plan.conflictModel, analysisFailure,
        exactColoringSearchStateLimit, requireMinimum);
  };
  FailureOr<PhysicalAllocationCandidate> allocation =
      computeAllocationWithinL1(moduleOp, exactColoringSearchStateLimit,
                                analysisFailure, computeAllocation);
  if (failed(allocation)) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }
  plan.assignments = std::move(allocation->assignments);
  plan.physicalDFBCount = allocation->physicalDFBCount;

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
