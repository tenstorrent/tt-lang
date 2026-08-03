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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

namespace {

/// Returns true when an attribute contains a copied physical DFB index.
static bool isDerivedDFBIndexAttribute(StringRef attributeName) {
  return attributeName == kUnpackToDestFp32AttrName ||
         attributeName.starts_with(kCBIndexAttrPrefix) ||
         attributeName == kBcastOutputCBIndexAttrName ||
         attributeName == kReduceOutputCBIndexAttrName ||
         attributeName == kTransposeOutputCBIndexAttrName;
}

/// Rejects IR containing copies of provisional physical DFB indices.
///
/// The listed attributes directly copy `cb_index`. Reassigning a declaration
/// after such a copy exists would leave the copy stale. Attribute-name matching
/// remains valid across operation and region changes, but this predicate must
/// include every attribute introduced by a pass that copies a DFB index.
static LogicalResult
verifyFinalizationPrecedesIndexCopies(ModuleOp moduleOp,
                                      DFBAnalysisFailure &analysisFailure) {
  WalkResult walkResult = moduleOp->walk([&](Operation *operation) {
    for (NamedAttribute attribute : operation->getAttrs()) {
      StringRef attributeName = attribute.getName().getValue();
      if (!isDerivedDFBIndexAttribute(attributeName)) {
        continue;
      }
      analysisFailure.set(
          operation,
          ("contains derived DFB-index attribute '" + attributeName +
           "' before DFB index finalization; run ttl-finalize-dfb-indices "
           "before ttl-set-compute-kernel-config and "
           "ttl-annotate-cb-associations")
              .str());
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

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

/// Colors the selected logical DFBs using concurrent-lifetime interference.
///
/// Candidate indices refer to the liveness analysis result. Physical indices
/// begin at `firstPhysicalIndex`, which permits user assignments to reserve a
/// prefix without changing the lifetime or coloring implementation.
struct ConcurrentAssignmentResult {
  SmallVector<int32_t> assignments;
  unsigned colorCount = 0;
  unsigned cliqueLowerBound = 0;
  bool minimumProven = false;
  bool exactSearchLimitReached = false;
  std::uint64_t exactSearchStateCount = 0;
};

static ConcurrentAssignmentResult computeConcurrentAssignments(
    ArrayRef<unsigned> candidateIndices, int32_t firstPhysicalIndex,
    const DFBPhysicalConflictModel &conflictModel, unsigned availableIndices,
    std::uint64_t exactColoringSearchStateLimit, bool requireMinimum = false) {
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
  bool minimumProven = bounds.provesMinimum();
  bool capacityAlreadyProven = colorCount > availableIndices &&
                               bounds.cliqueLowerBound > availableIndices;
  ExactInterferenceGraphColoring exactColoring;
  if ((!minimumProven && requireMinimum) ||
      (colorCount > availableIndices && !capacityAlreadyProven)) {
    exactColoring = colorInterferenceGraphExactly(
        interferenceGraph, exactColoringSearchStateLimit);
    if (exactColoring.isOptimal()) {
      selectedColors = std::move(exactColoring.colors);
      colorCount = exactColoring.colorCount;
      minimumProven = true;
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
        llvm_unreachable("exact coloring assigned one color to a conflict");
      }
    }
  }

  ConcurrentAssignmentResult result;
  result.assignments.reserve(candidateIndices.size());
  for (unsigned logicalIndex : candidateIndices) {
    result.assignments.push_back(assignmentByLogicalIndex.lookup(logicalIndex));
  }
  result.colorCount = colorCount;
  result.cliqueLowerBound = bounds.cliqueLowerBound;
  result.minimumProven = minimumProven;
  result.exactSearchLimitReached = !exactColoring.isOptimal();
  result.exactSearchStateCount = exactColoring.exploredStateCount;
  return result;
}

static void
setExactSearchLimitFailure(ModuleOp moduleOp, unsigned firstFitCount,
                           std::uint64_t exactSearchStateCount,
                           std::uint64_t exactColoringSearchStateLimit,
                           StringRef constrainedResource,
                           DFBAnalysisFailure &analysisFailure) {
  std::string message;
  llvm::raw_string_ostream messageStream(message);
  messageStream << "deterministic first-fit uses " << firstFitCount
                << " physical DFB indices; exact coloring explored "
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

/// Assignments produced without introducing new sharing among user DFBs.
struct DistinctUserAllocation {
  SmallVector<DFBPhysicalIndexAssignment> assignments;
  int32_t physicalDFBCount = 0;
  bool compilerMinimumProven = false;
  bool exactSearchLimitReached = false;
  std::uint64_t exactSearchStateCount = 0;
};

/// Compacts user indices and colors compiler-created DFB lifetimes.
///
/// User declarations retain sharing already expressed by equal provisional
/// indices. Compiler-created DFBs use the same concurrent lifetime proof as
/// the all-DFB allocator, so the option changes allocation policy without
/// selecting a second lifetime model.
static FailureOr<DistinctUserAllocation> computeDistinctUserAllocation(
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
  ConcurrentAssignmentResult compilerAssignment = computeConcurrentAssignments(
      compilerLogicalIndices, firstCompilerIndex, conflictModel,
      availableCompilerIndices, exactColoringSearchStateLimit, requireMinimum);
  DenseMap<unsigned, int32_t> compilerIndexByLogicalIndex;
  for (auto indexedCompilerIndex : llvm::enumerate(compilerLogicalIndices)) {
    compilerIndexByLogicalIndex[indexedCompilerIndex.value()] =
        compilerAssignment.assignments[indexedCompilerIndex.index()];
  }

  DistinctUserAllocation allocation;
  allocation.compilerMinimumProven = compilerAssignment.minimumProven;
  allocation.exactSearchLimitReached =
      compilerAssignment.exactSearchLimitReached;
  allocation.exactSearchStateCount = compilerAssignment.exactSearchStateCount;
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
          "the 32-index hardware limit", analysisFailure);
      return failure();
    }
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    if (allocation.compilerMinimumProven) {
      messageStream << "need " << allocation.physicalDFBCount;
    } else {
      messageStream << "need at least "
                    << firstCompilerIndex +
                           static_cast<int32_t>(
                               compilerAssignment.cliqueLowerBound);
    }
    messageStream << " unspilled DFB indices but hardware supports at most "
                  << kMaxCircularBuffers << " (" << compilerSlotCount
                  << " compiler-allocated after proven reuse)";
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }
  return allocation;
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
      llvm_unreachable("physical DFB allocation plan is not dense");
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

  DFBAnalysisFailure analysisFailure;
  if (failed(
          verifyFinalizationPrecedesIndexCopies(moduleOp, analysisFailure))) {
    errorOperation = analysisFailure.operation;
    errorMessage = std::move(analysisFailure.message);
    return;
  }

  const DFBConcurrentKernelLivenessAnalysis &liveness =
      analysisManager.getAnalysis<DFBConcurrentKernelLivenessAnalysis>();
  if (!liveness.succeeded()) {
    errorOperation = liveness.getErrorOperation();
    errorMessage = liveness.getErrorMessage().str();
    return;
  }
  plan.conflictModel = DFBPhysicalConflictModelBuilder::build(liveness);

  if (reuseUserDFBs) {
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    SmallVector<unsigned> logicalIndices =
        llvm::to_vector(llvm::seq<unsigned>(0, logicalDFBs.size()));

    ConcurrentAssignmentResult assignment = computeConcurrentAssignments(
        logicalIndices, 0, plan.conflictModel, kMaxCircularBuffers,
        exactColoringSearchStateLimit);
    auto recordAssignments = [&]() {
      plan.assignments.clear();
      plan.physicalDFBCount = 0;
      for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
        const DFBLogicalLifecycle &logicalDFB = indexedLogicalDFB.value();
        int32_t physicalIndex =
            assignment.assignments[indexedLogicalDFB.index()];
        plan.physicalDFBCount =
            std::max(plan.physicalDFBCount, physicalIndex + 1);
        plan.assignments.push_back({logicalDFB.logicalId, physicalIndex,
                                    logicalDFB.type, logicalDFB.declarations,
                                    logicalDFB.bounded});
      }
    };
    recordAssignments();
    if (plan.physicalDFBCount > kMaxCircularBuffers) {
      if (assignment.exactSearchLimitReached) {
        setExactSearchLimitFailure(
            moduleOp, plan.physicalDFBCount, assignment.exactSearchStateCount,
            exactColoringSearchStateLimit, "the 32-index hardware limit",
            analysisFailure);
        errorOperation = analysisFailure.operation;
        errorMessage = std::move(analysisFailure.message);
        return;
      }
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      if (assignment.minimumProven) {
        messageStream << "DFB allocation needs " << plan.physicalDFBCount;
      } else {
        messageStream << "DFB allocation needs at least "
                      << assignment.cliqueLowerBound;
      }
      messageStream << " unspilled physical indices but hardware supports at "
                       "most "
                    << kMaxCircularBuffers;
      errorOperation = moduleOp;
      errorMessage = messageStream.str();
      return;
    }

    FailureOr<uint64_t> allocationBytes =
        computeAllocationBytes(plan.assignments);
    uint64_t l1BudgetBytes = getUsableDFBL1Bytes(moduleOp);
    if (mlir::succeeded(allocationBytes) && *allocationBytes > l1BudgetBytes &&
        !assignment.minimumProven) {
      assignment = computeConcurrentAssignments(
          logicalIndices, 0, plan.conflictModel, kMaxCircularBuffers,
          exactColoringSearchStateLimit,
          /*requireMinimum=*/true);
      recordAssignments();
      allocationBytes = computeAllocationBytes(plan.assignments);
    }
    if (assignment.exactSearchLimitReached) {
      setExactSearchLimitFailure(moduleOp, plan.physicalDFBCount,
                                 assignment.exactSearchStateCount,
                                 exactColoringSearchStateLimit,
                                 "the target L1 budget", analysisFailure);
      errorOperation = analysisFailure.operation;
      errorMessage = std::move(analysisFailure.message);
      return;
    }
    if (failed(allocationBytes)) {
      errorOperation = moduleOp;
      errorMessage = "DFB allocation has an invalid negative element count";
      return;
    }
    if (*allocationBytes > l1BudgetBytes) {
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB allocation requires " << *allocationBytes
                    << " L1 bytes but the target supports " << l1BudgetBytes;
      errorOperation = moduleOp;
      errorMessage = messageStream.str();
      return;
    }
  } else {
    FailureOr<DistinctUserAllocation> allocation =
        computeDistinctUserAllocation(moduleOp, liveness, plan.conflictModel,
                                      analysisFailure,
                                      exactColoringSearchStateLimit);
    if (failed(allocation)) {
      errorOperation = analysisFailure.operation;
      errorMessage = std::move(analysisFailure.message);
      return;
    }
    plan.assignments = std::move(allocation->assignments);
    plan.physicalDFBCount = allocation->physicalDFBCount;

    FailureOr<uint64_t> allocationBytes =
        computeAllocationBytes(plan.assignments);
    uint64_t l1BudgetBytes = getUsableDFBL1Bytes(moduleOp);
    if (mlir::succeeded(allocationBytes) && *allocationBytes > l1BudgetBytes &&
        !allocation->compilerMinimumProven) {
      allocation = computeDistinctUserAllocation(
          moduleOp, liveness, plan.conflictModel, analysisFailure,
          exactColoringSearchStateLimit,
          /*requireMinimum=*/true);
      if (failed(allocation)) {
        errorOperation = analysisFailure.operation;
        errorMessage = std::move(analysisFailure.message);
        return;
      }
      plan.assignments = std::move(allocation->assignments);
      plan.physicalDFBCount = allocation->physicalDFBCount;
      allocationBytes = computeAllocationBytes(plan.assignments);
    }
    if (allocation->exactSearchLimitReached) {
      setExactSearchLimitFailure(moduleOp, plan.physicalDFBCount,
                                 allocation->exactSearchStateCount,
                                 exactColoringSearchStateLimit,
                                 "the target L1 budget", analysisFailure);
      errorOperation = analysisFailure.operation;
      errorMessage = std::move(analysisFailure.message);
      return;
    }
    if (failed(allocationBytes)) {
      errorOperation = moduleOp;
      errorMessage = "DFB allocation has an invalid negative element count";
      return;
    }
    if (*allocationBytes > l1BudgetBytes) {
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB allocation requires " << *allocationBytes
                    << " L1 bytes but the target supports " << l1BudgetBytes;
      errorOperation = moduleOp;
      errorMessage = messageStream.str();
      return;
    }
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
