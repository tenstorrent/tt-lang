// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBPhysicalAllocationPlan.h"

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

/// Rejects compiler-created DFBs with incomplete producer or consumer
/// lifecycles because no bounded live interval can be proven for them.
static LogicalResult
verifyCompilerDFBLifecycles(ModuleOp moduleOp,
                            DFBAnalysisFailure &analysisFailure) {
  WalkResult walkResult = moduleOp->walk([&](BindCBOp bindOp) {
    if (!bindOp->hasAttr(kCompilerAllocatedAttrName)) {
      return WalkResult::advance();
    }

    bool hasReserve = false;
    bool hasPush = false;
    bool hasWait = false;
    bool hasPop = false;
    for (Operation *user : bindOp.getResult().getUsers()) {
      hasReserve |= isa<CBReserveOp>(user);
      hasPush |= isa<CBPushOp>(user);
      hasWait |= isa<CBWaitOp>(user);
      hasPop |= isa<CBPopOp>(user);
    }

    if (!hasReserve && !hasPush && !hasWait && !hasPop) {
      return WalkResult::advance();
    }

    StringRef missingOperation;
    if (!hasReserve) {
      missingOperation = "ttl.cb_reserve";
    } else if (!hasPush) {
      missingOperation = "ttl.cb_push";
    } else if (!hasWait) {
      missingOperation = "ttl.cb_wait";
    } else if (!hasPop) {
      missingOperation = "ttl.cb_pop";
    } else {
      return WalkResult::advance();
    }

    analysisFailure.set(
        bindOp, ("compiler-allocated DFB has a partial lifecycle: missing " +
                 missingOperation)
                    .str());
    return WalkResult::interrupt();
  });
  return success(!walkResult.wasInterrupted());
}

/// Returns true unless two logical DFBs can share one physical allocation.
static bool
logicalDFBsConflict(const DFBLogicalLifecycle &lhs,
                    const DFBLogicalLifecycle &rhs, unsigned lhsIndex,
                    unsigned rhsIndex,
                    const DFBConcurrentKernelLivenessAnalysis &liveness) {
  // TT-Metal stores each physical DFB's counters and ring pointers in the
  // producer and consumer kernels. Different kernels have independent state.
  if (lhs.type != rhs.type || lhs.producerKernel != rhs.producerKernel ||
      lhs.consumerKernel != rhs.consumerKernel) {
    return true;
  }
  if (!lhs.transactionTileCount.has_value() ||
      lhs.transactionTileCount != rhs.transactionTileCount) {
    return true;
  }
  auto dfbType = cast<CircularBufferType>(lhs.type);
  if (dfbType.getTotalElements() % *lhs.transactionTileCount != 0) {
    return true;
  }
  return !liveness.isOrderedBefore(lhsIndex, rhsIndex) &&
         !liveness.isOrderedBefore(rhsIndex, lhsIndex);
}

/// Colors the selected logical DFBs using concurrent-lifetime interference.
///
/// Candidate indices refer to the liveness analysis result. Physical indices
/// begin at `firstPhysicalIndex`, which permits user assignments to reserve a
/// prefix without changing the lifetime or coloring implementation.
static FailureOr<SmallVector<int32_t>> computeConcurrentAssignments(
    ModuleOp moduleOp, const DFBConcurrentKernelLivenessAnalysis &liveness,
    ArrayRef<unsigned> candidateIndices, int32_t firstPhysicalIndex,
    const InterferenceGraphColoring &coloring,
    DFBAnalysisFailure &analysisFailure) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  SmallVector<unsigned> coloringOrder =
      llvm::to_vector(llvm::seq<unsigned>(0, candidateIndices.size()));
  llvm::sort(coloringOrder, [&](unsigned lhsVertex, unsigned rhsVertex) {
    unsigned lhsIndex = candidateIndices[lhsVertex];
    unsigned rhsIndex = candidateIndices[rhsVertex];
    const DFBLogicalLifecycle &lhs = logicalDFBs[lhsIndex];
    const DFBLogicalLifecycle &rhs = logicalDFBs[rhsIndex];
    return lhs.logicalId != rhs.logicalId ? lhs.logicalId < rhs.logicalId
                                          : lhsVertex < rhsVertex;
  });

  InterferenceGraph interferenceGraph(candidateIndices.size());
  for (unsigned lhsVertex = 0; lhsVertex < candidateIndices.size();
       ++lhsVertex) {
    for (unsigned rhsVertex = lhsVertex + 1;
         rhsVertex < candidateIndices.size(); ++rhsVertex) {
      unsigned lhsIndex = candidateIndices[lhsVertex];
      unsigned rhsIndex = candidateIndices[rhsVertex];
      if (logicalDFBsConflict(logicalDFBs[lhsIndex], logicalDFBs[rhsIndex],
                              lhsIndex, rhsIndex, liveness)) {
        interferenceGraph.addInterference(lhsVertex, rhsVertex);
      }
    }
  }

  SmallVector<unsigned> colors =
      coloring.color(interferenceGraph, coloringOrder);
  if (colors.size() != candidateIndices.size()) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "DFB coloring returned " << colors.size()
                  << " assignments for " << candidateIndices.size()
                  << " logical DFBs";
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }

  unsigned colorCount = 0;
  for (auto [vertex, color] : llvm::enumerate(colors)) {
    if (color >= candidateIndices.size()) {
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB coloring assigned out-of-range color " << color
                    << " to vertex " << vertex << "; expected less than "
                    << candidateIndices.size();
      analysisFailure.set(moduleOp, messageStream.str());
      return failure();
    }
    colorCount = std::max(colorCount, color + 1);
  }
  llvm::BitVector usedColors(colorCount);
  for (unsigned color : colors) {
    usedColors.set(color);
  }
  if (!usedColors.all()) {
    unsigned missingColor = usedColors.find_first_unset();
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "DFB coloring must use dense zero-based colors; color "
                  << missingColor << " is unused";
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }

  SmallVector<int32_t> assignments;
  assignments.reserve(colors.size());
  for (unsigned color : colors) {
    assignments.push_back(firstPhysicalIndex + static_cast<int32_t>(color));
  }

  for (unsigned lhsVertex = 0; lhsVertex < candidateIndices.size();
       ++lhsVertex) {
    for (unsigned rhsVertex = lhsVertex + 1;
         rhsVertex < candidateIndices.size(); ++rhsVertex) {
      if (assignments[lhsVertex] != assignments[rhsVertex]) {
        continue;
      }
      if (interferenceGraph.interferes(lhsVertex, rhsVertex)) {
        unsigned lhsIndex = candidateIndices[lhsVertex];
        unsigned rhsIndex = candidateIndices[rhsVertex];
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "DFB allocator assigned interfering logical DFBs "
                      << logicalDFBs[lhsIndex].logicalId << " and "
                      << logicalDFBs[rhsIndex].logicalId
                      << " to physical index " << assignments[lhsVertex];
        analysisFailure.set(moduleOp, messageStream.str());
        return failure();
      }
    }
  }

  return assignments;
}

/// Maps provisional user indices to a dense physical index range.
///
/// Frontend indices identify allocation groups before finalization but may
/// contain gaps when an uncaptured DFB has no declarations. Compacting the
/// distinct values preserves existing sharing without making logical identity
/// depend on provisional numbering.
static DenseMap<int64_t, int32_t>
computeCompactedUserIndices(ModuleOp moduleOp) {
  DenseSet<int64_t> provisionalIndices;
  moduleOp->walk([&](BindCBOp bindOp) {
    if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
      return;
    }
    provisionalIndices.insert(bindOp.getCbIndex().getSExtValue());
  });

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
};

/// Compacts user indices and colors compiler-created DFB lifetimes.
///
/// User declarations retain sharing already expressed by equal provisional
/// indices. Compiler-created DFBs use the same concurrent lifetime proof as
/// the all-DFB allocator, so the option changes allocation policy without
/// selecting a second lifetime model.
static FailureOr<DistinctUserAllocation> computeDistinctUserAllocation(
    ModuleOp moduleOp, const DFBConcurrentKernelLivenessAnalysis &liveness,
    const InterferenceGraphColoring &coloring,
    DFBAnalysisFailure &analysisFailure) {
  DenseMap<int64_t, int32_t> compactedUserIndices =
      computeCompactedUserIndices(moduleOp);
  int32_t firstCompilerIndex =
      static_cast<int32_t>(compactedUserIndices.size());

  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  SmallVector<unsigned> compilerLogicalIndices;
  for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
    if (indexedLogicalDFB.value().compilerCreated) {
      compilerLogicalIndices.push_back(indexedLogicalDFB.index());
    }
  }

  FailureOr<SmallVector<int32_t>> compilerPhysicalIndices =
      computeConcurrentAssignments(moduleOp, liveness, compilerLogicalIndices,
                                   firstCompilerIndex, coloring,
                                   analysisFailure);
  if (failed(compilerPhysicalIndices)) {
    return failure();
  }
  DenseMap<unsigned, int32_t> compilerIndexByLogicalIndex;
  for (auto indexedCompilerIndex : llvm::enumerate(compilerLogicalIndices)) {
    compilerIndexByLogicalIndex[indexedCompilerIndex.value()] =
        (*compilerPhysicalIndices)[indexedCompilerIndex.index()];
  }

  DistinctUserAllocation allocation;
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

  int32_t compilerSlotCount = allocation.physicalDFBCount - firstCompilerIndex;
  if (allocation.physicalDFBCount > kMaxCircularBuffers) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "need " << allocation.physicalDFBCount
                  << " DFB indices but hardware supports at most "
                  << kMaxCircularBuffers << " (" << compilerSlotCount
                  << " compiler-allocated after reuse); reduce the number of "
                     "user-declared dataflow buffers or split the computation "
                     "into multiple kernels";
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
      messageStream
          << "physical DFB indices must form a dense zero-based range; "
             "expected index "
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
    Operation *operation, bool reuseUserDFBs, AnalysisManager analysisManager)
    : DFBPhysicalAllocationPlanner(
          operation, reuseUserDFBs, analysisManager,
          getGreedyFirstFitInterferenceGraphColoring()) {}

DFBPhysicalAllocationPlanner::DFBPhysicalAllocationPlanner(
    Operation *operation, bool reuseUserDFBs, AnalysisManager analysisManager,
    const InterferenceGraphColoring &coloring) {
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

  if (failed(verifyCompilerDFBLifecycles(moduleOp, analysisFailure))) {
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

  if (reuseUserDFBs) {
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    SmallVector<unsigned> logicalIndices =
        llvm::to_vector(llvm::seq<unsigned>(0, logicalDFBs.size()));

    FailureOr<SmallVector<int32_t>> physicalIndices =
        computeConcurrentAssignments(moduleOp, liveness, logicalIndices, 0,
                                     coloring, analysisFailure);
    if (failed(physicalIndices)) {
      errorOperation = analysisFailure.operation;
      errorMessage = std::move(analysisFailure.message);
      return;
    }
    for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
      const DFBLogicalLifecycle &logicalDFB = indexedLogicalDFB.value();
      int32_t physicalIndex = (*physicalIndices)[indexedLogicalDFB.index()];
      plan.physicalDFBCount =
          std::max(plan.physicalDFBCount, physicalIndex + 1);
      plan.assignments.push_back({logicalDFB.logicalId, physicalIndex,
                                  logicalDFB.type, logicalDFB.declarations,
                                  logicalDFB.bounded});
    }
    if (plan.physicalDFBCount > kMaxCircularBuffers) {
      std::string message;
      llvm::raw_string_ostream messageStream(message);
      messageStream << "DFB allocation needs " << plan.physicalDFBCount
                    << " physical indices but hardware supports at most "
                    << kMaxCircularBuffers;
      errorOperation = moduleOp;
      errorMessage = messageStream.str();
      return;
    }
  } else {
    FailureOr<DistinctUserAllocation> allocation =
        computeDistinctUserAllocation(moduleOp, liveness, coloring,
                                      analysisFailure);
    if (failed(allocation)) {
      errorOperation = analysisFailure.operation;
      errorMessage = std::move(analysisFailure.message);
      return;
    }
    plan.assignments = std::move(allocation->assignments);
    plan.physicalDFBCount = allocation->physicalDFBCount;
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
    moduleOp->walk([&](func::FuncOp kernel) {
      if (kernel->hasAttr(kBaseCTAIndexAttrName)) {
        plan.kernelBaseIndices.push_back({kernel, plan.physicalDFBCount});
      }
    });
  }
}

} // namespace mlir::tt::ttl
