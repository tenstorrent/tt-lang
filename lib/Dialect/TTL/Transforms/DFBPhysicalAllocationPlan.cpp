// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBPhysicalAllocationPlan.h"

#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "DFBLogicalIdentityAnalysis.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/Transforms/InterferenceGraphColoring.h"
#include "ttlang/Dialect/TTL/Transforms/LiveIntervalUtils.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>

namespace mlir::tt::ttl {

namespace {

/// First validation failure produced while constructing an allocation plan.
///
/// Planning continues through MLIR walks, so later failures must not replace
/// the operation that first violated the allocation contract.
struct AnalysisFailure {
  Operation *operation = nullptr;
  std::string message;

  void set(Operation *failureOperation, std::string failureMessage) {
    if (!message.empty()) {
      return;
    }
    operation = failureOperation;
    message = std::move(failureMessage);
  }
};

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

/// Colors the concurrent-lifetime interference graph.
static FailureOr<SmallVector<int32_t>>
computeConcurrentAssignments(
    ModuleOp moduleOp,
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    const InterferenceGraphColoring &coloring,
    AnalysisFailure &analysisFailure) {
  ArrayRef<DFBLogicalLifecycle> logicalDFBs =
      liveness.getLogicalDFBLifecycles();
  SmallVector<unsigned> logicalIndices =
      llvm::to_vector(llvm::seq<unsigned>(0, logicalDFBs.size()));
  llvm::sort(logicalIndices, [&](unsigned lhsIndex, unsigned rhsIndex) {
    const DFBLogicalLifecycle &lhs = logicalDFBs[lhsIndex];
    const DFBLogicalLifecycle &rhs = logicalDFBs[rhsIndex];
    return lhs.logicalId != rhs.logicalId ? lhs.logicalId < rhs.logicalId
                                          : lhsIndex < rhsIndex;
  });

  InterferenceGraph interferenceGraph(logicalDFBs.size());
  for (unsigned lhsIndex = 0; lhsIndex < logicalDFBs.size(); ++lhsIndex) {
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < logicalDFBs.size();
         ++rhsIndex) {
      if (logicalDFBsConflict(logicalDFBs[lhsIndex], logicalDFBs[rhsIndex],
                              lhsIndex, rhsIndex, liveness)) {
        interferenceGraph.addInterference(lhsIndex, rhsIndex);
      }
    }
  }

  SmallVector<unsigned> colors =
      coloring.color(interferenceGraph, logicalIndices);
  assert(colors.size() == logicalDFBs.size() &&
         "coloring must assign every logical DFB");

  unsigned colorCount = 0;
  for (unsigned color : colors) {
    assert(color < logicalDFBs.size() &&
           "a dense coloring cannot use more colors than vertices");
    colorCount = std::max(colorCount, color + 1);
  }
  llvm::BitVector usedColors(colorCount);
  for (unsigned color : colors) {
    usedColors.set(color);
  }
  assert(usedColors.all() && "coloring must use dense zero-based colors");

  if (colorCount > kMaxCircularBuffers) {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    messageStream << "DFB allocation needs " << colorCount
                  << " physical indices but hardware supports at most "
                  << kMaxCircularBuffers;
    analysisFailure.set(moduleOp, messageStream.str());
    return failure();
  }

  SmallVector<int32_t> assignments;
  assignments.reserve(colors.size());
  for (unsigned color : colors) {
    assignments.push_back(static_cast<int32_t>(color));
  }

  for (unsigned lhsIndex = 0; lhsIndex < logicalDFBs.size(); ++lhsIndex) {
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < logicalDFBs.size();
         ++rhsIndex) {
      if (assignments[lhsIndex] != assignments[rhsIndex]) {
        continue;
      }
      if (interferenceGraph.interferes(lhsIndex, rhsIndex)) {
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "DFB allocator assigned interfering logical DFBs "
                      << logicalDFBs[lhsIndex].logicalId << " and "
                      << logicalDFBs[rhsIndex].logicalId
                      << " to physical index " << assignments[lhsIndex];
        analysisFailure.set(moduleOp, messageStream.str());
        return failure();
      }
    }
  }

  return assignments;
}

static int32_t getFirstCompilerDFBIndex(ModuleOp moduleOp) {
  int32_t maxUserIndex = -1;
  moduleOp->walk([&](BindCBOp bindOp) {
    if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
      return;
    }
    maxUserIndex = std::max(
        maxUserIndex, static_cast<int32_t>(bindOp.getCbIndex().getSExtValue()));
  });
  return maxUserIndex + 1;
}

/// Plans kernel-local linear-scan allocation for compiler-created DFBs.
static int32_t planCompilerDFBIndices(
    func::FuncOp kernel, ArrayRef<BindCBOp> dfbOps, int32_t firstPhysicalIndex,
    DenseMap<Operation *, int32_t> &physicalIndices) {
  Block &body = kernel.getBody().front();
  DenseMap<Operation *, int64_t> operationIndices;
  int64_t nextOperationIndex = 0;
  for (Operation &operation : body) {
    operationIndices[&operation] = nextOperationIndex++;
  }
  int64_t lastOperationIndex = nextOperationIndex - 1;

  auto getOperationIndex = [&](Operation *operation) -> int64_t {
    auto operationIndex = operationIndices.find(operation);
    assert(operationIndex != operationIndices.end() &&
           "operation must belong to the kernel body");
    return operationIndex->second;
  };
  auto getBodyIndex = [&](Operation *operation) -> int64_t {
    if (operation->getBlock() == &body) {
      return getOperationIndex(operation);
    }
    Operation *ancestor = body.findAncestorOpInBlock(*operation);
    assert(ancestor && "operation must be reachable from kernel body");
    return getOperationIndex(ancestor);
  };

  llvm::MapVector<Type, SmallVector<ValueLiveInterval>> typeToIntervals;
  DenseMap<Value, BindCBOp> valueToDeclaration;
  for (BindCBOp bindOp : dfbOps) {
    assert(bindOp->getBlock() == &body &&
           "compiler-created DFB declaration must be in the kernel body");

    Value dfb = bindOp.getResult();
    int64_t start = lastOperationIndex;
    int64_t end = getOperationIndex(bindOp);
    bool hasAcquire = false;
    for (OpOperand &use : dfb.getUses()) {
      Operation *user = use.getOwner();
      int64_t useIndex = getBodyIndex(user);
      if (isa<CBReserveOp, CBWaitOp>(user)) {
        start = std::min(start, useIndex);
        hasAcquire = true;
      }
      if (isa<CBPopOp>(user)) {
        end = std::max(end, useIndex);
      }
    }
    if (!hasAcquire) {
      start = getOperationIndex(bindOp);
    }
    if (end <= start) {
      end = lastOperationIndex;
    }

    SmallVector<ValueLiveInterval> &intervals = typeToIntervals[dfb.getType()];
    intervals.push_back(
        {start, end, dfb, static_cast<int64_t>(intervals.size())});
    valueToDeclaration[dfb] = bindOp;
  }

  int32_t nextSlotOffset = 0;
  for (auto &typeAndIntervals : typeToIntervals) {
    SmallVector<SmallVector<ValueLiveInterval>> colorUsers =
        assignGreedyIntervalColors<ValueLiveInterval>(
            typeAndIntervals.second, std::less<ValueLiveInterval>(),
            [](const ValueLiveInterval &lhs, const ValueLiveInterval &rhs) {
              return intervalsOverlap(lhs, rhs);
            });
    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      int32_t color = static_cast<int32_t>(indexedColor.index());
      for (const ValueLiveInterval &interval : indexedColor.value()) {
        auto declaration = valueToDeclaration.find(interval.value);
        assert(declaration != valueToDeclaration.end() &&
               "every interval must have a DFB declaration");
        physicalIndices[declaration->second.getOperation()] =
            firstPhysicalIndex + nextSlotOffset + color;
      }
    }
    nextSlotOffset += static_cast<int32_t>(colorUsers.size());
  }
  return nextSlotOffset;
}

/// Compiler-only assignments and the resulting module-wide index count.
struct CompilerOnlyAllocation {
  SmallVector<DFBPhysicalIndexAssignment> assignments;
  int32_t physicalDFBCount = 0;
};

static FailureOr<CompilerOnlyAllocation> computeCompilerOnlyAllocation(
    ModuleOp moduleOp,
    const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis,
    AnalysisFailure &analysisFailure) {
  llvm::MapVector<func::FuncOp, SmallVector<BindCBOp>> kernelDFBs;
  moduleOp->walk([&](BindCBOp bindOp) {
    if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
      func::FuncOp kernel = bindOp->getParentOfType<func::FuncOp>();
      assert(kernel && "compiler-created DFB must be nested in a kernel");
      kernelDFBs[kernel].push_back(bindOp);
    }
  });

  DenseMap<Operation *, int32_t> physicalIndices;
  int32_t nextCompilerIndex = getFirstCompilerDFBIndex(moduleOp);
  int32_t compilerSlotCount = 0;
  for (auto &[kernel, dfbOps] : kernelDFBs) {
    int32_t slotCount = planCompilerDFBIndices(
        kernel, dfbOps, nextCompilerIndex, physicalIndices);
    nextCompilerIndex += slotCount;
    compilerSlotCount += slotCount;
  }

  llvm::MapVector<int64_t, unsigned> logicalIdToAssignment;
  CompilerOnlyAllocation allocation;
  for (const DFBLogicalIdentityAssignment &identity :
       logicalIdentityAnalysis.getAssignments()) {
    BindCBOp declaration = identity.declaration;
    int32_t physicalIndex;
    if (declaration->hasAttr(kCompilerAllocatedAttrName)) {
      auto physicalIndexIt =
          physicalIndices.find(declaration.getOperation());
      assert(physicalIndexIt != physicalIndices.end() &&
             "every compiler-created DFB must have a physical index");
      physicalIndex = physicalIndexIt->second;
    } else {
      physicalIndex =
          static_cast<int32_t>(declaration.getCbIndex().getSExtValue());
    }

    auto [assignmentIt, inserted] = logicalIdToAssignment.insert(
        {identity.logicalId, allocation.assignments.size()});
    if (inserted) {
      allocation.assignments.push_back(
          {identity.logicalId, physicalIndex,
           declaration.getResult().getType(),
           {declaration}, false});
    } else {
      DFBPhysicalIndexAssignment &assignment =
          allocation.assignments[assignmentIt->second];
      if (assignment.physicalIndex != physicalIndex) {
        std::string message;
        llvm::raw_string_ostream messageStream(message);
        messageStream << "logical DFB " << identity.logicalId
                      << " has inconsistent physical indices "
                      << assignment.physicalIndex << " and " << physicalIndex;
        analysisFailure.set(declaration, messageStream.str());
        return failure();
      }
      assignment.declarations.push_back(declaration);
    }
    allocation.physicalDFBCount =
        std::max(allocation.physicalDFBCount, physicalIndex + 1);
  }
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

static FailureOr<SmallVector<DFBPhysicalAllocationDescriptor>>
buildDescriptors(ArrayRef<DFBPhysicalIndexAssignment> assignments,
                 AnalysisFailure &analysisFailure) {
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
  for (auto [physicalIndex, assignment] : sorted) {
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
    Operation *operation, bool reuseUserDFBs,
    AnalysisManager analysisManager,
    const InterferenceGraphColoring &coloring) {
  ModuleOp moduleOp = cast<ModuleOp>(operation);
  const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis =
      analysisManager.getAnalysis<DFBLogicalIdentityAnalysis>();
  if (!logicalIdentityAnalysis.succeeded()) {
    errorOperation = logicalIdentityAnalysis.getErrorOperation();
    errorMessage = logicalIdentityAnalysis.getErrorMessage().str();
    return;
  }

  AnalysisFailure analysisFailure;
  if (reuseUserDFBs) {
    const DFBConcurrentKernelLivenessAnalysis &liveness =
        analysisManager.getAnalysis<DFBConcurrentKernelLivenessAnalysis>();
    if (!liveness.succeeded()) {
      errorOperation = liveness.getErrorOperation();
      errorMessage = liveness.getErrorMessage().str();
      return;
    }

    FailureOr<SmallVector<int32_t>> physicalIndices =
        computeConcurrentAssignments(moduleOp, liveness, coloring,
                                     analysisFailure);
    if (failed(physicalIndices)) {
      errorOperation = analysisFailure.operation;
      errorMessage = std::move(analysisFailure.message);
      return;
    }
    ArrayRef<DFBLogicalLifecycle> logicalDFBs =
        liveness.getLogicalDFBLifecycles();
    for (auto indexedLogicalDFB : llvm::enumerate(logicalDFBs)) {
      const DFBLogicalLifecycle &logicalDFB = indexedLogicalDFB.value();
      int32_t physicalIndex = (*physicalIndices)[indexedLogicalDFB.index()];
      plan.physicalDFBCount =
          std::max(plan.physicalDFBCount, physicalIndex + 1);
      plan.assignments.push_back(
          {logicalDFB.logicalId, physicalIndex, logicalDFB.type,
           logicalDFB.declarations, logicalDFB.bounded});
    }
  } else {
    FailureOr<CompilerOnlyAllocation> allocation =
        computeCompilerOnlyAllocation(moduleOp, logicalIdentityAnalysis,
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
        plan.kernelBaseIndices.push_back(
            {kernel, plan.physicalDFBCount});
      }
    });
  }
}

} // namespace mlir::tt::ttl
