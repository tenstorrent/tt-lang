// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Finalize DFB Indices
//===----------------------------------------------------------------------===//
//
// Plans module-wide physical indices for compiler-created DFBs after DFB
// creation and synchronization. Pass-order, lifecycle, and capacity checks
// complete before the plan updates declarations, kernel attributes, or runtime
// metadata.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LiveIntervalUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <functional>

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFINALIZEDFBINDICES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static bool isDerivedDFBIndexAttribute(StringRef attributeName) {
  return attributeName == kUnpackToDestFp32AttrName ||
         attributeName.starts_with(kCBIndexAttrPrefix) ||
         attributeName == kBcastOutputCBIndexAttrName ||
         attributeName == kReduceOutputCBIndexAttrName ||
         attributeName == kTransposeOutputCBIndexAttrName;
}

/// Verifies that no pass has copied provisional compiler DFB indices.
///
/// The listed attributes are assumed to contain direct copies of `cb_index`.
/// Rejecting them before allocation prevents finalization from changing a DFB
/// declaration while leaving a stale copy elsewhere. Matching by attribute
/// name is independent of the containing operation or region structure. The
/// predicate must include every attribute introduced by a pass that copies a
/// DFB index.
static LogicalResult verifyFinalizationPrecedesIndexCopies(ModuleOp moduleOp) {
  WalkResult walkResult = moduleOp->walk([&](Operation *operation) {
    for (NamedAttribute attribute : operation->getAttrs()) {
      StringRef attributeName = attribute.getName().getValue();
      if (!isDerivedDFBIndexAttribute(attributeName)) {
        continue;
      }
      operation->emitOpError()
          << "contains derived DFB-index attribute '" << attributeName
          << "' before DFB index finalization; run ttl-finalize-dfb-indices "
             "before ttl-set-compute-kernel-config and "
             "ttl-annotate-cb-associations";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
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

/// Rejects compiler-created DFB declarations missing a lifecycle operation.
///
/// Finalization may span any number of transactions. Auto-sync is assumed to
/// have balanced every acquire with its corresponding release; this check
/// diagnoses declarations with only part of the required producer/consumer
/// protocol, which cannot define the expected compiler-created intermediate.
static LogicalResult verifyCompilerDFBLifecycleOperations(BindCBOp bindOp) {
  bool hasReserve = false;
  bool hasPush = false;
  bool hasWait = false;
  bool hasPop = false;
  for (OpOperand &use : bindOp.getResult().getUses()) {
    Operation *user = use.getOwner();
    hasReserve |= isa<CBReserveOp>(user);
    hasPush |= isa<CBPushOp>(user);
    hasWait |= isa<CBWaitOp>(user);
    hasPop |= isa<CBPopOp>(user);
  }

  if (!hasReserve && !hasPush && !hasWait && !hasPop) {
    return success();
  }
  const std::pair<bool, StringLiteral> requiredOperations[] = {
      {hasReserve, "ttl.cb_reserve"},
      {hasPush, "ttl.cb_push"},
      {hasWait, "ttl.cb_wait"},
      {hasPop, "ttl.cb_pop"},
  };
  for (auto [present, operationName] : requiredOperations) {
    if (!present) {
      return bindOp.emitOpError()
             << "compiler-allocated DFB has a partial lifecycle: missing "
             << operationName;
    }
  }
  return success();
}

/// One compiler-created DFB's final physical index.
struct DFBIndexAssignment {
  BindCBOp declaration;
  int32_t physicalIndex;
};

/// One runtime metadata entry for a physical compiler-created DFB.
struct CompilerDFBMetadataEntry {
  int32_t physicalIndex;
  BindCBOp declaration;
};

/// Complete compiler-created DFB allocation validated before IR mutation.
struct CompilerDFBAllocationPlan {
  SmallVector<DFBIndexAssignment> assignments;
  SmallVector<CompilerDFBMetadataEntry> metadata;
  int32_t physicalDFBCount = 0;
  int32_t compilerSlotCount = 0;
};

/// Plans physical indices for one kernel without modifying its declarations.
///
/// Compiler-created DFB declarations are assumed to be in the kernel body.
/// Nested acquires and releases are projected to their body-block ancestor, so
/// lifetimes within one top-level operation conservatively overlap. An unused
/// or unreleased DFB remains live through the end of the kernel. Auto-sync is
/// assumed to place the last pop after every use of the waited tensor, and
/// compiler-created intermediates have no later direct DFB access. These rules
/// may reduce reuse but cannot assign one index to potentially live DFBs. DFBs
/// share an index only when their half-open intervals do not overlap and their
/// exact types match. Returns the number of physical indices used by the
/// kernel.
static int32_t
planPhysicalDFBIndices(func::FuncOp kernel, ArrayRef<BindCBOp> dfbOps,
                       int32_t firstPhysicalIndex,
                       SmallVectorImpl<DFBIndexAssignment> &assignments) {
  Block &body = kernel.getBody().front();

  DenseMap<Operation *, int64_t> operationIndices;
  int64_t nextOperationIndex = 0;
  for (Operation &operation : body) {
    operationIndices[&operation] = nextOperationIndex++;
  }
  int64_t kernelEndIndex = nextOperationIndex;

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
           "compiler-allocated BindCBOp must be in kernel body block");

    Value dfb = bindOp.getResult();
    int64_t start = kernelEndIndex;
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
        // Include the pop in the half-open interval. Acquires and pops
        // projected to the same kernel-body operation must overlap.
        end = std::max(end, useIndex + 1);
      }
    }

    if (!hasAcquire) {
      start = getOperationIndex(bindOp);
    }
    if (end <= start) {
      end = kernelEndIndex;
    }

    SmallVector<ValueLiveInterval> &intervals = typeToIntervals[dfb.getType()];
    int64_t ordinal = static_cast<int64_t>(intervals.size());
    intervals.push_back({start, end, dfb, ordinal});
    valueToDeclaration[dfb] = bindOp;
  }

  DenseMap<Operation *, int32_t> plannedIndices;
  int32_t nextSlotOffset = 0;
  for (SmallVector<ValueLiveInterval> &intervals :
       llvm::make_second_range(typeToIntervals)) {
    SmallVector<SmallVector<ValueLiveInterval>> colorUsers =
        assignGreedyIntervalColors<ValueLiveInterval>(
            intervals, std::less<ValueLiveInterval>(),
            [](const ValueLiveInterval &lhs, const ValueLiveInterval &rhs) {
              return intervalsOverlap(lhs, rhs);
            });

    for (auto indexedColor : llvm::enumerate(colorUsers)) {
      int32_t color = static_cast<int32_t>(indexedColor.index());
      for (const ValueLiveInterval &interval : indexedColor.value()) {
        BindCBOp declaration = valueToDeclaration.lookup(interval.value);
        assert(declaration && "every live interval must have a declaration");
        plannedIndices[declaration.getOperation()] =
            firstPhysicalIndex + nextSlotOffset + color;

        LLVM_DEBUG({
          llvm::dbgs() << "DFB reuse: [" << interval.start << ", "
                       << interval.end << "] -> slot " << color << "\n";
        });
      }
    }
    nextSlotOffset += static_cast<int32_t>(colorUsers.size());
  }

  for (BindCBOp bindOp : dfbOps) {
    auto plannedIndex = plannedIndices.find(bindOp.getOperation());
    assert(plannedIndex != plannedIndices.end() &&
           "every compiler-created DFB must have a planned index");
    assignments.push_back({bindOp, plannedIndex->second});
  }

  LLVM_DEBUG({
    llvm::dbgs() << "DFB reuse: " << dfbOps.size()
                 << " compiler-allocated DFBs -> " << nextSlotOffset
                 << " physical slot(s)\n";
  });
  return nextSlotOffset;
}

/// Builds all physical assignments and metadata before applying either.
///
/// User-declared indices are assumed to be final. Compiler indices are placed
/// after the greatest user index, so allocation cannot alias user storage. The
/// type partition used by `planPhysicalDFBIndices` guarantees that shared
/// compiler indices have one exact DFB type; an assertion protects that
/// internal invariant while constructing the runtime metadata table.
static FailureOr<CompilerDFBAllocationPlan> buildCompilerDFBAllocationPlan(
    ModuleOp moduleOp,
    const llvm::MapVector<func::FuncOp, SmallVector<BindCBOp>> &kernelToDFBs) {
  CompilerDFBAllocationPlan plan;
  int32_t firstCompilerDFBIndex = getFirstCompilerDFBIndex(moduleOp);
  int32_t nextCompilerDFBIndex = firstCompilerDFBIndex;
  for (const auto &[kernel, dfbOps] : kernelToDFBs) {
    int32_t physicalSlotCount = planPhysicalDFBIndices(
        kernel, dfbOps, nextCompilerDFBIndex, plan.assignments);
    nextCompilerDFBIndex += physicalSlotCount;
    plan.compilerSlotCount += physicalSlotCount;
  }
  plan.physicalDFBCount = nextCompilerDFBIndex;

  if (plan.physicalDFBCount > kMaxCircularBuffers) {
    moduleOp.emitError()
        << "need " << plan.physicalDFBCount
        << " DFB indices but hardware supports at most " << kMaxCircularBuffers
        << " (" << plan.compilerSlotCount
        << " compiler-allocated after reuse); reduce the number of "
           "user-declared dataflow buffers or split the computation into "
           "multiple kernels";
    return failure();
  }

  DenseMap<int32_t, BindCBOp> uniqueByIndex;
  for (const DFBIndexAssignment &assignment : plan.assignments) {
    BindCBOp declaration = assignment.declaration;
    auto [existingAssignment, inserted] =
        uniqueByIndex.try_emplace(assignment.physicalIndex, declaration);
    if (!inserted) {
      assert(existingAssignment->second.getResult().getType() ==
                 declaration.getResult().getType() &&
             "shared compiler DFB index must have one exact type");
    }
  }

  SmallVector<std::pair<int32_t, BindCBOp>> sortedMetadata(
      uniqueByIndex.begin(), uniqueByIndex.end());
  llvm::sort(sortedMetadata, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (auto [physicalIndex, declaration] : sortedMetadata) {
    plan.metadata.push_back({physicalIndex, declaration});
  }
  return plan;
}

/// Applies a complete plan after every operation that can fail has succeeded.
static void
applyCompilerDFBAllocationPlan(ModuleOp moduleOp, OpBuilder &builder,
                               const CompilerDFBAllocationPlan &plan) {
  MLIRContext *context = moduleOp.getContext();
  for (const DFBIndexAssignment &assignment : plan.assignments) {
    BindCBOp declaration = assignment.declaration;
    declaration.setCbIndexAttr(
        IntegerAttr::get(IndexType::get(context), assignment.physicalIndex));
  }

  if (plan.physicalDFBCount <= 0) {
    return;
  }
  moduleOp->walk([&](func::FuncOp kernel) {
    if (kernel->hasAttr(kBaseCTAIndexAttrName)) {
      kernel->setAttr(kBaseCTAIndexAttrName,
                      builder.getI32IntegerAttr(plan.physicalDFBCount));
    }
  });

  if (plan.metadata.empty()) {
    return;
  }
  SmallVector<Attribute> metadataAttributes;
  for (const CompilerDFBMetadataEntry &metadata : plan.metadata) {
    BindCBOp declaration = metadata.declaration;
    auto dfbType = cast<CircularBufferType>(declaration.getResult().getType());
    SmallVector<NamedAttribute> entryAttributes;
    entryAttributes.push_back(builder.getNamedAttr(
        "dfb_index", builder.getI32IntegerAttr(metadata.physicalIndex)));
    entryAttributes.push_back(builder.getNamedAttr(
        "num_tiles", builder.getI32IntegerAttr(
                         static_cast<int32_t>(dfbType.getElementsPerBlock()))));
    entryAttributes.push_back(builder.getNamedAttr(
        "element_type", TypeAttr::get(dfbType.getElementType())));
    entryAttributes.push_back(builder.getNamedAttr(
        "block_count", builder.getI32IntegerAttr(
                           static_cast<int32_t>(dfbType.getBlockCount()))));
    metadataAttributes.push_back(DictionaryAttr::get(context, entryAttributes));
  }
  moduleOp->setAttr(kCompilerAllocatedDFBsAttrName,
                    ArrayAttr::get(context, metadataAttributes));
}

struct TTLFinalizeDFBIndicesPass
    : public impl::TTLFinalizeDFBIndicesBase<TTLFinalizeDFBIndicesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    llvm::MapVector<func::FuncOp, SmallVector<BindCBOp>> kernelToDFBs;
    moduleOp->walk([&](BindCBOp bindOp) {
      if (bindOp->hasAttr(kCompilerAllocatedAttrName)) {
        auto kernel = bindOp->getParentOfType<func::FuncOp>();
        kernelToDFBs[kernel].push_back(bindOp);
      }
    });

    if (!kernelToDFBs.empty() &&
        failed(verifyFinalizationPrecedesIndexCopies(moduleOp))) {
      signalPassFailure();
      return;
    }

    for (ArrayRef<BindCBOp> dfbOps : llvm::make_second_range(kernelToDFBs)) {
      for (BindCBOp bindOp : dfbOps) {
        if (failed(verifyCompilerDFBLifecycleOperations(bindOp))) {
          signalPassFailure();
          return;
        }
      }
    }

    FailureOr<CompilerDFBAllocationPlan> plan =
        buildCompilerDFBAllocationPlan(moduleOp, kernelToDFBs);
    if (failed(plan)) {
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Total DFB count: " << plan->physicalDFBCount << "\n");
    applyCompilerDFBAllocationPlan(moduleOp, builder, *plan);
  }
};

} // namespace

} // namespace mlir::tt::ttl
