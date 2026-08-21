// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Finalize DFB Indices
//===----------------------------------------------------------------------===//
//
// Module-level pass that applies a validated physical DFB allocation plan.
//
//===----------------------------------------------------------------------===//

#include "DFBAllocationLimits.h"
#include "DFBPhysicalAllocationPlan.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/ComputeKernelConfigAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFINALIZEDFBINDICES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static FailureOr<SmallVector<DFBStaticConfigurationConflict>>
collectStaticConfigurationConflicts(
    ModuleOp moduleOp,
    const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis) {
  LaunchNodeDomainState launchDomains;
  launchDomains.initialize(moduleOp);
  SmallVector<DFBStaticConfigurationConflict> conflicts;
  for (func::FuncOp function : moduleOp.getOps<func::FuncOp>()) {
    if (getKernelThreadType(function) != ttkernel::ThreadType::Compute) {
      continue;
    }
    FailureOr<std::unique_ptr<KernelTargetEnvironment>> target =
        KernelTargetEnvironment::get(function);
    if (failed(target)) {
      return failure();
    }
    FailureOr<KernelRequirements> requirements =
        collectKernelRequirements(function, launchDomains);
    if (failed(requirements)) {
      return failure();
    }
    for (const DFBConfigurationAliasConflict &conflict :
         collectDFBConfigurationAliasConflicts(**target, *requirements)) {
      FailureOr<int64_t> lhsLogicalId =
          logicalIdentityAnalysis.getLogicalId(conflict.lhsDFB);
      FailureOr<int64_t> rhsLogicalId =
          logicalIdentityAnalysis.getLogicalId(conflict.rhsDFB);
      assert(succeeded(lhsLogicalId) && succeeded(rhsLogicalId) &&
             "configuration DFB uses must resolve to logical identities");
      if (*lhsLogicalId != *rhsLogicalId) {
        conflicts.push_back({*lhsLogicalId, *rhsLogicalId,
                             conflict.lhsOperation, conflict.rhsOperation});
      }
    }
  }
  return conflicts;
}

/// Materializes decisions already validated by physical allocation analysis.
static void
applyPhysicalAllocationPlan(ModuleOp moduleOp, OpBuilder &builder,
                            const DFBPhysicalAllocationPlan &allocationPlan) {
  MLIRContext *context = moduleOp.getContext();
  for (const DFBPhysicalIndexAssignment &assignment :
       allocationPlan.getAssignments()) {
    for (BindCBOp declaration : assignment.declarations) {
      declaration.setDfbIdAttr(
          IntegerAttr::get(IndexType::get(context), assignment.logicalId));
      declaration.setCbIndexAttr(
          IntegerAttr::get(IndexType::get(context), assignment.physicalIndex));
    }
    LLVM_DEBUG({
      llvm::dbgs() << "DFB assignment: logical DFB " << assignment.logicalId
                   << " -> physical index " << assignment.physicalIndex
                   << (assignment.bounded ? " (bounded)\n" : " (unbounded)\n");
    });
  }

  for (const DFBKernelBaseIndexAssignment &baseIndex :
       allocationPlan.getKernelBaseIndices()) {
    baseIndex.kernel->setAttr(kBaseCTAIndexAttrName,
                              builder.getI32IntegerAttr(baseIndex.baseIndex));
  }

  SmallVector<Attribute> descriptorAttributes;
  for (const DFBPhysicalAllocationDescriptor &descriptor :
       allocationPlan.getDescriptors()) {
    SmallVector<NamedAttribute> entryAttributes;
    entryAttributes.push_back(builder.getNamedAttr(
        "dfb_index", builder.getI32IntegerAttr(descriptor.physicalIndex)));
    entryAttributes.push_back(builder.getNamedAttr(
        "num_tiles", builder.getI32IntegerAttr(descriptor.numTiles)));
    entryAttributes.push_back(builder.getNamedAttr(
        "element_type", TypeAttr::get(descriptor.elementType)));
    entryAttributes.push_back(builder.getNamedAttr(
        "page_size", builder.getI32IntegerAttr(descriptor.pageSize)));
    entryAttributes.push_back(builder.getNamedAttr(
        "block_count", builder.getI32IntegerAttr(descriptor.blockCount)));
    SmallVector<Attribute> storageSegmentAttributes;
    for (const DFBPhysicalStorageSegment &segment :
         descriptor.storageSegments) {
      SmallVector<Attribute> nodeAttributes;
      for (LaunchNodeCoord node : segment.launchDomain.nodes) {
        nodeAttributes.push_back(
            builder.getArrayAttr({builder.getI64IntegerAttr(node.x),
                                  builder.getI64IntegerAttr(node.y)}));
      }
      SmallVector<NamedAttribute> segmentAttributes;
      segmentAttributes.push_back(
          builder.getNamedAttr("nodes", builder.getArrayAttr(nodeAttributes)));
      if (segment.tensorBacking) {
        segmentAttributes.push_back(
            builder.getNamedAttr("tensor_backing", segment.tensorBacking));
      }
      storageSegmentAttributes.push_back(
          builder.getDictionaryAttr(segmentAttributes));
    }
    if (!storageSegmentAttributes.empty()) {
      entryAttributes.push_back(builder.getNamedAttr(
          "storage_segments", builder.getArrayAttr(storageSegmentAttributes)));
    }
    descriptorAttributes.push_back(
        DictionaryAttr::get(context, entryAttributes));
  }
  moduleOp->setAttr(kDFBAllocationsAttrName,
                    ArrayAttr::get(context, descriptorAttributes));
}

struct TTLFinalizeDFBIndicesPass
    : public impl::TTLFinalizeDFBIndicesBase<TTLFinalizeDFBIndicesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (failed(validateSynchronizedDFBResetTarget(moduleOp))) {
      signalPassFailure();
      return;
    }
    const DFBLogicalIdentityAnalysis &logicalIdentityAnalysis =
        getAnalysis<DFBLogicalIdentityAnalysis>();
    if (!logicalIdentityAnalysis.succeeded()) {
      Operation *errorOperation = logicalIdentityAnalysis.getErrorOperation();
      if (!errorOperation) {
        errorOperation = moduleOp.getOperation();
      }
      errorOperation->emitOpError()
          << logicalIdentityAnalysis.getErrorMessage();
      signalPassFailure();
      return;
    }
    FailureOr<SmallVector<DFBStaticConfigurationConflict>>
        staticConfigurationConflicts = collectStaticConfigurationConflicts(
            moduleOp, logicalIdentityAnalysis);
    if (failed(staticConfigurationConflicts)) {
      signalPassFailure();
      return;
    }
    DFBPhysicalAllocationPlanner allocationPlanner(
        moduleOp, reuseUserDFBs, exactColoringSearchStateLimit,
        l1BudgetOverride == 0 ? std::nullopt
                              : std::optional<uint64_t>(l1BudgetOverride),
        *staticConfigurationConflicts, getAnalysisManager());
    if (!allocationPlanner.succeeded()) {
      Operation *errorOperation = allocationPlanner.getErrorOperation();
      if (!errorOperation) {
        errorOperation = moduleOp.getOperation();
      }
      errorOperation->emitOpError() << allocationPlanner.getErrorMessage();
      signalPassFailure();
      return;
    }

    const DFBPhysicalAllocationPlan &allocationPlan =
        allocationPlanner.getPlan();
    LLVM_DEBUG(llvm::dbgs() << "Total DFB count: "
                            << allocationPlan.getPhysicalDFBCount() << "\n");

    OpBuilder builder(moduleOp.getContext());
    applyPhysicalAllocationPlan(moduleOp, builder, allocationPlan);
    moduleOp->removeAttr(kPipeConservativeL1BytesAttrName);
  }
};

} // namespace

} // namespace mlir::tt::ttl
