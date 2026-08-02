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

#include "DFBPhysicalAllocationPlan.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-finalize-dfb-indices"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLFINALIZEDFBINDICES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

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
    descriptorAttributes.push_back(
        DictionaryAttr::get(context, entryAttributes));
  }
  moduleOp->setAttr(kDFBAllocationsAttrName,
                    ArrayAttr::get(context, descriptorAttributes));
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
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    DFBPhysicalAllocationPlanner allocationPlanner(moduleOp, reuseUserDFBs,
                                                   getAnalysisManager());
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
  }
};

} // namespace

} // namespace mlir::tt::ttl
