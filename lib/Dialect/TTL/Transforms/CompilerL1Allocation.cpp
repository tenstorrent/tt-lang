// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "CompilerL1Allocation.h"
#include "CompilerL1Allocator.h"
#include "DFBAllocationLimits.h"
#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "DFBPhysicalAllocationPlan.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <limits>

namespace mlir::tt::ttl {

namespace {
struct L1Region {
  int64_t logicalId;
  CircularBufferType type;
  uint64_t pages;
  uint64_t pageBytes;
  uint64_t allocationBytes;
  uint64_t stateOffset = 0;
  SmallVector<BindCBOp> declarations;
};

struct L1AllocationPlan {
  SmallVector<L1Region> regions;
  CompilerL1AllocationSolution solution;
};

static FailureOr<L1AllocationPlan>
planRegions(ModuleOp module, const DFBLogicalIdentityAnalysis &identities,
            uint64_t budget, bool reuseStorage,
            const CompilerL1Allocator &allocator,
            const DFBConcurrentKernelLivenessAnalysis &liveness) {
  std::string targetFailure;
  FailureOr<std::optional<ttcore::Arch>> targetArch =
      resolveTargetArch(module, targetFailure);
  if (failed(targetArch)) {
    module.emitOpError() << targetFailure;
    return failure();
  }
  if (*targetArch && !supportsCompilerSRAM(**targetArch)) {
    module.emitOpError()
        << "compiler-sram supports only Wormhole B0 and Blackhole; selected "
           "target is "
        << ttcore::ArchAttr::get(module.getContext(), **targetArch);
    return failure();
  }
  uint64_t alignment = *targetArch
                           ? getTargetL1AllocationQuantumBytes(**targetArch)
                           : getConservativeL1AllocationQuantumBytes();
  llvm::MapVector<int64_t, L1Region> regions;
  for (const auto &assignment : identities.getAssignments()) {
    BindCBOp declaration = assignment.declaration;
    auto type = cast<CircularBufferType>(declaration.getResult().getType());
    if (declaration.getTensorBackingAttr() || assignment.allocationGroup) {
      declaration.emitOpError("compiler-sram requires independently owned "
                              "storage without tensor backing or allocation "
                              "groups");
      return failure();
    }
    auto found = regions.find(assignment.logicalId);
    if (found != regions.end()) {
      assert(found->second.type == type &&
             "logical identity analysis validates declaration types");
      found->second.declarations.push_back(declaration);
      continue;
    }
    FailureOr<uint64_t> pages = getDFBPagesPerBlock(type);
    FailureOr<uint64_t> pageBytes = getDFBPageSizeBytes(type);
    std::string failureReason;
    FailureOr<uint64_t> payloadBytes =
        getDFBAllocationSizeBytes(type, failureReason);
    if (failed(payloadBytes)) {
      declaration.emitOpError() << "compiler-sram " << failureReason;
      return failure();
    }
    if (failed(pages) || failed(pageBytes) ||
        *payloadBytes > std::numeric_limits<uint32_t>::max() ||
        *pageBytes > std::numeric_limits<int32_t>::max() ||
        type.getBlockCount() > std::numeric_limits<int32_t>::max() ||
        *pages > std::numeric_limits<int32_t>::max()) {
      declaration.emitOpError(
          "compiler-sram storage size is not representable");
      return failure();
    }
    FailureOr<uint64_t> allocationBytes =
        getL1AllocationSizeBytes(module, *payloadBytes);
    if (failed(allocationBytes)) {
      declaration.emitOpError(
          "compiler-sram target-aligned storage size is not representable");
      return failure();
    }
    regions.insert({assignment.logicalId,
                    {assignment.logicalId,
                     type,
                     *pages,
                     *pageBytes,
                     *allocationBytes,
                     0,
                     {declaration}}});
  }
  SmallVector<L1Region> plan;
  for (auto &entry : regions) {
    plan.push_back(std::move(entry.second));
  }
  const auto conflicts = DFBPhysicalConflictModel::buildStorage(
      liveness, DFBStorageConflictMode::CompilerManaged);
  DenseMap<int64_t, unsigned> lifecycleIndices;
  for (auto [lifecycleIndex, lifecycle] :
       llvm::enumerate(liveness.getLogicalDFBLifecycles())) {
    lifecycleIndices[lifecycle.logicalId] = lifecycleIndex;
  }
  // Payload liveness does not prove that counter state can change ownership.
  std::optional<uint64_t> unalignedControlBytes = llvm::checkedMulUnsigned(
      static_cast<uint64_t>(plan.size()), kCompilerSRAMControlRecordBytes);
  FailureOr<uint64_t> controlBytes =
      unalignedControlBytes
          ? getL1AllocationSizeBytes(module, *unalignedControlBytes)
          : FailureOr<uint64_t>(failure());
  if (failed(controlBytes) || *controlBytes > budget) {
    module.emitOpError(
        "compiler-sram control records exceed the available L1 budget");
    return failure();
  }
  CompilerL1AllocationProblem problem;
  problem.alignmentBytes = alignment;
  problem.payloadBaseOffset = *controlBytes;
  problem.budgetBytes = budget;
  problem.conflicts.assign(plan.size(), llvm::BitVector(plan.size()));
  for (unsigned regionIndex = 0; regionIndex < plan.size(); ++regionIndex) {
    L1Region &region = plan[regionIndex];
    region.stateOffset = regionIndex * kCompilerSRAMControlRecordBytes;
    problem.regionBytes.push_back(region.allocationBytes);
    assert(lifecycleIndices.contains(region.logicalId));
    for (unsigned previousIndex = 0; previousIndex < regionIndex;
         ++previousIndex) {
      assert(lifecycleIndices.contains(plan[previousIndex].logicalId));
      if (reuseStorage &&
          !conflicts.conflicts(
              lifecycleIndices.lookup(region.logicalId),
              lifecycleIndices.lookup(plan[previousIndex].logicalId))) {
        continue;
      }
      problem.conflicts[regionIndex].set(previousIndex);
      problem.conflicts[previousIndex].set(regionIndex);
    }
  }
  SRAMPlacementFailure placementFailure;
  FailureOr<CompilerL1AllocationSolution> solution =
      solveCompilerL1Allocation(allocator, problem, placementFailure);
  if (failed(solution)) {
    auto diagnostic = placementFailure.regionIndex
                          ? plan[*placementFailure.regionIndex]
                                .declarations.front()
                                .emitOpError()
                          : module.emitOpError();
    diagnostic << "compiler-sram " << placementFailure.reason;
    if (placementFailure.kind == SRAMPlacementFailureKind::BudgetExceeded) {
      diagnostic << " (payload, control records, and alignment included); "
                 << allocator.getName()
                 << " placement does not prove infeasibility";
    }
    return failure();
  }
  return L1AllocationPlan{std::move(plan), std::move(*solution)};
}
} // namespace

LogicalResult
allocateCompilerL1(ModuleOp module,
                   const DFBLogicalIdentityAnalysis &identities,
                   uint64_t budgetOverride, bool reuseStorage,
                   const CompilerL1Allocator &allocator,
                   const DFBConcurrentKernelLivenessAnalysis &liveness) {
  auto budget = getUsableDFBL1Bytes(
      module,
      budgetOverride ? std::optional<uint64_t>(budgetOverride) : std::nullopt);
  FailureOr<L1AllocationPlan> maybePlan = planRegions(
      module, identities, budget, reuseStorage, allocator, liveness);
  if (failed(maybePlan)) {
    return failure();
  }
  const L1AllocationPlan &plan = *maybePlan;
  OpBuilder builder(module.getContext());
  SmallVector<Attribute> allocations;
  DenseMap<int64_t, int32_t> allocationIndexByLogicalId;
  for (auto [regionIndex, region] : llvm::enumerate(plan.regions)) {
    uint64_t payloadOffset = plan.solution.offsets[regionIndex];
    allocationIndexByLogicalId.try_emplace(region.logicalId,
                                           static_cast<int32_t>(regionIndex));
    for (BindCBOp declaration : region.declarations) {
      declaration.setDfbIdAttr(builder.getIndexAttr(region.logicalId));
      declaration.setCbIndexAttr(builder.getIndexAttr(regionIndex));
    }
    allocations.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr(kDFBAllocationIndexField,
                             builder.getI32IntegerAttr(regionIndex)),
        builder.getNamedAttr(kDFBAllocationStorageIndexField,
                             builder.getI32IntegerAttr(regionIndex)),
        builder.getNamedAttr(kDFBAllocationNumTilesField,
                             builder.getI32IntegerAttr(region.pages)),
        builder.getNamedAttr(kDFBAllocationPageSizeField,
                             builder.getI32IntegerAttr(region.pageBytes)),
        builder.getNamedAttr(
            kDFBAllocationBlockCountField,
            builder.getI32IntegerAttr(region.type.getBlockCount())),
        builder.getNamedAttr(kDFBAllocationElementTypeField,
                             TypeAttr::get(region.type.getElementType())),
        builder.getNamedAttr(kDFBAllocationStateOffsetField,
                             builder.getI64IntegerAttr(region.stateOffset)),
        builder.getNamedAttr(kDFBAllocationPayloadOffsetField,
                             builder.getI64IntegerAttr(payloadOffset)),
        builder.getNamedAttr(kDFBAllocationBytesField,
                             builder.getI64IntegerAttr(region.allocationBytes)),
    }));
  }
  DenseMap<int64_t, SmallVector<int32_t>> resetsByReconfiguration;
  for (const DFBLogicalLifecycle &lifecycle :
       liveness.getLogicalDFBLifecycles()) {
    auto allocationIt = allocationIndexByLogicalId.find(lifecycle.logicalId);
    assert(allocationIt != allocationIndexByLogicalId.end() &&
           "every logical DFB must have a compiler-sram allocation");
    auto collectTerminalReconfigurations = [&](const DFBPerNodeLifetime &node) {
      for (const DFBLifecycleEpoch &epoch : node.epochs) {
        if (epoch.terminalReconfigurationOrdinal) {
          resetsByReconfiguration[*epoch.terminalReconfigurationOrdinal]
              .push_back(allocationIt->second);
        }
      }
    };
    for (const DFBPerNodeLifetime &node : lifecycle.nodeLifetimes) {
      collectTerminalReconfigurations(node);
    }
    for (const DFBPerNodeLifetime &node : lifecycle.possibleNodeLifetimes) {
      collectTerminalReconfigurations(node);
    }
  }
  SmallVector<Attribute> reconfigurationResets;
  for (int64_t ordinal : liveness.getReconfigurationBoundaryOrdinals()) {
    auto resetIt = resetsByReconfiguration.find(ordinal);
    if (resetIt == resetsByReconfiguration.end()) {
      continue;
    }
    SmallVector<int32_t> &indices = resetIt->second;
    llvm::sort(indices);
    indices.erase(llvm::unique(indices), indices.end());
    reconfigurationResets.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
        builder.getNamedAttr("dfb_indices",
                             builder.getDenseI32ArrayAttr(indices)),
    }));
  }
  assert(reconfigurationResets.size() == resetsByReconfiguration.size() &&
         "every terminal epoch must reference a known reconfiguration");
  if (reconfigurationResets.empty()) {
    module->removeAttr(kCompilerSRAMReconfigurationResetsAttrName);
  } else {
    module->setAttr(kCompilerSRAMReconfigurationResetsAttrName,
                    builder.getArrayAttr(reconfigurationResets));
  }
  module->setAttr(kL1ArenaBytesAttrName,
                  builder.getI64IntegerAttr(plan.solution.arenaBytes));
  module->setAttr(kDFBAllocationsAttrName, builder.getArrayAttr(allocations));
  module->setAttr(kMemoryModelAttrName,
                  builder.getStringAttr(kCompilerSRAMMemoryModel));
  int32_t baseCTAIndex = plan.regions.empty() ? 0 : 1;
  for (func::FuncOp kernel : module.getOps<func::FuncOp>()) {
    if (kernel->hasAttr(kBaseCTAIndexAttrName)) {
      kernel->setAttr(kBaseCTAIndexAttrName,
                      builder.getI32IntegerAttr(baseCTAIndex));
    }
  }
  return success();
}
} // namespace mlir::tt::ttl
