// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "CompilerL1Allocation.h"
#include "DFBAllocationLimits.h"
#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "DFBPhysicalAllocationPlan.h"
#include "mlir/IR/Builders.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Target/TargetInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <tuple>

namespace mlir::tt::ttl {
namespace {
struct L1Region {
  int64_t logicalId;
  CircularBufferType type;
  uint64_t pages;
  uint64_t pageBytes;
  uint64_t allocationBytes;
  uint64_t offset = 0;
  uint64_t stateOffset = 0;
  SmallVector<BindCBOp> declarations;
};

struct L1AllocationPlan {
  SmallVector<L1Region> regions;
  uint64_t arenaBytes;
};

static FailureOr<L1AllocationPlan>
planRegions(ModuleOp module, const DFBLogicalIdentityAnalysis &identities,
            uint64_t budget, bool reuseStorage,
            const DFBConcurrentKernelLivenessAnalysis &liveness) {
  WalkResult supportedStorage = module.walk([](Operation *operation) {
    if (isa<DFBReconfigurationOp, ResetDFBsOp, ResetAllDFBsOp>(operation)) {
      operation->emitOpError(
          "compiler-l1 requires static storage ownership; DFB reset and "
          "reconfiguration are unsupported");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (supportedStorage.wasInterrupted()) {
    return failure();
  }
  std::string targetFailure;
  FailureOr<uint64_t> alignment =
      resolveTargetL1AllocationQuantumBytes(module, targetFailure);
  if (failed(alignment)) {
    module.emitOpError() << targetFailure;
    return failure();
  }
  llvm::MapVector<int64_t, L1Region> regions;
  for (const auto &assignment : identities.getAssignments()) {
    BindCBOp declaration = assignment.declaration;
    auto type = cast<CircularBufferType>(declaration.getResult().getType());
    if (declaration.getTensorBackingAttr() || assignment.allocationGroup) {
      declaration.emitOpError("compiler-l1 does not yet support tensor-backed "
                              "storage or allocation groups");
      return failure();
    }
    auto found = regions.find(assignment.logicalId);
    if (found != regions.end()) {
      if (found->second.type != type) {
        declaration.emitOpError("compiler-l1 requires identical declarations "
                                "for each logical storage region");
        return failure();
      }
      found->second.declarations.push_back(declaration);
      continue;
    }
    FailureOr<uint64_t> pages = getDFBPagesPerBlock(type);
    FailureOr<uint64_t> pageBytes = getDFBPageSizeBytes(type);
    std::string failureReason;
    FailureOr<uint64_t> payloadBytes =
        getDFBAllocationSizeBytes(type, failureReason);
    if (failed(pages) || failed(pageBytes) || failed(payloadBytes) ||
        *payloadBytes > std::numeric_limits<uint32_t>::max() ||
        *pageBytes > std::numeric_limits<int32_t>::max() ||
        type.getBlockCount() > std::numeric_limits<int32_t>::max() ||
        *pages > std::numeric_limits<int32_t>::max()) {
      declaration.emitOpError("compiler-l1 storage size is not representable");
      return failure();
    }
    FailureOr<uint64_t> allocationBytes =
        getL1AllocationSizeBytes(module, *payloadBytes);
    if (failed(allocationBytes)) {
      return failure();
    }
    regions.insert({assignment.logicalId,
                    {assignment.logicalId,
                     type,
                     *pages,
                     *pageBytes,
                     *allocationBytes,
                     0,
                     0,
                     {declaration}}});
  }
  SmallVector<L1Region> plan;
  for (auto &entry : regions) {
    plan.push_back(std::move(entry.second));
  }
  SmallVector<unsigned> placementOrder;
  for (unsigned regionIndex = 0; regionIndex < plan.size(); ++regionIndex) {
    placementOrder.push_back(regionIndex);
  }
  llvm::stable_sort(placementOrder, [&](unsigned lhsIndex, unsigned rhsIndex) {
    return plan[lhsIndex].allocationBytes > plan[rhsIndex].allocationBytes;
  });
  const auto conflicts = DFBPhysicalConflictModel::buildStorage(liveness);
  DenseMap<int64_t, unsigned> lifecycleIndices;
  for (auto [lifecycleIndex, lifecycle] :
       llvm::enumerate(liveness.getLogicalDFBLifecycles())) {
    lifecycleIndices[lifecycle.logicalId] = lifecycleIndex;
  }
  // Control words retain distinct ownership even when payload bytes overlap.
  uint64_t controlBytes = llvm::alignTo(plan.size() * uint64_t{8}, *alignment);
  SmallVector<unsigned> placed;
  for (unsigned regionIndex : placementOrder) {
    L1Region &region = plan[regionIndex];
    region.stateOffset = regionIndex * uint64_t{8};
    SmallVector<unsigned> interfering;
    for (unsigned previousIndex : placed) {
      assert(lifecycleIndices.contains(region.logicalId) &&
             lifecycleIndices.contains(plan[previousIndex].logicalId));
      if (!reuseStorage ||
          conflicts.conflicts(
              lifecycleIndices.lookup(region.logicalId),
              lifecycleIndices.lookup(plan[previousIndex].logicalId))) {
        interfering.push_back(previousIndex);
      }
    }
    llvm::sort(interfering, [&](unsigned lhsIndex, unsigned rhsIndex) {
      return std::tie(plan[lhsIndex].offset, lhsIndex) <
             std::tie(plan[rhsIndex].offset, rhsIndex);
    });
    uint64_t offset = controlBytes;
    for (unsigned previousIndex : interfering) {
      const auto &previous = plan[previousIndex];
      if (offset + region.allocationBytes <= previous.offset) {
        break;
      }
      if (offset < previous.offset + previous.allocationBytes) {
        offset = llvm::alignTo(previous.offset + previous.allocationBytes,
                               *alignment);
      }
    }
    if (region.allocationBytes > budget ||
        offset > budget - region.allocationBytes) {
      region.declarations.front().emitOpError()
          << "compiler-l1 placement exceeds L1 budget " << budget
          << " bytes (payload, control records, and alignment included); "
             "greedy placement does not prove infeasibility";
      return failure();
    }
    region.offset = offset;
    placed.push_back(regionIndex);
  }
  uint64_t arenaBytes = 0;
  for (const L1Region &region : plan) {
    arenaBytes = std::max(arenaBytes, region.offset + region.allocationBytes);
  }
  return L1AllocationPlan{std::move(plan), arenaBytes};
}
} // namespace

LogicalResult
allocateCompilerL1(ModuleOp module,
                   const DFBLogicalIdentityAnalysis &identities,
                   uint64_t budgetOverride, bool reuseStorage,
                   const DFBConcurrentKernelLivenessAnalysis &liveness) {
  auto budget = getUsableDFBL1Bytes(
      module,
      budgetOverride ? std::optional<uint64_t>(budgetOverride) : std::nullopt);
  FailureOr<L1AllocationPlan> maybePlan =
      planRegions(module, identities, budget, reuseStorage, liveness);
  if (failed(maybePlan)) {
    return failure();
  }
  const L1AllocationPlan &plan = *maybePlan;
  OpBuilder builder(module.getContext());
  SmallVector<Attribute> allocations;
  for (auto [regionIndex, region] : llvm::enumerate(plan.regions)) {
    for (BindCBOp declaration : region.declarations) {
      declaration.setDfbIdAttr(builder.getIndexAttr(region.logicalId));
      declaration.setCbIndexAttr(builder.getIndexAttr(regionIndex));
    }
    allocations.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("dfb_index",
                             builder.getI32IntegerAttr(regionIndex)),
        builder.getNamedAttr("storage_index",
                             builder.getI32IntegerAttr(regionIndex)),
        builder.getNamedAttr("num_tiles",
                             builder.getI32IntegerAttr(region.pages)),
        builder.getNamedAttr("page_size",
                             builder.getI32IntegerAttr(region.pageBytes)),
        builder.getNamedAttr("block_count", builder.getI32IntegerAttr(
                                                region.type.getBlockCount())),
        builder.getNamedAttr("element_type",
                             TypeAttr::get(region.type.getElementType())),
        builder.getNamedAttr("l1_offset",
                             builder.getI64IntegerAttr(region.stateOffset)),
        builder.getNamedAttr("l1_payload_offset",
                             builder.getI64IntegerAttr(region.offset)),
        builder.getNamedAttr("l1_allocation_bytes",
                             builder.getI64IntegerAttr(region.allocationBytes)),
    }));
  }
  module->setAttr("ttl.l1_arena_bytes",
                  builder.getI64IntegerAttr(plan.arenaBytes));
  module->setAttr(kDFBAllocationsAttrName, builder.getArrayAttr(allocations));
  module->setAttr("ttl.memory_model", builder.getStringAttr("compiler-l1"));
  for (func::FuncOp kernel : module.getOps<func::FuncOp>()) {
    if (kernel->hasAttr(kBaseCTAIndexAttrName)) {
      kernel->setAttr(kBaseCTAIndexAttrName, builder.getI32IntegerAttr(1));
    }
  }
  return success();
}
} // namespace mlir::tt::ttl
