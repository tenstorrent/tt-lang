// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocation.h"
#include "DFBAllocationLimits.h"
#include "DFBAnalysisFailure.h"
#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "DFBPhysicalAllocationPlan.h"
#include "SRAMAllocator.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <limits>

namespace mlir::tt::ttl {
namespace {
constexpr uint64_t kControlWordCount = 2;
constexpr uint64_t kControlRecordBytes = kControlWordCount * sizeof(uint32_t);

struct L1Region {
  int64_t logicalId;
  CircularBufferType type;
  TensorBackingAttr tensorBacking;
  DFBAllocationGroupAttr allocationGroup;
  LaunchNodeDomain launchDomain;
  uint64_t pages;
  uint64_t pageBytes;
  uint64_t capacityPages;
  uint64_t allocationBytes;
  unsigned storageIndex = 0;
  SmallVector<BindCBOp> declarations;
};

struct L1Storage {
  uint64_t capacityPages = 0;
  uint64_t allocationBytes = 0;
  uint64_t offset = 0;
  uint64_t stateOffset = 0;
  SmallVector<unsigned> members;
};

struct L1AllocationPlan {
  SmallVector<L1Region> regions;
  SmallVector<L1Storage> storage;
  uint64_t arenaBytes;
};

static FailureOr<L1AllocationPlan>
planRegions(ModuleOp module, const DFBLogicalIdentityAnalysis &identities,
            uint64_t budget, bool reuseStorage,
            llvm::StringRef allocationStrategy, uint64_t exactSearchLimit,
            const DFBConcurrentKernelLivenessAnalysis &liveness) {
  std::string targetFailure;
  FailureOr<uint64_t> alignment =
      resolveTargetL1AllocationQuantumBytes(module, targetFailure);
  if (failed(alignment)) {
    module.emitOpError() << targetFailure;
    return failure();
  }
  DenseMap<int64_t, const DFBLogicalLifecycle *> lifecycleByLogicalId;
  for (const DFBLogicalLifecycle &lifecycle :
       liveness.getLogicalDFBLifecycles()) {
    lifecycleByLogicalId.try_emplace(lifecycle.logicalId, &lifecycle);
  }
  llvm::MapVector<int64_t, L1Region> regions;
  for (const auto &assignment : identities.getAssignments()) {
    BindCBOp declaration = assignment.declaration;
    auto type = cast<CircularBufferType>(declaration.getResult().getType());
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
    if (failed(pages) || failed(pageBytes) || failed(payloadBytes)) {
      declaration.emitOpError("compiler-l1 storage size is not representable");
      return failure();
    }
    std::optional<uint64_t> capacityPages = llvm::checkedMulUnsigned(
        *pages, static_cast<uint64_t>(type.getBlockCount()));
    if (!capacityPages || *capacityPages >= (uint64_t{1} << 31) ||
        *payloadBytes > std::numeric_limits<uint32_t>::max() ||
        *pageBytes > std::numeric_limits<int32_t>::max() ||
        type.getBlockCount() > std::numeric_limits<int32_t>::max() ||
        *pages > std::numeric_limits<int32_t>::max()) {
      declaration.emitOpError("compiler-l1 storage size is not representable");
      return failure();
    }
    TensorBackingAttr tensorBacking = declaration.getTensorBackingAttr();
    uint64_t allocationBytes = 0;
    if (!tensorBacking) {
      FailureOr<uint64_t> alignedBytes =
          getL1AllocationSizeBytes(module, *payloadBytes);
      if (failed(alignedBytes)) {
        return failure();
      }
      allocationBytes = *alignedBytes;
    }
    const DFBLogicalLifecycle *lifecycle =
        lifecycleByLogicalId.lookup(assignment.logicalId);
    assert(lifecycle && "every logical identity must have a lifecycle");
    if (tensorBacking && (!lifecycle->launchDomain.known ||
                          lifecycle->launchDomain.nodes.empty())) {
      declaration.emitOpError(
          "compiler-l1 tensor backing requires an exact non-empty "
          "launch-node domain");
      return failure();
    }
    regions.insert({assignment.logicalId,
                    {assignment.logicalId,
                     type,
                     tensorBacking,
                     assignment.allocationGroup,
                     lifecycle->launchDomain,
                     *pages,
                     *pageBytes,
                     *capacityPages,
                     allocationBytes,
                     0,
                     {declaration}}});
  }
  SmallVector<L1Region> plan;
  for (auto &entry : regions) {
    plan.push_back(std::move(entry.second));
  }
  SmallVector<L1Storage> storage;
  DenseMap<int64_t, unsigned> storageByAllocationGroup;
  for (auto [regionIndex, region] : llvm::enumerate(plan)) {
    unsigned storageIndex = storage.size();
    bool createStorage = true;
    if (region.allocationGroup) {
      auto [groupIt, inserted] = storageByAllocationGroup.try_emplace(
          region.allocationGroup.getOrdinal(), storageIndex);
      storageIndex = groupIt->second;
      createStorage = inserted;
    }
    if (createStorage) {
      storage.emplace_back();
    }
    L1Storage &allocation = storage[storageIndex];
    allocation.capacityPages =
        std::max(allocation.capacityPages, region.capacityPages);
    allocation.allocationBytes =
        std::max(allocation.allocationBytes, region.allocationBytes);
    allocation.members.push_back(regionIndex);
    region.storageIndex = storageIndex;
  }
  const auto conflicts = DFBPhysicalConflictModel::buildStorage(
      liveness, DFBStorageConflictMode::CompilerManaged);
  DenseMap<int64_t, unsigned> lifecycleIndices;
  for (auto [lifecycleIndex, lifecycle] :
       llvm::enumerate(liveness.getLogicalDFBLifecycles())) {
    lifecycleIndices[lifecycle.logicalId] = lifecycleIndex;
  }
  for (unsigned lhsIndex = 0; lhsIndex < plan.size(); ++lhsIndex) {
    L1Region &lhs = plan[lhsIndex];
    if (!lhs.tensorBacking) {
      continue;
    }
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < plan.size(); ++rhsIndex) {
      L1Region &rhs = plan[rhsIndex];
      if (!rhs.tensorBacking ||
          lhs.tensorBacking.getTensorIndex() !=
              rhs.tensorBacking.getTensorIndex() ||
          lhs.launchDomain.intersectWith(rhs.launchDomain).nodes.empty()) {
        continue;
      }
      int64_t lhsStart = lhs.tensorBacking.getByteOffset();
      int64_t lhsEnd = lhsStart + lhs.tensorBacking.getByteSize();
      int64_t rhsStart = rhs.tensorBacking.getByteOffset();
      int64_t rhsEnd = rhsStart + rhs.tensorBacking.getByteSize();
      if (lhsStart >= rhsEnd || rhsStart >= lhsEnd) {
        continue;
      }
      if (lhs.tensorBacking != rhs.tensorBacking) {
        rhs.declarations.front().emitOpError(
            "compiler-l1 tensor-backed DFB byte ranges partially overlap on "
            "a shared launch node");
        return failure();
      }
      assert(lifecycleIndices.contains(lhs.logicalId) &&
             lifecycleIndices.contains(rhs.logicalId));
      if (lhs.storageIndex != rhs.storageIndex &&
          conflicts.conflicts(lifecycleIndices.lookup(lhs.logicalId),
                              lifecycleIndices.lookup(rhs.logicalId))) {
        rhs.declarations.front().emitOpError(
            "compiler-l1 identical tensor-backed DFB ranges have "
            "overlapping lifetimes on a shared launch node");
        return failure();
      }
    }
  }
  // Only a validated allocation group may transfer control-state ownership.
  std::optional<uint64_t> unalignedControlBytes = llvm::checkedMulUnsigned(
      static_cast<uint64_t>(storage.size()), kControlRecordBytes);
  FailureOr<uint64_t> controlBytes =
      unalignedControlBytes
          ? getL1AllocationSizeBytes(module, *unalignedControlBytes)
          : FailureOr<uint64_t>(failure());
  if (failed(controlBytes) || *controlBytes > budget) {
    module.emitOpError(
        "compiler-l1 control records exceed the available SRAM budget");
    return failure();
  }
  SRAMAllocationProblem problem;
  problem.alignmentBytes = *alignment;
  problem.payloadBaseOffset = *controlBytes;
  problem.budgetBytes = budget;
  SmallVector<unsigned> storageIndexByAllocationRegion;
  for (unsigned storageIndex = 0; storageIndex < storage.size();
       ++storageIndex) {
    L1Storage &allocation = storage[storageIndex];
    allocation.stateOffset = storageIndex * kControlRecordBytes;
    if (allocation.allocationBytes == 0) {
      continue;
    }
    storageIndexByAllocationRegion.push_back(storageIndex);
    problem.regionBytes.push_back(allocation.allocationBytes);
  }
  unsigned allocationRegionCount = storageIndexByAllocationRegion.size();
  problem.conflicts = InterferenceGraph(allocationRegionCount);
  for (unsigned allocationRegionIndex = 0;
       allocationRegionIndex < allocationRegionCount; ++allocationRegionIndex) {
    unsigned storageIndex =
        storageIndexByAllocationRegion[allocationRegionIndex];
    const L1Storage &allocation = storage[storageIndex];
    for (unsigned previousRegionIndex = 0;
         previousRegionIndex < allocationRegionIndex; ++previousRegionIndex) {
      unsigned previousStorageIndex =
          storageIndexByAllocationRegion[previousRegionIndex];
      const L1Storage &previousAllocation = storage[previousStorageIndex];
      bool hasConflict = !reuseStorage;
      for (unsigned member : allocation.members) {
        if (hasConflict) {
          break;
        }
        assert(lifecycleIndices.contains(plan[member].logicalId));
        for (unsigned previousMember : previousAllocation.members) {
          assert(lifecycleIndices.contains(plan[previousMember].logicalId));
          if (conflicts.conflicts(
                  lifecycleIndices.lookup(plan[member].logicalId),
                  lifecycleIndices.lookup(plan[previousMember].logicalId))) {
            hasConflict = true;
            break;
          }
        }
      }
      if (!hasConflict) {
        continue;
      }
      problem.conflicts.addInterference(allocationRegionIndex,
                                        previousRegionIndex);
    }
  }
  std::string allocationFailure;
  SRAMAllocatorOptions allocatorOptions{exactSearchLimit};
  FailureOr<std::unique_ptr<SRAMAllocator>> allocator = createSRAMAllocator(
      allocationStrategy, allocatorOptions, allocationFailure);
  if (failed(allocator)) {
    module.emitOpError() << allocationFailure;
    return failure();
  }
  std::optional<unsigned> failureRegionIndex;
  FailureOr<SRAMAllocationSolution> solution =
      (*allocator)->allocate(problem, failureRegionIndex, allocationFailure);
  if (failed(solution)) {
    auto diagnostic =
        failureRegionIndex
            ? plan[storage[storageIndexByAllocationRegion[*failureRegionIndex]]
                       .members.front()]
                  .declarations.front()
                  .emitOpError()
            : module.emitOpError();
    diagnostic << "compiler-l1 " << allocationFailure;
    if (llvm::StringRef(allocationFailure)
            .starts_with("placement exceeds SRAM budget")) {
      diagnostic << " (payload, control records, and alignment included); "
                 << (*allocator)->getName()
                 << " placement does not prove infeasibility";
    }
    return failure();
  }
  for (auto [allocationRegionIndex, storageIndex] :
       llvm::enumerate(storageIndexByAllocationRegion)) {
    storage[storageIndex].offset = solution->offsets[allocationRegionIndex];
  }
  uint64_t arenaBytes = std::max(*controlBytes, solution->arenaBytes);
  return L1AllocationPlan{std::move(plan), std::move(storage), arenaBytes};
}
} // namespace

LogicalResult allocateSRAM(
    ModuleOp module, const DFBLogicalIdentityAnalysis &identities,
    uint64_t budgetOverride, bool reuseStorage,
    llvm::StringRef allocationStrategy, uint64_t exactSearchLimit,
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    ArrayRef<DFBStaticConfigurationConflict> staticConfigurationConflicts,
    bool unsafeAssumeAllocationGroups,
    SmallVectorImpl<DFBAssumedAllocationGroup> &assumedAllocationGroups) {
  auto groupedLifecycle =
      llvm::find_if(liveness.getLogicalDFBLifecycles(),
                    [](const DFBLogicalLifecycle &lifecycle) {
                      return static_cast<bool>(lifecycle.allocationGroup);
                    });
  if (!reuseStorage &&
      groupedLifecycle != liveness.getLogicalDFBLifecycles().end()) {
    groupedLifecycle->declarations.front()->emitOpError(
        "DFB allocation groups require user DFB reuse to be enabled");
    return failure();
  }
  DFBAnalysisFailure analysisFailure;
  if (failed(validateDFBAllocationGroups(
          liveness, staticConfigurationConflicts, unsafeAssumeAllocationGroups,
          assumedAllocationGroups, analysisFailure))) {
    Operation *errorOperation = analysisFailure.operation
                                    ? analysisFailure.operation
                                    : module.getOperation();
    errorOperation->emitError(analysisFailure.message);
    return failure();
  }
  auto budget = getUsableDFBL1Bytes(
      module,
      budgetOverride ? std::optional<uint64_t>(budgetOverride) : std::nullopt);
  FailureOr<L1AllocationPlan> maybePlan =
      planRegions(module, identities, budget, reuseStorage, allocationStrategy,
                  exactSearchLimit, liveness);
  if (failed(maybePlan)) {
    return failure();
  }
  const L1AllocationPlan &plan = *maybePlan;
  OpBuilder builder(module.getContext());
  SmallVector<Attribute> allocations;
  DenseMap<int64_t, int32_t> allocationIndexByLogicalId;
  for (auto [regionIndex, region] : llvm::enumerate(plan.regions)) {
    const L1Storage &storage = plan.storage[region.storageIndex];
    allocationIndexByLogicalId.try_emplace(region.logicalId,
                                           static_cast<int32_t>(regionIndex));
    for (BindCBOp declaration : region.declarations) {
      declaration.setDfbIdAttr(builder.getIndexAttr(region.logicalId));
      declaration.setCbIndexAttr(builder.getIndexAttr(regionIndex));
    }
    SmallVector<NamedAttribute> entryAttributes{
        builder.getNamedAttr("dfb_index",
                             builder.getI32IntegerAttr(regionIndex)),
        builder.getNamedAttr("storage_index",
                             builder.getI32IntegerAttr(region.storageIndex)),
        builder.getNamedAttr("num_tiles",
                             builder.getI32IntegerAttr(region.pages)),
        builder.getNamedAttr("page_size",
                             builder.getI32IntegerAttr(region.pageBytes)),
        builder.getNamedAttr("block_count", builder.getI32IntegerAttr(
                                                region.type.getBlockCount())),
        builder.getNamedAttr("storage_capacity_pages",
                             builder.getI32IntegerAttr(storage.capacityPages)),
        builder.getNamedAttr("element_type",
                             TypeAttr::get(region.type.getElementType())),
        builder.getNamedAttr("l1_offset",
                             builder.getI64IntegerAttr(storage.stateOffset)),
    };
    if (region.tensorBacking) {
      SmallVector<Attribute> nodes;
      for (LaunchNodeCoord node : region.launchDomain.nodes) {
        nodes.push_back(
            builder.getArrayAttr({builder.getI64IntegerAttr(node.x),
                                  builder.getI64IntegerAttr(node.y)}));
      }
      entryAttributes.push_back(builder.getNamedAttr(
          "allocation_nodes", builder.getArrayAttr(nodes)));
      auto storageSegment = builder.getDictionaryAttr({
          builder.getNamedAttr("nodes", builder.getArrayAttr(nodes)),
          builder.getNamedAttr("tensor_backing", region.tensorBacking),
      });
      entryAttributes.push_back(builder.getNamedAttr(
          "storage_segments", builder.getArrayAttr({storageSegment})));
    } else {
      entryAttributes.push_back(builder.getNamedAttr(
          "l1_payload_offset", builder.getI64IntegerAttr(storage.offset)));
      entryAttributes.push_back(builder.getNamedAttr(
          "l1_allocation_bytes",
          builder.getI64IntegerAttr(storage.allocationBytes)));
    }
    allocations.push_back(builder.getDictionaryAttr(entryAttributes));
  }
  DenseMap<int64_t, SmallVector<int32_t>> resetsByReconfiguration;
  for (const DFBLogicalLifecycle &lifecycle :
       liveness.getLogicalDFBLifecycles()) {
    auto allocationIt = allocationIndexByLogicalId.find(lifecycle.logicalId);
    assert(allocationIt != allocationIndexByLogicalId.end() &&
           "every logical DFB must have a compiler-l1 allocation");
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
    module->removeAttr(kCompilerL1ReconfigurationResetsAttrName);
  } else {
    module->setAttr(kCompilerL1ReconfigurationResetsAttrName,
                    builder.getArrayAttr(reconfigurationResets));
  }
  module->setAttr(kL1ArenaBytesAttrName,
                  builder.getI64IntegerAttr(plan.arenaBytes));
  module->setAttr(kDFBAllocationsAttrName, builder.getArrayAttr(allocations));
  module->setAttr(kMemoryModelAttrName,
                  builder.getStringAttr(kCompilerL1MemoryModel));
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
