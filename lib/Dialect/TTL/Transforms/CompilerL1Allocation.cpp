// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "CompilerL1Allocation.h"
#include "CompilerL1Allocator.h"
#include "DFBAllocationLimits.h"
#include "DFBAnalysisFailure.h"
#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "DFBPhysicalAllocationPlan.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <limits>
#include <set>
#include <tuple>

namespace mlir::tt::ttl {

namespace {
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
    if (failed(payloadBytes)) {
      declaration.emitOpError() << "compiler-sram " << failureReason;
      return failure();
    }
    if (failed(pages) || failed(pageBytes)) {
      declaration.emitOpError(
          "compiler-sram storage size is not representable");
      return failure();
    }
    std::optional<uint64_t> capacityPages = llvm::checkedMulUnsigned(
        *pages, static_cast<uint64_t>(type.getBlockCount()));
    if (!capacityPages || *capacityPages >= (uint64_t{1} << 31) ||
        *payloadBytes > std::numeric_limits<uint32_t>::max() ||
        *pageBytes > std::numeric_limits<int32_t>::max() ||
        type.getBlockCount() > std::numeric_limits<int32_t>::max() ||
        *pages > std::numeric_limits<int32_t>::max()) {
      declaration.emitOpError(
          "compiler-sram storage size is not representable");
      return failure();
    }
    TensorBackingAttr tensorBacking = declaration.getTensorBackingAttr();
    uint64_t allocationBytes = 0;
    if (!tensorBacking) {
      FailureOr<uint64_t> alignedBytes =
          getL1AllocationSizeBytes(module, *payloadBytes);
      if (failed(alignedBytes)) {
        declaration.emitOpError(
            "compiler-sram target-aligned storage size is not representable");
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
          "compiler-sram tensor backing requires an exact non-empty "
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
  // Only a validated allocation group may transfer control-state ownership.
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
            "compiler-sram tensor-backed DFB byte ranges partially overlap on "
            "a shared launch node");
        return failure();
      }
      assert(lifecycleIndices.contains(lhs.logicalId) &&
             lifecycleIndices.contains(rhs.logicalId));
      if (lhs.storageIndex != rhs.storageIndex &&
          conflicts.conflicts(lifecycleIndices.lookup(lhs.logicalId),
                              lifecycleIndices.lookup(rhs.logicalId))) {
        rhs.declarations.front().emitOpError(
            "compiler-sram identical tensor-backed DFB ranges have "
            "overlapping lifetimes on a shared launch node");
        return failure();
      }
    }
  }
  std::optional<uint64_t> unalignedControlBytes = llvm::checkedMulUnsigned(
      static_cast<uint64_t>(storage.size()), kCompilerSRAMControlRecordBytes);
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
  SmallVector<unsigned> storageIndexByAllocationRegion;
  for (unsigned storageIndex = 0; storageIndex < storage.size();
       ++storageIndex) {
    L1Storage &allocation = storage[storageIndex];
    allocation.stateOffset = storageIndex * kCompilerSRAMControlRecordBytes;
    if (allocation.allocationBytes == 0) {
      continue;
    }
    storageIndexByAllocationRegion.push_back(storageIndex);
    problem.regionBytes.push_back(allocation.allocationBytes);
  }
  unsigned allocationRegionCount = storageIndexByAllocationRegion.size();
  problem.conflicts.assign(allocationRegionCount,
                           llvm::BitVector(allocationRegionCount));
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
      problem.conflicts[allocationRegionIndex].set(previousRegionIndex);
      problem.conflicts[previousRegionIndex].set(allocationRegionIndex);
    }
  }
  SRAMPlacementFailure placementFailure;
  FailureOr<CompilerL1AllocationSolution> solution =
      solveCompilerL1Allocation(allocator, problem, placementFailure);
  if (failed(solution)) {
    auto diagnostic =
        placementFailure.regionIndex
            ? plan[storage[storageIndexByAllocationRegion[*placementFailure
                                                               .regionIndex]]
                       .members.front()]
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
  for (auto [allocationRegionIndex, storageIndex] :
       llvm::enumerate(storageIndexByAllocationRegion)) {
    storage[storageIndex].offset = solution->offsets[allocationRegionIndex];
  }
  uint64_t arenaBytes = std::max(*controlBytes, solution->arenaBytes);
  return L1AllocationPlan{std::move(plan), std::move(storage), arenaBytes};
}

using BackingHandoffsByOrdinal = DenseMap<int64_t, SmallVector<Attribute>>;

static FailureOr<BackingHandoffsByOrdinal>
buildBackingHandoffs(const L1AllocationPlan &plan,
                     const DFBConcurrentKernelLivenessAnalysis &liveness,
                     OpBuilder &builder) {
  DenseMap<int64_t, unsigned> lifecycleIndexByLogicalId;
  for (auto [index, lifecycle] :
       llvm::enumerate(liveness.getLogicalDFBLifecycles())) {
    lifecycleIndexByLogicalId.try_emplace(lifecycle.logicalId, index);
  }
  BackingHandoffsByOrdinal handoffs;
  std::set<std::tuple<int64_t, unsigned, unsigned, int64_t, int64_t>> seen;
  for (const L1Storage &storage : plan.storage) {
    for (auto [memberPosition, firstIndex] : llvm::enumerate(storage.members)) {
      const L1Region &first = plan.regions[firstIndex];
      for (unsigned secondIndex :
           ArrayRef<unsigned>(storage.members).drop_front(memberPosition + 1)) {
        const L1Region &second = plan.regions[secondIndex];
        if (first.tensorBacking == second.tensorBacking) {
          continue;
        }
        if (!first.launchDomain.known || first.launchDomain.nodes.empty() ||
            !second.launchDomain.known || second.launchDomain.nodes.empty()) {
          second.declarations.front()->emitOpError(
              "compiler-sram changed payload backing requires exact, "
              "non-empty launch domains");
          return failure();
        }
        const L1Region &exactDomain = first.tensorBacking ? first : second;
        for (LaunchNodeCoord node : exactDomain.launchDomain.nodes) {
          const L1Region &other = first.tensorBacking ? second : first;
          if (other.launchDomain.known && other.launchDomain.nodes.find(node) ==
                                              other.launchDomain.nodes.end()) {
            continue;
          }
          assert(lifecycleIndexByLogicalId.contains(first.logicalId) &&
                 lifecycleIndexByLogicalId.contains(second.logicalId));
          unsigned firstLifecycleIndex =
              lifecycleIndexByLogicalId.lookup(first.logicalId);
          unsigned secondLifecycleIndex =
              lifecycleIndexByLogicalId.lookup(second.logicalId);
          const DFBLogicalLifecycle &firstLifecycle =
              liveness.getLogicalDFBLifecycles()[firstLifecycleIndex];
          const DFBLogicalLifecycle &secondLifecycle =
              liveness.getLogicalDFBLifecycles()[secondLifecycleIndex];
          const DFBPerNodeLifetime *firstLifetime =
              firstLifecycle.findNodeLifetime(node);
          const DFBPerNodeLifetime *secondLifetime =
              secondLifecycle.findNodeLifetime(node);
          bool possibleDomain = !firstLifetime || !secondLifetime;
          if (!firstLifetime) {
            firstLifetime = firstLifecycle.findPossibleNodeLifetime(node);
          }
          if (!secondLifetime) {
            secondLifetime = secondLifecycle.findPossibleNodeLifetime(node);
          }
          if (!firstLifetime || !secondLifetime ||
              !firstLifetime->mayBeActive || !secondLifetime->mayBeActive) {
            continue;
          }
          bool hasHandoff = false;
          for (auto [firstEpochIndex, firstEpoch] :
               llvm::enumerate(firstLifetime->epochs)) {
            for (auto [secondEpochIndex, secondEpoch] :
                 llvm::enumerate(secondLifetime->epochs)) {
              bool firstBeforeSecond =
                  possibleDomain
                      ? liveness.isConditionallyEpochOrderedBefore(
                            firstLifecycleIndex, firstEpochIndex,
                            secondLifecycleIndex, secondEpochIndex, node)
                      : liveness.isEpochOrderedBefore(
                            firstLifecycleIndex, firstEpochIndex,
                            secondLifecycleIndex, secondEpochIndex, node);
              bool secondBeforeFirst =
                  possibleDomain
                      ? liveness.isConditionallyEpochOrderedBefore(
                            secondLifecycleIndex, secondEpochIndex,
                            firstLifecycleIndex, firstEpochIndex, node)
                      : liveness.isEpochOrderedBefore(
                            secondLifecycleIndex, secondEpochIndex,
                            firstLifecycleIndex, firstEpochIndex, node);
              const DFBLifecycleEpoch &earlierEpoch =
                  firstBeforeSecond ? firstEpoch : secondEpoch;
              if (firstBeforeSecond == secondBeforeFirst ||
                  !earlierEpoch.terminalReconfigurationOrdinal) {
                second.declarations.front()->emitOpError(
                    "compiler-sram changed payload backing requires a "
                    "proved terminal reconfiguration on each shared node");
                return failure();
              }
              unsigned fromIndex = firstBeforeSecond ? firstIndex : secondIndex;
              unsigned toIndex = firstBeforeSecond ? secondIndex : firstIndex;
              int64_t ordinal = *earlierEpoch.terminalReconfigurationOrdinal;
              hasHandoff = true;
              if (!seen.insert({ordinal, fromIndex, toIndex, node.x, node.y})
                       .second) {
                continue;
              }
              handoffs[ordinal].push_back(builder.getDictionaryAttr({
                  builder.getNamedAttr("from_dfb_index",
                                       builder.getI32IntegerAttr(fromIndex)),
                  builder.getNamedAttr("to_dfb_index",
                                       builder.getI32IntegerAttr(toIndex)),
                  builder.getNamedAttr(
                      "node", builder.getArrayAttr(
                                  {builder.getI64IntegerAttr(node.x),
                                   builder.getI64IntegerAttr(node.y)})),
              }));
            }
          }
          if (!hasHandoff) {
            second.declarations.front()->emitOpError(
                "compiler-sram changed payload backing requires an active "
                "reconfiguration epoch on each shared node");
            return failure();
          }
        }
      }
    }
  }
  return handoffs;
}
} // namespace

LogicalResult allocateCompilerL1(
    ModuleOp module, const DFBLogicalIdentityAnalysis &identities,
    uint64_t budgetOverride, bool reuseStorage,
    const CompilerL1Allocator &allocator,
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
    const L1Storage &storage = plan.storage[region.storageIndex];
    allocationIndexByLogicalId.try_emplace(region.logicalId,
                                           static_cast<int32_t>(regionIndex));
    SmallVector<NamedAttribute> entryAttributes{
        builder.getNamedAttr(kDFBAllocationIndexField,
                             builder.getI32IntegerAttr(regionIndex)),
        builder.getNamedAttr(kDFBAllocationStorageIndexField,
                             builder.getI32IntegerAttr(region.storageIndex)),
        builder.getNamedAttr(kDFBAllocationNumTilesField,
                             builder.getI32IntegerAttr(region.pages)),
        builder.getNamedAttr(kDFBAllocationPageSizeField,
                             builder.getI32IntegerAttr(region.pageBytes)),
        builder.getNamedAttr(
            kDFBAllocationBlockCountField,
            builder.getI32IntegerAttr(region.type.getBlockCount())),
        builder.getNamedAttr(kDFBAllocationCapacityPagesField,
                             builder.getI32IntegerAttr(storage.capacityPages)),
        builder.getNamedAttr(kDFBAllocationElementTypeField,
                             TypeAttr::get(region.type.getElementType())),
        builder.getNamedAttr(kDFBAllocationStateOffsetField,
                             builder.getI64IntegerAttr(storage.stateOffset)),
    };
    SmallVector<Attribute> nodes;
    if (region.launchDomain.known) {
      for (LaunchNodeCoord node : region.launchDomain.nodes) {
        nodes.push_back(
            builder.getArrayAttr({builder.getI64IntegerAttr(node.x),
                                  builder.getI64IntegerAttr(node.y)}));
      }
    }
    if (!nodes.empty()) {
      entryAttributes.push_back(builder.getNamedAttr(
          "allocation_nodes", builder.getArrayAttr(nodes)));
    }
    if (region.tensorBacking) {
      auto storageSegment = builder.getDictionaryAttr({
          builder.getNamedAttr("nodes", builder.getArrayAttr(nodes)),
          builder.getNamedAttr("tensor_backing", region.tensorBacking),
      });
      entryAttributes.push_back(builder.getNamedAttr(
          "storage_segments", builder.getArrayAttr({storageSegment})));
    } else {
      entryAttributes.push_back(
          builder.getNamedAttr(kDFBAllocationPayloadOffsetField,
                               builder.getI64IntegerAttr(storage.offset)));
      entryAttributes.push_back(builder.getNamedAttr(
          kDFBAllocationBytesField,
          builder.getI64IntegerAttr(storage.allocationBytes)));
    }
    allocations.push_back(builder.getDictionaryAttr(entryAttributes));
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
  FailureOr<BackingHandoffsByOrdinal> handoffs =
      buildBackingHandoffs(plan, liveness, builder);
  if (failed(handoffs)) {
    return failure();
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
    SmallVector<NamedAttribute> resetAttributes{
        builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
        builder.getNamedAttr("dfb_indices",
                             builder.getDenseI32ArrayAttr(indices)),
    };
    if (auto handoffIt = handoffs->find(ordinal);
        handoffIt != handoffs->end()) {
      resetAttributes.push_back(builder.getNamedAttr(
          "backing_handoffs", builder.getArrayAttr(handoffIt->second)));
    }
    reconfigurationResets.push_back(builder.getDictionaryAttr(resetAttributes));
  }
  assert(reconfigurationResets.size() == resetsByReconfiguration.size() &&
         "every terminal epoch must reference a known reconfiguration");
  for (const auto &[ordinal, records] : *handoffs) {
    auto resetIt = resetsByReconfiguration.find(ordinal);
    if (resetIt == resetsByReconfiguration.end()) {
      module.emitOpError(
          "compiler-sram backing handoff has no terminal reset boundary");
      return failure();
    }
    for (Attribute record : records) {
      auto from =
          cast<DictionaryAttr>(record).getAs<IntegerAttr>("from_dfb_index");
      if (!llvm::is_contained(resetIt->second, from.getInt())) {
        module.emitOpError(
            "compiler-sram backing handoff must reset its earlier DFB");
        return failure();
      }
    }
  }
  for (auto [regionIndex, region] : llvm::enumerate(plan.regions)) {
    for (BindCBOp declaration : region.declarations) {
      declaration.setDfbIdAttr(builder.getIndexAttr(region.logicalId));
      declaration.setCbIndexAttr(builder.getIndexAttr(regionIndex));
    }
  }
  if (reconfigurationResets.empty()) {
    module->removeAttr(kCompilerSRAMReconfigurationResetsAttrName);
  } else {
    module->setAttr(kCompilerSRAMReconfigurationResetsAttrName,
                    builder.getArrayAttr(reconfigurationResets));
  }
  module->setAttr(kL1ArenaBytesAttrName,
                  builder.getI64IntegerAttr(plan.arenaBytes));
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
