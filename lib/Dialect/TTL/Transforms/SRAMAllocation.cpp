// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocation.h"
#include "DFBAllocationLimits.h"
#include "DFBAnalysisFailure.h"
#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "DFBPhysicalAllocationPlan.h"
#include "SRAMAllocationPlan.h"
#include "SRAMAllocationReport.h"
#include "SRAMAllocator.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Target/TargetInfo.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <limits>
#include <set>
#include <tuple>

namespace mlir::tt::ttl {

namespace {
// Multicast writes carry one destination address; their receivers must share a
// layout.
static SmallVector<SmallVector<LaunchNodeCoord>>
collectSRAMCoreDomains(ModuleOp module, ArrayRef<LaunchNodeCoord> nodes) {
  llvm::EquivalenceClasses<unsigned> equivalence;
  for (unsigned index = 0; index < nodes.size(); ++index) {
    equivalence.insert(index);
  }
  auto mergeReceivers = [&](int64_t startX, int64_t startY, int64_t endX,
                            int64_t endY) {
    std::optional<unsigned> leader;
    for (auto [index, node] : llvm::enumerate(nodes)) {
      if (node.x < startX || node.x > endX || node.y < startY ||
          node.y > endY) {
        continue;
      }
      if (leader) {
        equivalence.unionSets(*leader, index);
      } else {
        leader = index;
      }
    }
  };
  module.walk([&](Operation *operation) {
    for (Type type : operation->getResultTypes()) {
      if (auto pipe = dyn_cast<PipeType>(type);
          pipe && pipe.hasMultipleReceivers()) {
        mergeReceivers(pipe.getDstStartX(), pipe.getDstStartY(),
                       pipe.getDstEndX(), pipe.getDstEndY());
      }
    }
    for (NamedAttribute attribute : operation->getAttrs()) {
      if (auto records = dyn_cast<PipeNetRecordsAttr>(attribute.getValue())) {
        for (auto pipe : records.getPipes()) {
          if (pipe.getIsCollective()) {
            mergeReceivers(pipe.getDstStartX(), pipe.getDstStartY(),
                           pipe.getDstEndX(), pipe.getDstEndY());
          }
        }
      }
    }
  });
  llvm::MapVector<unsigned, SmallVector<LaunchNodeCoord>> groups;
  for (auto [index, node] : llvm::enumerate(nodes)) {
    groups[equivalence.getLeaderValue(index)].push_back(node);
  }
  SmallVector<SmallVector<LaunchNodeCoord>> result;
  for (auto &group : groups) {
    result.push_back(std::move(group.second));
  }
  return result;
}

static FailureOr<SRAMAllocationPlan>
planRegions(ModuleOp module, const DFBLogicalIdentityAnalysis &identities,
            uint64_t budget, bool reuseStorage, const SRAMAllocator &allocator,
            llvm::StringRef allocationMode,
            const DFBConcurrentKernelLivenessAnalysis &liveness) {
  if (allocationMode != "uniform" && allocationMode != "per-core") {
    module.emitOpError("SRAM allocation mode must be uniform or per-core");
    return failure();
  }
  if (allocationMode == "per-core" && !liveness.hasExactLaunchGrid()) {
    module.emitOpError(
        "per-core SRAM allocation requires an exact launch grid");
    return failure();
  }
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
  llvm::MapVector<int64_t, SRAMRegion> regions;
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
  SmallVector<SRAMRegion> plan;
  for (auto &entry : regions) {
    plan.push_back(std::move(entry.second));
  }
  SmallVector<SRAMStorage> storage;
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
    SRAMStorage &allocation = storage[storageIndex];
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
    SRAMRegion &lhs = plan[lhsIndex];
    if (!lhs.tensorBacking) {
      continue;
    }
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < plan.size(); ++rhsIndex) {
      SRAMRegion &rhs = plan[rhsIndex];
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
  SRAMAllocationProblem problem;
  problem.alignmentBytes = alignment;
  problem.payloadBaseOffset = *controlBytes;
  problem.budgetBytes = budget;
  SmallVector<unsigned> storageIndexByAllocationRegion;
  for (unsigned storageIndex = 0; storageIndex < storage.size();
       ++storageIndex) {
    SRAMStorage &allocation = storage[storageIndex];
    allocation.stateOffset = storageIndex * kCompilerSRAMControlRecordBytes;
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
    const SRAMStorage &allocation = storage[storageIndex];
    for (unsigned previousRegionIndex = 0;
         previousRegionIndex < allocationRegionIndex; ++previousRegionIndex) {
      unsigned previousStorageIndex =
          storageIndexByAllocationRegion[previousRegionIndex];
      const SRAMStorage &previousAllocation = storage[previousStorageIndex];
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
  SmallVector<SRAMAllocationDomainProblem> requests;
  SmallVector<SmallVector<LaunchNodeCoord>> coreDomains;
  if (allocationMode == "per-core") {
    coreDomains = collectSRAMCoreDomains(module, liveness.getLaunchNodes());
  }
  if (allocationMode == "uniform") {
    requests.push_back(
        {std::move(problem), std::move(storageIndexByAllocationRegion)});
  } else {
    for (const auto &coreDomain : coreDomains) {
      SRAMAllocationDomainProblem request;
      request.allocation.alignmentBytes = alignment;
      request.allocation.payloadBaseOffset = *controlBytes;
      request.allocation.budgetBytes = budget;
      SmallVector<unsigned> sourceRegions;
      for (auto indexedStorage :
           llvm::enumerate(storageIndexByAllocationRegion)) {
        unsigned sourceRegion = indexedStorage.index();
        unsigned storageIndex = indexedStorage.value();
        bool active = llvm::any_of(coreDomain, [&](LaunchNodeCoord node) {
          return llvm::any_of(
              storage[storageIndex].members, [&](unsigned member) {
                const auto &lifecycle =
                    liveness.getLogicalDFBLifecycles()[lifecycleIndices.lookup(
                        plan[member].logicalId)];
                if (lifecycle.launchDomain.known &&
                    !llvm::is_contained(lifecycle.launchDomain.nodes, node)) {
                  return false;
                }
                const auto *lifetime =
                    lifecycle.launchDomain.known
                        ? lifecycle.findNodeLifetime(node)
                        : lifecycle.findPossibleNodeLifetime(node);
                return !lifetime || lifetime->mayBeActive;
              });
        });
        if (!active) {
          continue;
        }
        sourceRegions.push_back(sourceRegion);
        request.storageIndices.push_back(storageIndex);
        request.allocation.regionBytes.push_back(
            problem.regionBytes[sourceRegion]);
      }
      request.allocation.conflicts = InterferenceGraph(sourceRegions.size());
      for (unsigned left = 0; left < sourceRegions.size(); ++left) {
        for (unsigned right = 0; right < left; ++right) {
          if (problem.conflicts.interferes(sourceRegions[left],
                                           sourceRegions[right])) {
            request.allocation.conflicts.addInterference(left, right);
          }
        }
      }
      requests.push_back(std::move(request));
    }
  }
  SRAMAllocationDomainFailure domainFailure;
  auto domains = allocator.allocateDomains(requests, domainFailure);
  if (failed(domains)) {
    auto diagnostic =
        domainFailure.storageIndex
            ? plan[storage[*domainFailure.storageIndex].members.front()]
                  .declarations.front()
                  .emitOpError()
            : module.emitOpError();
    diagnostic << "compiler-sram " << domainFailure.reason;
    if (domainFailure.kind == SRAMPlacementFailureKind::BudgetExceeded) {
      diagnostic << " (payload, control records, and alignment included); "
                 << allocator.getName()
                 << " placement does not prove infeasibility";
    }
    return failure();
  }
  SRAMAllocationPlan result{std::move(plan), std::move(storage),
                            *controlBytes, alignment, *controlBytes, {}};
  for (auto [domainIndex, domain] : llvm::enumerate(*domains)) {
    result.arenaBytes = std::max(result.arenaBytes, domain.arenaBytes);
    if (allocationMode == "uniform") {
      for (const auto &placement : domain.placements) {
        result.storage[placement.storageIndex].offset = placement.offset;
      }
    } else {
      for (LaunchNodeCoord node : coreDomains[domainIndex]) {
        SRAMCoreLayout layout{
            node, {}, domain.arenaBytes, static_cast<unsigned>(domainIndex)};
        layout.payloadOffsets.resize(result.storage.size());
        for (const auto &placement : domain.placements) {
          layout.payloadOffsets[placement.storageIndex] = placement.offset;
        }
        result.coreLayouts.push_back(std::move(layout));
      }
    }
  }
  if (allocationMode != "uniform") {
    for (auto &owner : result.storage) {
      owner.offset = *controlBytes;
    }
  }
  return result;
}

using BackingHandoffsByOrdinal = DenseMap<int64_t, SmallVector<Attribute>>;

static FailureOr<BackingHandoffsByOrdinal>
buildBackingHandoffs(const SRAMAllocationPlan &plan,
                     const DFBConcurrentKernelLivenessAnalysis &liveness,
                     OpBuilder &builder) {
  DenseMap<int64_t, unsigned> lifecycleIndexByLogicalId;
  for (auto [index, lifecycle] :
       llvm::enumerate(liveness.getLogicalDFBLifecycles())) {
    lifecycleIndexByLogicalId.try_emplace(lifecycle.logicalId, index);
  }
  BackingHandoffsByOrdinal handoffs;
  std::set<std::tuple<int64_t, unsigned, unsigned, int64_t, int64_t>> seen;
  for (const SRAMStorage &storage : plan.storage) {
    for (auto [memberPosition, firstIndex] : llvm::enumerate(storage.members)) {
      const SRAMRegion &first = plan.regions[firstIndex];
      for (unsigned secondIndex :
           ArrayRef<unsigned>(storage.members).drop_front(memberPosition + 1)) {
        const SRAMRegion &second = plan.regions[secondIndex];
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
        const SRAMRegion &exactDomain = first.tensorBacking ? first : second;
        for (LaunchNodeCoord node : exactDomain.launchDomain.nodes) {
          const SRAMRegion &other = first.tensorBacking ? second : first;
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

LogicalResult allocateSRAM(
    ModuleOp module, const DFBLogicalIdentityAnalysis &identities,
    uint64_t budgetOverride, bool reuseStorage, const SRAMAllocator &allocator,
    bool reportAllocation, llvm::StringRef allocationMode,
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
  FailureOr<SRAMAllocationPlan> maybePlan = planRegions(
      module, identities, budget, reuseStorage, allocator, allocationMode,
      liveness);
  if (failed(maybePlan)) {
    return failure();
  }
  const SRAMAllocationPlan &plan = *maybePlan;
  OpBuilder builder(module.getContext());
  SmallVector<Attribute> allocations;
  DenseMap<int64_t, int32_t> allocationIndexByLogicalId;
  for (auto [regionIndex, region] : llvm::enumerate(plan.regions)) {
    const SRAMStorage &storage = plan.storage[region.storageIndex];
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
    if (!plan.coreLayouts.empty()) {
      SmallVector<Attribute> layouts;
      for (const auto &layout : plan.coreLayouts) {
        auto offset = layout.payloadOffsets[region.storageIndex];
        layouts.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr(
                "node", builder.getArrayAttr(
                            {builder.getI64IntegerAttr(layout.node.x),
                             builder.getI64IntegerAttr(layout.node.y)})),
            builder.getNamedAttr("payload_offset",
                                 builder.getI64IntegerAttr(
                                     offset.value_or(storage.stateOffset))),
            builder.getNamedAttr("payload_present",
                                 builder.getBoolAttr(offset.has_value())),
            builder.getNamedAttr("arena_bytes",
                                 builder.getI64IntegerAttr(layout.arenaBytes)),
            builder.getNamedAttr("domain",
                                 builder.getI64IntegerAttr(layout.domain)),
        }));
      }
      entryAttributes.push_back(builder.getNamedAttr(
          "sram_core_layouts", builder.getArrayAttr(layouts)));
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
  if (reportAllocation) {
    const auto conflicts = DFBPhysicalConflictModel::buildStorage(
        liveness, DFBStorageConflictMode::CompilerManaged);
    if (plan.coreLayouts.empty()) {
      printSRAMAllocationReport(llvm::errs(), plan, liveness, conflicts,
                                allocator.getName(), reuseStorage, budget);
    } else {
      SmallVector<unsigned> reportedDomains;
      for (const SRAMCoreLayout &layout : plan.coreLayouts) {
        if (llvm::is_contained(reportedDomains, layout.domain)) {
          continue;
        }
        reportedDomains.push_back(layout.domain);
        SRAMAllocationPlan domainPlan = plan;
        domainPlan.arenaBytes = layout.arenaBytes;
        llvm::erase_if(domainPlan.coreLayouts, [&](const SRAMCoreLayout &entry) {
          return entry.domain != layout.domain;
        });
        for (auto [ownerIndex, owner] : llvm::enumerate(domainPlan.storage)) {
          if (layout.payloadOffsets[ownerIndex]) {
            owner.offset = *layout.payloadOffsets[ownerIndex];
          } else {
            owner.allocationBytes = 0;
          }
        }
        printSRAMAllocationReport(llvm::errs(), domainPlan, liveness, conflicts,
                                  allocator.getName(), reuseStorage, budget);
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
  if (!plan.coreLayouts.empty()) {
    module->setAttr("ttl.sram_allocation_mode",
                    builder.getStringAttr("per-core"));
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
