// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// PipeNet Lowering Plans
//===----------------------------------------------------------------------===//
//
// This file declares the immutable protocol and resource decisions consumed by
// TTL-to-TTKernel PipeNet lowering.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPEPLANNING_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPEPLANNING_H

#include "PipeCapacityAnalysis.h"
#include "PipeCounter.h"
#include "PipeLowering.h"
#include "PipeTransportPlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <optional>
#include <variant>

namespace mlir::tt {
class ValueOriginAnalysis;
}

namespace mlir::tt::ttl {

class PipeModulePlan;
class PipeTransferIndex;

/// Options that control PipeNet protocol and resource planning.
struct PipePlanningOptions {
  /// Compute receiver DFB addresses instead of publishing them at runtime.
  bool enableComputedAddresses = false;

  /// Use sender-local capacity counters for transfers proven safe.
  bool enableCapacitySynchronization = false;

  /// Select storage for compiler-managed synchronization counters.
  PipeCounterAllocationPolicy counterAllocationPolicy =
      PipeCounterAllocationPolicy::LocalThenGlobal;

  /// Select routing-plane fabric synchronization for routed transfers.
  FabricRoutePlan *fabricRoutePlan = nullptr;

  /// Additional compiler state appended after PipeNet scratch storage.
  int64_t trailingSramScratchBytes = 0;

  /// Alignment required by the appended compiler state.
  int64_t trailingSramScratchAlignment = 1;
};

/// Protocol selection used while allocating readiness resources.
class PipeSynchronizationSelection {
public:
  /// Return whether `op` uses sender-side capacity synchronization.
  bool usesCapacityProtocol(Operation *op) const;

  /// Return whether `op` uses routing-plane fabric synchronization.
  bool usesFabricProtocol(Operation *op) const;

private:
  friend FailureOr<PipeModulePlan>
  buildPipeModulePlan(ModuleOp, ValueOriginAnalysis &,
                      const PipeTransferIndex &, const PipeGraph &,
                      const PipeNetIndex &, const PipePlanningOptions &);

  llvm::SmallPtrSet<Operation *, 16> capacityTransferOps;
  llvm::SmallPtrSet<Operation *, 16> fabricTransferOps;
};

/// Capacity consumed by one sender before issuing a payload write.
struct PipeCapacityAcquireInfo {
  PipeCounterInfo counter;
  int64_t count = 1;
};

/// Capacity returned to a sender after one receiver DFB pop.
struct PipeCapacityReleaseInfo {
  /// PipeGraph transfer whose receiver capacity is released.
  PipeTransferNodeId transferNode = 0;
  PipeCapacityReleaseTarget target;
  PipeCounterInfo counter;
  int64_t count = 1;
};

/// Initial shared capacity and storage selected for one sender counter.
struct PipeCapacityInitInfo {
  PipeCounterInfo counter;
  int64_t initialCapacity = 0;
};

/// Lowering resources for transfers selected for capacity synchronization.
class PipeCapacityPlan {
public:
  /// Return the capacity consumed before each execution of `op`.
  ArrayRef<PipeCapacityAcquireInfo> lookupAcquires(PipeTransferSendOp op) const;

  /// Return the capacity released immediately after `op`.
  ArrayRef<PipeCapacityReleaseInfo> lookupReleases(CBPopOp op) const;

  /// Find the DFB pops that release capacity for `transferNode`.
  SmallVector<CBPopOp, 1> findReleaseOps(PipeTransferNodeId transferNode) const;

  /// Return the shared-counter initializations grouped by sender function.
  const llvm::MapVector<func::FuncOp, SmallVector<PipeCapacityInitInfo>> &
  getInitializations() const {
    return initializations;
  }

  /// Return whether no transfer has been assigned capacity resources.
  bool empty() const {
    return acquires.empty() && releases.empty() && initializations.empty();
  }

  /// Return the combined completion, readiness, and capacity totals.
  PipeCounterAllocationCounts getCounterAllocationCounts() const {
    return counterAllocator.getCounts();
  }

private:
  friend class PipeCapacityPlanBuilder;

  /// Record one sender acquire for `op`.
  void addAcquire(PipeTransferSendOp op, PipeCapacityAcquireInfo info);

  /// Record one receiver release for `op`.
  void addRelease(CBPopOp op, PipeCapacityReleaseInfo info);

  /// Record the initial shared count for `counter` in `func`.
  void addInitialization(func::FuncOp func, PipeCapacityInitInfo info);

  /// Continue allocation after the completion and readiness counters.
  void initializeCounterAllocation(PipeCounterAllocationCounts counts,
                                   PipeCounterAllocationPolicy policy);

  /// Allocate storage for one proven sender-capacity counter.
  PipeCounterInfo allocateCounter();

  llvm::MapVector<Operation *, SmallVector<PipeCapacityAcquireInfo>> acquires;
  llvm::MapVector<Operation *, SmallVector<PipeCapacityReleaseInfo>> releases;
  llvm::MapVector<func::FuncOp, SmallVector<PipeCapacityInitInfo>>
      initializations;
  PipeCounterAllocator counterAllocator;
};

/// Element count and byte size transferred by one sender operation.
struct PipeTransferPayload {
  int64_t elementCount = 0;
  int64_t elementSizeBytes = 0;
  int64_t sizeBytes = 0;
};

/// Return the payload represented by `blockSpan` consecutive source DFB blocks.
FailureOr<PipeTransferPayload> getPipeTransferPayload(PipeTransferSendOp sendOp,
                                                      int64_t blockSpan);

/// Sender-side DFB access and payload size for one transfer.
struct PipeSendPlan {
  bool usesReadPointer = false;
  int64_t payloadSizeBytes = 0;
  SmallVector<std::size_t> fabricRouteIndices;
};

/// Receiver information needed to publish a destination DFB address.
struct PipeReceiverAddressPublicationPlan {
  Value receiverDFB;
  int64_t tileSizeBytes = 0;
};

/// Receiver-side address publication for one transfer post.
struct PipePostPlan {
  std::optional<PipeReceiverAddressPublicationPlan> addressPublication;
  SmallVector<PipeAddressMode> addressModes;
  SmallVector<std::size_t> fabricRouteIndices;
};

/// Receiver-wait lowering has no operation-specific decisions.
struct PipeWaitPlan {};

/// Static or record-selected pipe resources shared by protocol operations and
/// wait-any candidates.
class PipeResourceAccessPlan {
public:
  using Resources =
      std::variant<PipeResourceInfo, SmallVector<PipeResourceInfo>>;

  PipeResourceAccessPlan(PipeReference pipeReference, Resources resources)
      : pipeReference(std::move(pipeReference)),
        resources(std::move(resources)) {}

  const PipeReference &getPipeReference() const { return pipeReference; }

  bool isSelected() const {
    return std::holds_alternative<SmallVector<PipeResourceInfo>>(resources);
  }

  const PipeResourceInfo &getResources() const {
    assert(!isSelected() &&
           "static resources requested for a record-selected pipe");
    return std::get<PipeResourceInfo>(resources);
  }

  ArrayRef<PipeResourceInfo> getSelectedResources() const {
    assert(isSelected() &&
           "record-selected resources requested for a static pipe");
    return std::get<SmallVector<PipeResourceInfo>>(resources);
  }

private:
  PipeReference pipeReference;
  Resources resources;
};

/// Immutable candidate decisions for one wait-any operation.
class PipeWaitAnyPlan {
public:
  ArrayRef<PipeResourceAccessPlan> getCandidates() const { return candidates; }

private:
  friend FailureOr<PipeModulePlan>
  buildPipeModulePlan(ModuleOp, ValueOriginAnalysis &,
                      const PipeTransferIndex &, const PipeGraph &,
                      const PipeNetIndex &, const PipePlanningOptions &);

  SmallVector<PipeResourceAccessPlan, 0> candidates;
};

/// Complete lowering decisions for one active pipe protocol operation.
class PipeTransferPlan {
public:
  /// Return pipe and resource decisions shared with wait-any candidates.
  const PipeResourceAccessPlan &getResourceAccessPlan() const {
    return resourceAccessPlan;
  }

  /// Return the selected sender-readiness protocol.
  PipeSynchronizationProtocol getSynchronizationProtocol() const {
    return synchronizationProtocol;
  }

  /// Return whether this plan describes a sender operation.
  bool isSend() const {
    return std::holds_alternative<PipeSendPlan>(operationPlan);
  }

  /// Return whether this plan describes a receiver-post operation.
  bool isPost() const {
    return std::holds_alternative<PipePostPlan>(operationPlan);
  }

  /// Return whether this plan describes a receiver-wait operation.
  bool isWait() const {
    return std::holds_alternative<PipeWaitPlan>(operationPlan);
  }

  /// Return sender-only lowering information.
  const PipeSendPlan &getSend() const {
    assert(isSend() && "send plan requested for a receiver post");
    return std::get<PipeSendPlan>(operationPlan);
  }

  /// Return receiver-post-only lowering information.
  const PipePostPlan &getPost() const {
    assert(isPost() && "receiver-post plan requested for another operation");
    return std::get<PipePostPlan>(operationPlan);
  }

private:
  friend FailureOr<PipeModulePlan>
  buildPipeModulePlan(ModuleOp, ValueOriginAnalysis &,
                      const PipeTransferIndex &, const PipeGraph &,
                      const PipeNetIndex &, const PipePlanningOptions &);

  using OperationPlan = std::variant<PipeSendPlan, PipePostPlan, PipeWaitPlan>;

  PipeTransferPlan(PipeReference pipeReference,
                   PipeResourceAccessPlan::Resources resources,
                   PipeSynchronizationProtocol synchronizationProtocol,
                   OperationPlan operationPlan)
      : resourceAccessPlan(std::move(pipeReference), std::move(resources)),
        synchronizationProtocol(synchronizationProtocol),
        operationPlan(std::move(operationPlan)) {}

  PipeResourceAccessPlan resourceAccessPlan;
  PipeSynchronizationProtocol synchronizationProtocol;
  OperationPlan operationPlan;
};

/// PipeNet decisions shared by all lowering patterns for one module.
class PipeModulePlan {
public:
  /// Return the resource assignment for every transfer operation.
  const PipeResourcePlan &getResourcePlan() const { return resourcePlan; }

  /// Return the selected capacity synchronization resources.
  const PipeCapacityPlan &getCapacityPlan() const { return capacityPlan; }

  /// Return the module-level PipeNet resource totals.
  const PipeResourceRequirements &getResourceRequirements() const {
    return resourceRequirements;
  }

  /// Return the byte offset assigned to appended compiler state.
  int64_t getTrailingSramScratchOffset() const {
    return trailingSramScratchOffset;
  }

  /// Return transfer topology grouped by PipeNet id.
  const PipeNetIndex &getPipeNetIndex() const { return pipeNetIndex; }

  /// Return backend-independent scheduling and storage decisions.
  const PipeTransportPlan &getTransportPlan() const { return transportPlan; }

  /// Return send waits whose payload writes complete within the send operation.
  const llvm::SmallPtrSetImpl<Operation *> &getCompletedPipeSendWaits() const {
    return completedPipeSendWaits;
  }

  /// Return the lowering plan for an active pipe protocol operation.
  const PipeTransferPlan &getTransferPlan(Operation *operation) const;

  /// Return the lowering plan for an internal wait-any operation.
  const PipeWaitAnyPlan &getWaitAnyPlan(PipeTransferWaitAnyOp operation) const;

private:
  friend FailureOr<PipeModulePlan>
  buildPipeModulePlan(ModuleOp, ValueOriginAnalysis &,
                      const PipeTransferIndex &, const PipeGraph &,
                      const PipeNetIndex &, const PipePlanningOptions &);

  PipeResourcePlan resourcePlan;
  PipeCapacityPlan capacityPlan;
  PipeResourceRequirements resourceRequirements;
  int64_t trailingSramScratchOffset = 0;
  PipeNetIndex pipeNetIndex;
  PipeTransportPlan transportPlan;
  llvm::SmallPtrSet<Operation *, 8> completedPipeSendWaits;
  llvm::MapVector<Operation *, PipeTransferPlan> transferPlans;
  llvm::MapVector<Operation *, PipeWaitAnyPlan> waitAnyPlans;
};

/// Compute all PipeNet decisions required after transfer IR expansion.
FailureOr<PipeModulePlan> buildPipeModulePlan(
    ModuleOp module, ValueOriginAnalysis &analysis,
    const PipeTransferIndex &transferIndex, const PipeGraph &pipeGraph,
    const PipeNetIndex &pipeNetIndex, const PipePlanningOptions &options);

/// Materialize the module and function attributes recorded by `plan`.
void applyPipeModuleAttributes(ModuleOp module, const PipeModulePlan &plan);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEPLANNING_H
