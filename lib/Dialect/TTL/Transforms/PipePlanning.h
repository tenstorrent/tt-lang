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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

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
};

/// Protocol selection used while allocating readiness resources.
class PipeSynchronizationSelection {
public:
  /// Return whether `op` uses sender-side capacity synchronization.
  bool usesCapacityProtocol(Operation *op) const;

private:
  friend FailureOr<PipeModulePlan>
  buildPipeModulePlan(ModuleOp, ValueOriginAnalysis &,
                      const PipeTransferIndex &, const PipeGraph &,
                      const PipePlanningOptions &);

  llvm::SmallPtrSet<Operation *, 16> capacityTransferOps;
};

/// Capacity consumed by one sender before issuing a payload write.
struct PipeCapacityAcquireInfo {
  PipeCounterInfo counter;
  int64_t count = 1;
};

/// Capacity returned to a sender after one receiver DFB pop.
struct PipeCapacityReleaseInfo {
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

/// Synchronization performed before the sender writes a transfer payload.
enum class PipeSynchronizationProtocol {
  ReceiverPost,
  Capacity,
};

/// Sender-side DFB access and payload size for one transfer.
struct PipeSendPlan {
  bool usesReadPointer = false;
  int64_t payloadSizeBytes = 0;
};

/// Receiver information needed to publish a destination DFB address.
struct PipeReceiverAddressPublicationPlan {
  Value receiverDFB;
  int64_t tileSizeBytes = 0;
};

/// Receiver-side address publication for one transfer post.
struct PipePostPlan {
  std::optional<PipeReceiverAddressPublicationPlan> addressPublication;
};

/// Complete lowering decisions for one send or receiver post operation.
class PipeTransferPlan {
public:
  /// Return the transfer's source, receiver range, and PipeNet id.
  PipeType getPipeType() const { return pipeType; }

  /// Return the completion, readiness, and address resources.
  const PipeResourceInfo &getResources() const { return resources; }

  /// Return the selected sender-readiness protocol.
  PipeSynchronizationProtocol getSynchronizationProtocol() const {
    return synchronizationProtocol;
  }

  /// Return whether this plan describes a sender operation.
  bool isSend() const {
    return std::holds_alternative<PipeSendPlan>(operationPlan);
  }

  /// Return sender-only lowering information.
  const PipeSendPlan &getSend() const {
    assert(isSend() && "send plan requested for a receiver post");
    return std::get<PipeSendPlan>(operationPlan);
  }

  /// Return receiver-post-only lowering information.
  const PipePostPlan &getPost() const {
    assert(!isSend() && "receiver-post plan requested for a send");
    return std::get<PipePostPlan>(operationPlan);
  }

private:
  friend FailureOr<PipeModulePlan>
  buildPipeModulePlan(ModuleOp, ValueOriginAnalysis &,
                      const PipeTransferIndex &, const PipeGraph &,
                      const PipePlanningOptions &);

  PipeTransferPlan(PipeType pipeType, const PipeResourceInfo &resources,
                   PipeSynchronizationProtocol synchronizationProtocol,
                   PipeSendPlan sendPlan)
      : pipeType(pipeType), resources(resources),
        synchronizationProtocol(synchronizationProtocol),
        operationPlan(std::move(sendPlan)) {}

  PipeTransferPlan(PipeType pipeType, const PipeResourceInfo &resources,
                   PipeSynchronizationProtocol synchronizationProtocol,
                   PipePostPlan postPlan)
      : pipeType(pipeType), resources(resources),
        synchronizationProtocol(synchronizationProtocol),
        operationPlan(std::move(postPlan)) {}

  PipeType pipeType;
  PipeResourceInfo resources;
  PipeSynchronizationProtocol synchronizationProtocol;
  std::variant<PipeSendPlan, PipePostPlan> operationPlan;
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

  /// Return transfer topology grouped by PipeNet id.
  const PipeNetIndex &getPipeNetIndex() const { return pipeNetIndex; }

  /// Return send waits whose payload writes complete within the send operation.
  const llvm::SmallPtrSetImpl<Operation *> &getCompletedPipeSendWaits() const {
    return completedPipeSendWaits;
  }

  /// Return the lowering plan for an active send or receiver post.
  const PipeTransferPlan &getTransferPlan(Operation *operation) const;

private:
  friend FailureOr<PipeModulePlan>
  buildPipeModulePlan(ModuleOp, ValueOriginAnalysis &,
                      const PipeTransferIndex &, const PipeGraph &,
                      const PipePlanningOptions &);

  PipeResourcePlan resourcePlan;
  PipeCapacityPlan capacityPlan;
  PipeResourceRequirements resourceRequirements;
  PipeNetIndex pipeNetIndex;
  llvm::SmallPtrSet<Operation *, 8> completedPipeSendWaits;
  llvm::MapVector<Operation *, PipeTransferPlan> transferPlans;
};

/// Compute all PipeNet decisions required after transfer IR expansion.
FailureOr<PipeModulePlan>
buildPipeModulePlan(ModuleOp module, ValueOriginAnalysis &analysis,
                    const PipeTransferIndex &transferIndex,
                    const PipeGraph &pipeGraph,
                    const PipePlanningOptions &options);

/// Materialize the module and function attributes recorded by `plan`.
void applyPipeModuleAttributes(ModuleOp module, const PipeModulePlan &plan);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPEPLANNING_H
