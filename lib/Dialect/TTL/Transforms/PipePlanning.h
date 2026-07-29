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

  /// Select storage for compiler-managed synchronization counters.
  PipeCounterAllocationPolicy counterAllocationPolicy =
      PipeCounterAllocationPolicy::LocalThenGlobal;
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
                   PipeSendPlan sendPlan)
      : pipeType(pipeType), resources(resources),
        operationPlan(std::move(sendPlan)) {}

  PipeTransferPlan(PipeType pipeType, const PipeResourceInfo &resources,
                   PipePostPlan postPlan)
      : pipeType(pipeType), resources(resources),
        operationPlan(std::move(postPlan)) {}

  PipeType pipeType;
  PipeResourceInfo resources;
  std::variant<PipeSendPlan, PipePostPlan> operationPlan;
};

/// PipeNet decisions shared by all lowering patterns for one module.
class PipeModulePlan {
public:
  /// Return the resource assignment for every transfer operation.
  const PipeResourcePlan &getResourcePlan() const { return resourcePlan; }

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
