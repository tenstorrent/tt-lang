// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipePlanning.h"

#include "mlir/IR/Dominance.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl {

const PipeTransferPlan &
PipeModulePlan::getTransferPlan(Operation *operation) const {
  auto planIt = transferPlans.find(operation);
  assert(planIt != transferPlans.end() &&
         "active pipe send or receiver post has no transfer plan");
  return planIt->second;
}

static PipeType getPipeType(MLIRContext *context,
                            const PipeResourceInfo &resources) {
  const PipeKey &pipe = resources.pipe;
  return PipeType::get(context, pipe.srcX, pipe.srcY, pipe.dstStartX,
                       pipe.dstStartY, pipe.dstEndX, pipe.dstEndY,
                       pipe.pipeNetId);
}

static FailureOr<PipeSendPlan>
buildPipeSendPlan(PipeTransferSendOp sendOp,
                  const DominanceInfo &dominanceInfo) {
  FailureOr<CircularBufferType> maybeDFBType =
      utils::getTTLCircularBufferType(sendOp.getSrc());
  if (failed(maybeDFBType)) {
    sendOp.emitError("pipe transfer source must have a TTL DFB type");
    return failure();
  }
  auto tileType =
      llvm::dyn_cast<ttcore::TileType>((*maybeDFBType).getElementType());
  if (!tileType) {
    sendOp.emitError("pipe transfer source DFB element type must be tile");
    return failure();
  }

  bool readFromDFB =
      llvm::any_of(sendOp.getSrc().getUsers(), [&](Operation *user) {
        return isa<CBWaitOp>(user) && user->getOperand(0) == sendOp.getSrc() &&
               dominanceInfo.dominates(user, sendOp);
      });

  int64_t elementCount = 1;
  for (int64_t dimension : (*maybeDFBType).getShape()) {
    elementCount *= dimension;
  }
  return PipeSendPlan{readFromDFB, elementCount * static_cast<int64_t>(
                                                      tileType.getSizeBytes())};
}

static FailureOr<PipePostPlan>
buildPipePostPlan(PipeTransferPostOp postOp,
                  const PipeResourceInfo &resources) {
  if (resources.addressStorage.usesComputedReceiverDFB()) {
    return PipePostPlan{};
  }

  Value receiverDFB = getAttachedCB(postOp.getDst());
  if (!receiverDFB) {
    postOp.emitError("pipe receive destination is not attached to a DFB");
    return failure();
  }
  FailureOr<CircularBufferType> maybeDFBType =
      utils::getTTLCircularBufferType(receiverDFB);
  if (failed(maybeDFBType)) {
    postOp.emitError("pipe receive destination is not attached to a TTL DFB");
    return failure();
  }
  auto tileType =
      llvm::dyn_cast<ttcore::TileType>((*maybeDFBType).getElementType());
  if (!tileType) {
    postOp.emitError("pipe receiver DFB element type must be tile");
    return failure();
  }
  return PipePostPlan{PipeReceiverAddressPublicationPlan{
      receiverDFB, static_cast<int64_t>(tileType.getSizeBytes())}};
}

FailureOr<PipeModulePlan>
buildPipeModulePlan(ModuleOp module, ValueOriginAnalysis &analysis,
                    const PipeTransferIndex &transferIndex,
                    const PipeGraph &pipeGraph, bool enableComputedAddresses) {
  PipeModulePlan plan;
  buildPipeNetIndex(module, plan.pipeNetIndex);

  if (failed(buildPipeResourcePlan(module, transferIndex, pipeGraph,
                                   plan.resourcePlan,
                                   enableComputedAddresses))) {
    return failure();
  }

  plan.resourceRequirements = getPipeResourceRequirements(plan.resourcePlan);

  module.walk([&](WaitOp waitOp) {
    if (analysis.getOrigins(waitOp.getXf()).allMatch([](Value origin) {
          return static_cast<bool>(origin.getDefiningOp<PipeTransferSendOp>());
        })) {
      plan.completedPipeSendWaits.insert(waitOp);
    }
  });

  DominanceInfo dominanceInfo(module);
  for (const auto &resourceEntry : plan.resourcePlan.resources) {
    Operation *operation = resourceEntry.first;
    const PipeResourceInfo &resources = resourceEntry.second;
    auto addTransferPlan = [&](auto operationPlan) {
      PipeTransferPlan transferPlan(getPipeType(module.getContext(), resources),
                                    resources, std::move(operationPlan));
      auto [planIt, inserted] =
          plan.transferPlans.insert({operation, std::move(transferPlan)});
      (void)planIt;
      assert(inserted && "pipe operation has more than one transfer plan");
    };

    if (auto sendOp = dyn_cast<PipeTransferSendOp>(operation)) {
      FailureOr<PipeSendPlan> maybeSendPlan =
          buildPipeSendPlan(sendOp, dominanceInfo);
      if (failed(maybeSendPlan)) {
        return failure();
      }
      addTransferPlan(*maybeSendPlan);
    } else if (auto postOp = dyn_cast<PipeTransferPostOp>(operation)) {
      FailureOr<PipePostPlan> maybePostPlan =
          buildPipePostPlan(postOp, resources);
      if (failed(maybePostPlan)) {
        return failure();
      }
      addTransferPlan(*maybePostPlan);
    }
  }

  return plan;
}

void applyPipeModuleAttributes(ModuleOp module, const PipeModulePlan &plan) {
  Builder builder(module.getContext());
  const PipeResourcePlan &resources = plan.getResourcePlan();
  for (const auto &[function, dfbIndices] :
       resources.computedAddressDFBIndices) {
    function->setAttr(kPipeComputedAddressDFBIndicesAttrName,
                      builder.getDenseI32ArrayAttr(dfbIndices));
  }

  const PipeResourceRequirements &requirements = plan.getResourceRequirements();
  module->setAttr(kPipeSyncSemaphoreCountAttrName,
                  builder.getI64IntegerAttr(requirements.syncSemaphoreCount));
  if (requirements.globalSemaphoreCount > 0) {
    module->setAttr(
        kPipeGlobalSemaphoreCountAttrName,
        builder.getI64IntegerAttr(requirements.globalSemaphoreCount));
  }
  if (requirements.sramScratchBytes > 0) {
    module->setAttr(kPipeSramScratchBytesAttrName,
                    builder.getI64IntegerAttr(requirements.sramScratchBytes));
  }
}

} // namespace mlir::tt::ttl
