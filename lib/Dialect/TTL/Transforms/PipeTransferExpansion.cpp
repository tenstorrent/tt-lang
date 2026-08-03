// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeTransferExpansion.h"

#include "PipeGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttl {
namespace {

/// Resolve an untyped receive handle to its unique high-level pipe copy.
static FailureOr<CopyOp> findPipeReceiveCopy(ValueOriginAnalysis &analysis,
                                             Value value) {
  return analysis.getOrigins(value).uniqueDefiningOp<CopyOp>(isPipeReceiveCopy);
}

/// Convert a semantic transfer contract to its explicit IR enum.
static PipeTransferKind getPipeTransferKind(PipeTransferContract contract) {
  return isCollectiveTransfer(contract) ? PipeTransferKind::Collective
                                        : PipeTransferKind::PointToPoint;
}

/// Return the contract shared by every possible value of a pipe operand.
///
/// Create and selected-pipe operations preserve an explicit collective
/// contract. A block argument has no defining pipe op, so its type supplies
/// the contract.
static FailureOr<PipeTransferContract>
getPipeTransferContractForPipeValue(ValueOriginAnalysis &analysis, Value pipe) {
  return analysis.getOrigins(pipe).uniqueMapped<PipeTransferContract>(
      [](Value origin) -> FailureOr<PipeTransferContract> {
        if (auto createPipe = origin.getDefiningOp<CreatePipeOp>()) {
          return getPipeTransferContract(createPipe);
        }
        if (auto selectedSrc = origin.getDefiningOp<SelectPipeSrcOp>()) {
          return getPipeTransferContract(
              selectedSrc.getRecords().getPipes().front());
        }
        if (auto selectedDst = origin.getDefiningOp<SelectPipeDstOp>()) {
          return getPipeTransferContract(
              selectedDst.getRecords().getPipes().front());
        }
        if (isa<BlockArgument>(origin) && isa<PipeType>(origin.getType())) {
          return cast<PipeType>(origin.getType()).hasMultipleReceivers()
                     ? PipeTransferContract::Collective
                     : PipeTransferContract::PointToPoint;
        }
        return failure();
      });
}

/// Prove that every possible pipe definition has the same device transfer.
static FailureOr<std::optional<DeviceTransferAttr>>
getDeviceTransferForPipeValue(ValueOriginAnalysis &analysis, Value pipe) {
  return analysis.getOrigins(pipe)
      .uniqueMapped<std::optional<DeviceTransferAttr>>(
          [](Value origin) -> FailureOr<std::optional<DeviceTransferAttr>> {
            if (auto createPipe = origin.getDefiningOp<CreatePipeOp>()) {
              DeviceTransferAttr deviceTransfer =
                  createPipe.getDeviceTransferAttr();
              return deviceTransfer
                         ? std::optional<DeviceTransferAttr>(deviceTransfer)
                         : std::optional<DeviceTransferAttr>();
            }
            return failure();
          });
}

/// Create one scalar transfer reference for `pipe`.
static PipeTransferCreateOp createPipeTransfer(OpBuilder &builder,
                                               Location location, Value pipe,
                                               PipeTransferContract contract,
                                               DeviceTransferAttr deviceTransfer) {
  auto kindAttr = PipeTransferKindAttr::get(builder.getContext(),
                                            getPipeTransferKind(contract));
  return PipeTransferCreateOp::create(
      builder, location, PipeTransferType::get(builder.getContext()), pipe,
      kindAttr, deviceTransfer);
}

/// Return the PipeNet id represented by a static or selected pipe value.
static FailureOr<int64_t> getPipeNetIdForPipeValue(Operation *op, Value pipe) {
  FailureOr<PipeReference> pipeRef = getPipeReference(op, pipe);
  if (failed(pipeRef)) {
    return failure();
  }
  return pipeRef->getPipeNetId();
}

/// Reuse a transfer for a direct create op or create one at the use site.
static Value getOrCreatePipeTransfer(
    OpBuilder &builder, Location location, Value pipe,
    PipeTransferContract contract, DeviceTransferAttr deviceTransfer,
    llvm::MapVector<Value, Value> &transferByDirectCreatePipe) {
  Value key = traceUnrealizedCasts(pipe);
  if (auto createPipe = key.getDefiningOp<CreatePipeOp>()) {
    auto transferIt = transferByDirectCreatePipe.find(key);
    if (transferIt != transferByDirectCreatePipe.end()) {
      return transferIt->second;
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(createPipe);
    auto transferOp = createPipeTransfer(builder, createPipe.getLoc(), key,
                                         contract, deviceTransfer);
    transferByDirectCreatePipe[key] = transferOp.getTransfer();
    return transferOp.getTransfer();
  }

  // A shared transfer for block arguments and region results would require a
  // new dominance choice. Keeping it at the use preserves current semantics.
  return createPipeTransfer(builder, location, pipe, contract, deviceTransfer)
      .getTransfer();
}

/// High-level pipe copy and its proven protocol properties.
struct PipeCopyExpansion {
  CopyOp copy;
  PipeTransferContract contract;
  DeviceTransferAttr deviceTransfer;
  int64_t pipeNetId = 0;
};

/// Receive-handle wait and the PipeNet id used to create its typed token.
struct PipeReceiveWaitExpansion {
  WaitOp wait;
  int64_t pipeNetId = 0;
};

/// Operations replaced when high-level pipe copies become pipe transfer IR.
struct PipeTransferExpansionPlan {
  SmallVector<CreatePipeOp> createPipes;
  SmallVector<PipeCopyExpansion> receiveCopies;
  SmallVector<PipeCopyExpansion> sendCopies;
  SmallVector<PipeReceiveWaitExpansion> receiveWaits;
  SmallVector<WaitOp> unreachableReceiveWaits;
};

/// Collect every replacement before expansion invalidates origin analysis.
static FailureOr<PipeTransferExpansionPlan>
buildPipeTransferExpansionPlan(ModuleOp module, ValueOriginAnalysis &analysis) {
  PipeTransferExpansionPlan plan;
  module.walk([&](CreatePipeOp op) { plan.createPipes.push_back(op); });

  LogicalResult result = success();
  auto recordCopy = [&](CopyOp copyOp, Value pipe,
                        SmallVectorImpl<PipeCopyExpansion> &expansions) {
    FailureOr<PipeTransferContract> contract =
        getPipeTransferContractForPipeValue(analysis, pipe);
    if (failed(contract)) {
      copyOp.emitError()
          << "requires a consistent transfer contract for all possible "
             "pipe values";
      result = failure();
      return;
    }
    FailureOr<std::optional<DeviceTransferAttr>> maybeDeviceTransfer =
        getDeviceTransferForPipeValue(analysis, pipe);
    if (failed(maybeDeviceTransfer)) {
      copyOp.emitError() << "requires every possible pipe definition to be "
                            "ttl.create_pipe with the same device transfer";
      result = failure();
      return;
    }
    FailureOr<int64_t> pipeNetId =
        getPipeNetIdForPipeValue(copyOp.getOperation(), pipe);
    if (failed(pipeNetId)) {
      result = failure();
      return;
    }
    expansions.push_back({copyOp, *contract,
                          maybeDeviceTransfer->value_or(DeviceTransferAttr()),
                          *pipeNetId});
  };

  module.walk([&](CopyOp op) {
    if (isPipeReceiveCopy(op)) {
      recordCopy(op, op.getSrc(), plan.receiveCopies);
      return;
    }
    if (isPipeSendCopy(op)) {
      recordCopy(op, op.getDst(), plan.sendCopies);
    }
  });
  if (failed(result)) {
    return failure();
  }

  module.walk([&](WaitOp waitOp) {
    auto handleType =
        mlir::dyn_cast<TransferHandleType>(waitOp.getXf().getType());
    if (!handleType || handleType.getKind()) {
      return;
    }
    if (analysis.getOrigins(waitOp.getXf()).empty()) {
      plan.unreachableReceiveWaits.push_back(waitOp);
      return;
    }
    FailureOr<CopyOp> maybeCopyOp =
        findPipeReceiveCopy(analysis, waitOp.getXf());
    if (failed(maybeCopyOp)) {
      waitOp.emitError() << "untyped transfer handle wait requires every "
                            "possible source to be the same pipe receive "
                            "ttl.copy";
      result = failure();
      return;
    }
    FailureOr<int64_t> pipeNetId = getPipeNetIdForPipeValue(
        waitOp.getOperation(), maybeCopyOp->getSrc());
    if (failed(pipeNetId)) {
      result = failure();
      return;
    }
    plan.receiveWaits.push_back({waitOp, *pipeNetId});
  });
  if (failed(result)) {
    return failure();
  }
  return plan;
}

/// Apply a complete expansion plan without querying invalidated analysis.
static void
applyPipeTransferExpansionPlan(ModuleOp module,
                               const PipeTransferExpansionPlan &plan) {
  // An unreachable receive handle has no observable completion to wait for.
  for (WaitOp waitOp : plan.unreachableReceiveWaits) {
    waitOp.erase();
  }

  OpBuilder builder(module.getContext());
  llvm::MapVector<Value, Value> transferByDirectCreatePipe;
  for (CreatePipeOp createPipe : plan.createPipes) {
    builder.setInsertionPointAfter(createPipe);
    auto transferOp =
        createPipeTransfer(builder, createPipe.getLoc(), createPipe.getResult(),
                           getPipeTransferContract(createPipe),
                           createPipe.getDeviceTransferAttr());
    transferByDirectCreatePipe[createPipe.getResult()] =
        transferOp.getTransfer();
  }

  for (const PipeCopyExpansion &expansion : plan.receiveCopies) {
    CopyOp copyOp = expansion.copy;
    builder.setInsertionPoint(copyOp);
    Value transfer = getOrCreatePipeTransfer(
        builder, copyOp.getLoc(), copyOp.getSrc(), expansion.contract,
        expansion.deviceTransfer, transferByDirectCreatePipe);
    auto postOp = PipeTransferPostOp::create(
        builder, copyOp.getLoc(),
        PipeTokenType::get(builder.getContext(), expansion.pipeNetId),
        transfer, copyOp.getDst());
    auto handleCast = UnrealizedConversionCastOp::create(
        builder, copyOp.getLoc(), copyOp.getResult().getType(),
        ValueRange{postOp.getToken()});
    copyOp.getResult().replaceAllUsesWith(handleCast.getResult(0));
    copyOp->erase();
  }

  for (const PipeCopyExpansion &expansion : plan.sendCopies) {
    CopyOp copyOp = expansion.copy;
    builder.setInsertionPoint(copyOp);
    Value transfer = getOrCreatePipeTransfer(
        builder, copyOp.getLoc(), copyOp.getDst(), expansion.contract,
        expansion.deviceTransfer, transferByDirectCreatePipe);
    auto sendOp = PipeTransferSendOp::create(builder, copyOp.getLoc(),
                                             copyOp.getResult().getType(),
                                             transfer, copyOp.getSrc());
    copyOp.getResult().replaceAllUsesWith(sendOp.getXf());
    copyOp->erase();
  }

  for (const PipeReceiveWaitExpansion &wait : plan.receiveWaits) {
    WaitOp waitOp = wait.wait;
    builder.setInsertionPoint(waitOp);
    auto tokenCast = UnrealizedConversionCastOp::create(
        builder, waitOp.getLoc(),
        PipeTokenType::get(builder.getContext(), wait.pipeNetId),
        ValueRange{waitOp.getXf()});
    PipeTransferWaitOp::create(builder, waitOp.getLoc(),
                               tokenCast.getResult(0));
    waitOp->erase();
  }
}

} // namespace

LogicalResult expandPipeTransfers(ModuleOp module,
                                  ValueOriginAnalysis &analysis) {
  FailureOr<PipeTransferExpansionPlan> maybePlan =
      buildPipeTransferExpansionPlan(module, analysis);
  if (failed(maybePlan)) {
    return failure();
  }
  applyPipeTransferExpansionPlan(module, *maybePlan);
  return success();
}

} // namespace mlir::tt::ttl
