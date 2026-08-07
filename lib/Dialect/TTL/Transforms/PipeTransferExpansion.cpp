// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeTransferExpansion.h"

#include "PipeGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttl {
namespace {

enum class PipeTransferExpansionMode {
  All,
  StaticPipesOnly,
};

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

/// Create one scalar transfer reference for `pipe`.
static PipeTransferCreateOp createPipeTransfer(OpBuilder &builder,
                                               Location location, Value pipe,
                                               PipeTransferContract contract) {
  auto kindAttr = PipeTransferKindAttr::get(builder.getContext(),
                                            getPipeTransferKind(contract));
  return PipeTransferCreateOp::create(
      builder, location, PipeTransferType::get(builder.getContext()), pipe,
      kindAttr);
}

/// Return the PipeNet id encoded by a static or selected pipe value.
static FailureOr<int64_t> getPipeNetIdForPipeValue(Operation *operation,
                                                   Value pipe) {
  FailureOr<PipeReference> pipeRef = getPipeReference(operation, pipe);
  if (failed(pipeRef)) {
    return failure();
  }
  return pipeRef->getPipeNetId();
}

/// Reuse a transfer for a direct create op or create one at the use site.
static Value getOrCreatePipeTransfer(
    OpBuilder &builder, Location location, Value pipe,
    PipeTransferContract contract,
    llvm::MapVector<Value, Value> &transferByDirectCreatePipe) {
  Value key = traceUnrealizedCasts(pipe);
  if (auto createPipe = key.getDefiningOp<CreatePipeOp>()) {
    auto transferIt = transferByDirectCreatePipe.find(key);
    if (transferIt != transferByDirectCreatePipe.end()) {
      return transferIt->second;
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(createPipe);
    auto transferOp =
        createPipeTransfer(builder, createPipe.getLoc(), key, contract);
    transferByDirectCreatePipe[key] = transferOp.getTransfer();
    return transferOp.getTransfer();
  }

  // A shared transfer for block arguments and region results would require a
  // new dominance choice. Keeping it at the use preserves current semantics.
  return createPipeTransfer(builder, location, pipe, contract).getTransfer();
}

/// High-level pipe copy and its proven transfer contract.
struct PipeCopyExpansion {
  CopyOp copy;
  PipeTransferContract contract;
  std::optional<int64_t> pipeNetId;
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

/// Return whether the selected expansion mode includes `pipe`.
static bool
shouldExpandPipeValue(Value pipe, PipeTransferExpansionMode mode,
                      const llvm::DenseSet<PipeKey> &selectedPipeKeys) {
  if (mode == PipeTransferExpansionMode::All) {
    return true;
  }
  auto pipeType =
      mlir::dyn_cast<PipeType>(traceUnrealizedCasts(pipe).getType());
  return pipeType && !selectedPipeKeys.contains(getPipeKey(pipeType));
}

/// Return the static pipe relations that also occur in selected callbacks.
///
/// Partially expanding either endpoint would expose an incomplete transfer to
/// PipeGraph while its corresponding selected endpoint remains high-level IR.
static FailureOr<llvm::DenseSet<PipeKey>>
collectSelectedPipeKeys(ModuleOp module) {
  llvm::DenseSet<PipeKey> selectedPipeKeys;
  LogicalResult result = success();
  module.walk([&](CopyOp copyOp) {
    Value pipe;
    if (isPipeReceiveCopy(copyOp)) {
      pipe = copyOp.getSrc();
    } else if (isPipeSendCopy(copyOp)) {
      pipe = copyOp.getDst();
    } else {
      return;
    }
    if (!mlir::isa<SelectedPipeSrcType, SelectedPipeDstType>(
            traceUnrealizedCasts(pipe).getType())) {
      return;
    }
    FailureOr<SelectedPipeRecords> maybeRecords = getSelectedPipeRecords(pipe);
    if (failed(maybeRecords)) {
      copyOp.emitError("cannot resolve selected PipeNet records");
      result = failure();
      return;
    }
    for (PipeRecordAttr record : maybeRecords->records.getPipes()) {
      selectedPipeKeys.insert(
          getPipeKey(record, maybeRecords->records.getPipeNetId()));
    }
  });
  if (failed(result)) {
    return failure();
  }
  return selectedPipeKeys;
}

/// Collect every replacement before expansion invalidates origin analysis.
static FailureOr<PipeTransferExpansionPlan>
buildPipeTransferExpansionPlan(ModuleOp module, ValueOriginAnalysis &analysis,
                               PipeTransferExpansionMode mode) {
  PipeTransferExpansionPlan plan;
  FailureOr<llvm::DenseSet<PipeKey>> maybeSelectedPipeKeys =
      collectSelectedPipeKeys(module);
  if (failed(maybeSelectedPipeKeys)) {
    return failure();
  }
  const llvm::DenseSet<PipeKey> &selectedPipeKeys = *maybeSelectedPipeKeys;
  module.walk([&](CreatePipeOp op) { plan.createPipes.push_back(op); });

  LogicalResult result = success();
  module.walk([&](CopyOp op) {
    if (isPipeReceiveCopy(op)) {
      if (!shouldExpandPipeValue(op.getSrc(), mode, selectedPipeKeys)) {
        return;
      }
      FailureOr<PipeTransferContract> contract =
          getPipeTransferContractForPipeValue(analysis, op.getSrc());
      if (failed(contract)) {
        op.emitError()
            << "requires a consistent transfer contract for all possible "
               "pipe values";
        result = failure();
      } else {
        FailureOr<int64_t> pipeNetId =
            getPipeNetIdForPipeValue(op, op.getSrc());
        if (failed(pipeNetId)) {
          result = failure();
        } else {
          plan.receiveCopies.push_back({op, *contract, *pipeNetId});
        }
      }
      return;
    }
    if (isPipeSendCopy(op)) {
      if (!shouldExpandPipeValue(op.getDst(), mode, selectedPipeKeys)) {
        return;
      }
      FailureOr<PipeTransferContract> contract =
          getPipeTransferContractForPipeValue(analysis, op.getDst());
      if (failed(contract)) {
        op.emitError()
            << "requires a consistent transfer contract for all possible "
               "pipe values";
        result = failure();
      } else {
        plan.sendCopies.push_back({op, *contract, std::nullopt});
      }
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
      if (mode == PipeTransferExpansionMode::All) {
        plan.unreachableReceiveWaits.push_back(waitOp);
      }
      return;
    }
    FailureOr<std::optional<CopyOp>> maybeCopyOp =
        findUniquePipeReceiveCopy(analysis, waitOp.getXf());
    if (failed(maybeCopyOp) || !*maybeCopyOp) {
      waitOp.emitError() << "untyped transfer handle wait requires every "
                            "possible source to be the same pipe receive "
                            "ttl.copy";
      result = failure();
      return;
    }
    CopyOp copyOp = **maybeCopyOp;
    if (!shouldExpandPipeValue(copyOp.getSrc(), mode, selectedPipeKeys)) {
      return;
    }
    FailureOr<int64_t> pipeNetId =
        getPipeNetIdForPipeValue(waitOp, copyOp.getSrc());
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
                           getPipeTransferContract(createPipe));
    transferByDirectCreatePipe[createPipe.getResult()] =
        transferOp.getTransfer();
  }

  for (const PipeCopyExpansion &expansion : plan.receiveCopies) {
    CopyOp copyOp = expansion.copy;
    assert(expansion.pipeNetId && "receiver expansion is missing PipeNet id");
    builder.setInsertionPoint(copyOp);
    Value transfer =
        getOrCreatePipeTransfer(builder, copyOp.getLoc(), copyOp.getSrc(),
                                expansion.contract, transferByDirectCreatePipe);
    auto postOp = PipeTransferPostOp::create(
        builder, copyOp.getLoc(),
        PipeTokenType::get(builder.getContext(), *expansion.pipeNetId),
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
    Value transfer =
        getOrCreatePipeTransfer(builder, copyOp.getLoc(), copyOp.getDst(),
                                expansion.contract, transferByDirectCreatePipe);
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
      buildPipeTransferExpansionPlan(module, analysis,
                                     PipeTransferExpansionMode::All);
  if (failed(maybePlan)) {
    return failure();
  }
  applyPipeTransferExpansionPlan(module, *maybePlan);
  return success();
}

LogicalResult expandStaticPipeTransfers(ModuleOp module,
                                        ValueOriginAnalysis &analysis) {
  FailureOr<PipeTransferExpansionPlan> maybePlan =
      buildPipeTransferExpansionPlan(
          module, analysis, PipeTransferExpansionMode::StaticPipesOnly);
  if (failed(maybePlan)) {
    return failure();
  }
  applyPipeTransferExpansionPlan(module, *maybePlan);
  return success();
}

} // namespace mlir::tt::ttl
