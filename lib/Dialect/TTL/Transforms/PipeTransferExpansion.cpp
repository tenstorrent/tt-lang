// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeTransferExpansion.h"

#include "PipeGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"
#include "llvm/ADT/DenseMap.h"
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
static PipeTransferCreateOp
createPipeTransfer(OpBuilder &builder, Location location, Value pipe,
                   PipeTransferContract contract,
                   DeviceTransferAttr deviceTransfer) {
  auto kindAttr = PipeTransferKindAttr::get(builder.getContext(),
                                            getPipeTransferKind(contract));
  return PipeTransferCreateOp::create(
      builder, location, PipeTransferType::get(builder.getContext()), pipe,
      kindAttr, deviceTransfer);
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

/// High-level pipe copy and its proven transfer contract.
struct PipeCopyExpansion {
  CopyOp copy;
  PipeTransferContract contract;
  DeviceTransferAttr deviceTransfer;
  std::optional<int64_t> pipeNetId;
};

/// Receive-handle wait and the PipeNet id used to create its typed token.
struct PipeReceiveWaitExpansion {
  WaitOp wait;
  int64_t pipeNetId = 0;
};

/// Wait-any and the PipeNet id inferred for each request operand.
struct PipeReceiveWaitAnyExpansion {
  WaitAnyOp wait;
  SmallVector<int64_t> pipeNetIds;
};

/// Operations replaced when high-level pipe copies become pipe transfer IR.
struct PipeTransferExpansionPlan {
  SmallVector<CreatePipeOp> createPipes;
  SmallVector<PipeCopyExpansion> receiveCopies;
  SmallVector<PipeCopyExpansion> sendCopies;
  SmallVector<PipeReceiveWaitExpansion> receiveWaits;
  SmallVector<PipeReceiveWaitAnyExpansion> receiveWaitAnys;
  SmallVector<WaitOp> unreachableReceiveWaits;
};

/// Return whether the selected expansion mode includes `pipe`.
static bool
shouldExpandPipeValue(Value pipe, PipeTransferExpansionMode mode,
                      const llvm::DenseSet<PipeKey> &deferredStaticPipeKeys) {
  if (mode == PipeTransferExpansionMode::All) {
    return true;
  }
  auto pipeType =
      mlir::dyn_cast<PipeType>(traceUnrealizedCasts(pipe).getType());
  return pipeType && !deferredStaticPipeKeys.contains(getPipeKey(pipeType));
}

/// Return static pipe relations that must remain with selected transfers.
///
/// A wait-any operation and its candidate receives are expanded together.
/// Static candidates therefore remain unexpanded when any candidate uses a
/// selected pipe.
static FailureOr<llvm::DenseSet<PipeKey>>
collectDeferredStaticPipeKeys(ModuleOp module, ValueOriginAnalysis &analysis) {
  llvm::DenseSet<PipeKey> deferredStaticPipeKeys;
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
      deferredStaticPipeKeys.insert(
          getPipeKey(record, maybeRecords->records.getPipeNetId()));
    }
  });
  if (failed(result)) {
    return failure();
  }

  bool changed;
  do {
    changed = false;
    module.walk([&](WaitAnyOp waitOp) {
      SmallVector<PipeKey> candidateStaticPipes;
      bool containsDeferredCandidate = false;
      for (Value request : waitOp.getRequests()) {
        FailureOr<SmallVector<CopyOp>> maybeCopyOps =
            findPipeReceiveCopies(analysis, request);
        if (failed(maybeCopyOps)) {
          waitOp.emitError()
              << "requires every request origin to be a pipe receive ttl.copy";
          result = failure();
          return;
        }
        for (CopyOp copyOp : *maybeCopyOps) {
          Type pipeType = traceUnrealizedCasts(copyOp.getSrc()).getType();
          if (mlir::isa<SelectedPipeSrcType, SelectedPipeDstType>(pipeType)) {
            containsDeferredCandidate = true;
            continue;
          }
          auto staticPipeType = mlir::dyn_cast<PipeType>(pipeType);
          assert(staticPipeType && "verified pipe receive has a pipe source");
          PipeKey pipeKey = getPipeKey(staticPipeType);
          candidateStaticPipes.push_back(pipeKey);
          containsDeferredCandidate |= deferredStaticPipeKeys.contains(pipeKey);
        }
      }
      if (!containsDeferredCandidate) {
        return;
      }
      for (const PipeKey &pipeKey : candidateStaticPipes) {
        changed |= deferredStaticPipeKeys.insert(pipeKey).second;
      }
    });
  } while (succeeded(result) && changed);
  if (failed(result)) {
    return failure();
  }
  return deferredStaticPipeKeys;
}

/// Collect every replacement before expansion invalidates origin analysis.
static FailureOr<PipeTransferExpansionPlan>
buildPipeTransferExpansionPlan(ModuleOp module, ValueOriginAnalysis &analysis,
                               PipeTransferExpansionMode mode) {
  PipeTransferExpansionPlan plan;
  FailureOr<llvm::DenseSet<PipeKey>> maybeDeferredStaticPipeKeys =
      collectDeferredStaticPipeKeys(module, analysis);
  if (failed(maybeDeferredStaticPipeKeys)) {
    return failure();
  }
  const llvm::DenseSet<PipeKey> &deferredStaticPipeKeys =
      *maybeDeferredStaticPipeKeys;
  module.walk([&](CreatePipeOp op) { plan.createPipes.push_back(op); });

  DenseMap<BlockArgument, SmallVector<Value>> operandsByFunctionArgument;
  SymbolTableCollection symbolTables;
  module.walk([&](func::CallOp call) {
    func::FuncOp callee = symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee || callee.isExternal()) {
      return;
    }
    for (auto [argument, operand] :
         llvm::zip_equal(callee.getArguments(), call.getOperands())) {
      operandsByFunctionArgument[argument].push_back(operand);
    }
  });
  auto resolveFunctionArguments =
      [&](BlockArgument argument) -> FailureOr<SmallVector<Value>> {
    auto function =
        dyn_cast_if_present<func::FuncOp>(argument.getOwner()->getParentOp());
    if (!function || argument.getOwner() != &function.getBody().front()) {
      return failure();
    }
    return operandsByFunctionArgument.lookup(argument);
  };

  struct PipeCopyFacts {
    PipeTransferContract contract;
    DeviceTransferAttr deviceTransfer;
  };
  auto collectPipeCopyFacts = [&](CopyOp op,
                                  Value pipe) -> FailureOr<PipeCopyFacts> {
    FailureOr<PipeTransferContract> contract =
        getPipeTransferContractForPipeValue(analysis, pipe);
    if (failed(contract)) {
      op.emitError() << "requires a consistent transfer contract for all "
                        "possible pipe values";
      return failure();
    }
    FailureOr<std::optional<DeviceTransferAttr>> maybeDeviceTransfer =
        findUniquePipeDeviceTransfer(analysis, pipe, resolveFunctionArguments);
    if (failed(maybeDeviceTransfer)) {
      op.emitError() << "requires every possible pipe definition to use the "
                        "same logical-device transfer";
      return failure();
    }
    return PipeCopyFacts{*contract,
                         maybeDeviceTransfer->value_or(DeviceTransferAttr())};
  };

  LogicalResult result = success();
  module.walk([&](CopyOp op) {
    if (isPipeReceiveCopy(op)) {
      if (!shouldExpandPipeValue(op.getSrc(), mode, deferredStaticPipeKeys)) {
        return;
      }
      FailureOr<PipeCopyFacts> maybeFacts =
          collectPipeCopyFacts(op, op.getSrc());
      if (failed(maybeFacts)) {
        result = failure();
        return;
      }
      FailureOr<int64_t> pipeNetId = getPipeNetIdForPipeValue(op, op.getSrc());
      if (failed(pipeNetId)) {
        result = failure();
      } else {
        plan.receiveCopies.push_back(
            {op, maybeFacts->contract, maybeFacts->deviceTransfer, *pipeNetId});
      }
      return;
    }
    if (isPipeSendCopy(op)) {
      if (!shouldExpandPipeValue(op.getDst(), mode, deferredStaticPipeKeys)) {
        return;
      }
      FailureOr<PipeCopyFacts> maybeFacts =
          collectPipeCopyFacts(op, op.getDst());
      if (failed(maybeFacts)) {
        result = failure();
      } else {
        plan.sendCopies.push_back({op, maybeFacts->contract,
                                   maybeFacts->deviceTransfer, std::nullopt});
      }
    }
  });
  if (failed(result)) {
    return failure();
  }

  module.walk([&](WaitOp waitOp) {
    if (!mlir::isa<ReceiveRequestType>(waitOp.getXf().getType())) {
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
      waitOp.emitError() << "receive request wait requires every possible "
                            "source to be the same pipe receive ttl.copy";
      result = failure();
      return;
    }
    CopyOp copyOp = **maybeCopyOp;
    if (!shouldExpandPipeValue(copyOp.getSrc(), mode, deferredStaticPipeKeys)) {
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
  module.walk([&](WaitAnyOp waitOp) {
    PipeReceiveWaitAnyExpansion expansion;
    expansion.wait = waitOp;
    llvm::DenseSet<Operation *> receiveCopies;
    for (Value request : waitOp.getRequests()) {
      FailureOr<SmallVector<CopyOp>> maybeCopyOps =
          findPipeReceiveCopies(analysis, request);
      if (failed(maybeCopyOps)) {
        waitOp.emitError()
            << "requires every request origin to be a pipe receive ttl.copy";
        result = failure();
        return;
      }
      std::optional<int64_t> candidatePipeNetId;
      for (CopyOp copyOp : *maybeCopyOps) {
        if (!receiveCopies.insert(copyOp.getOperation()).second) {
          waitOp.emitError()
              << "requires request values with disjoint pipe receive origins";
          result = failure();
          return;
        }
        if (!shouldExpandPipeValue(copyOp.getSrc(), mode,
                                   deferredStaticPipeKeys)) {
          return;
        }
        FailureOr<int64_t> pipeNetId =
            getPipeNetIdForPipeValue(waitOp, copyOp.getSrc());
        if (failed(pipeNetId)) {
          result = failure();
          return;
        }
        if (candidatePipeNetId && *candidatePipeNetId != *pipeNetId) {
          waitOp.emitError()
              << "requires each request's origins to belong to one PipeNet";
          result = failure();
          return;
        }
        candidatePipeNetId = *pipeNetId;
      }
      assert(candidatePipeNetId && "request origin set must be nonempty");
      expansion.pipeNetIds.push_back(*candidatePipeNetId);
    }
    if (expansion.pipeNetIds.size() == waitOp.getRequests().size()) {
      plan.receiveWaitAnys.push_back(std::move(expansion));
    }
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
    assert(expansion.pipeNetId && "receiver expansion is missing PipeNet id");
    builder.setInsertionPoint(copyOp);
    Value transfer = getOrCreatePipeTransfer(
        builder, copyOp.getLoc(), copyOp.getSrc(), expansion.contract,
        expansion.deviceTransfer, transferByDirectCreatePipe);
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

  for (const PipeReceiveWaitAnyExpansion &wait : plan.receiveWaitAnys) {
    WaitAnyOp waitOp = wait.wait;
    builder.setInsertionPoint(waitOp);
    SmallVector<Value> tokens;
    tokens.reserve(wait.pipeNetIds.size());
    for (auto [request, pipeNetId] :
         llvm::zip_equal(waitOp.getRequests(), wait.pipeNetIds)) {
      tokens.push_back(UnrealizedConversionCastOp::create(
                           builder, waitOp.getLoc(),
                           PipeTokenType::get(builder.getContext(), pipeNetId),
                           ValueRange{request})
                           .getResult(0));
    }
    auto internalWait = PipeTransferWaitAnyOp::create(
        builder, waitOp.getLoc(), waitOp.getReady().getType(), tokens,
        waitOp.getStart());
    waitOp.getReady().replaceAllUsesWith(internalWait.getReady());
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
