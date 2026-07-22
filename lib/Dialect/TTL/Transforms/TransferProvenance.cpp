// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYTRANSFERPROVENANCE
#include "ttlang/Dialect/TTL/Passes.h.inc"

FailureOr<PipeTransferCreateOp>
findPipeTransferCreateForTransfer(ValueOriginAnalysis &analysis,
                                  Value transfer) {
  return analysis.getOrigins(transfer).uniqueDefiningOp<PipeTransferCreateOp>();
}

namespace {

bool isReachable(DataFlowSolver &solver, Operation *operation) {
  if (!operation->getBlock()) {
    return true;
  }
  ProgramPoint *blockStart =
      solver.getProgramPointBefore(operation->getBlock());
  const auto *executable = solver.lookupState<dataflow::Executable>(blockStart);
  return !executable || executable->isLive();
}

LogicalResult verifyWait(WaitOp op, ValueOriginAnalysis &analysis) {
  const OriginSet &origins = analysis.getOrigins(op.getXf());
  if (!origins.allMatch([](Value origin) {
        return origin.getDefiningOp<CopyOp>() ||
               origin.getDefiningOp<PipeTransferSendOp>();
      })) {
    return op.emitOpError() << "expects operand to be derived from ttl.copy or "
                               "ttl.pipe_transfer.send";
  }

  bool hasPipeSend = false;
  bool hasCopy = false;
  for (Value origin : origins) {
    hasPipeSend |=
        static_cast<bool>(origin.getDefiningOp<PipeTransferSendOp>());
    hasCopy |= static_cast<bool>(origin.getDefiningOp<CopyOp>());
  }
  if (hasPipeSend && hasCopy) {
    return op.emitOpError()
           << "requires all possible sources to have the same wait semantics";
  }
  return success();
}

LogicalResult verifyPost(PipeTransferPostOp op, ValueOriginAnalysis &analysis) {
  if (!findCBReserveForPipeReceive(op.getDst())) {
    return op.emitOpError() << "requires a cb_reserve destination";
  }
  FailureOr<PipeTransferCreateOp> create =
      findPipeTransferCreateForTransfer(analysis, op.getTransfer());
  if (failed(create)) {
    return op.emitOpError()
           << "requires every possible transfer value to derive from the "
              "same ttl.pipe_transfer.create";
  }
  auto pipeType = cast<PipeType>(create->getPipe().getType());
  auto tokenType = cast<PipeTokenType>(op.getToken().getType());
  if (tokenType.getPipeNetId() != pipeType.getPipeNetId()) {
    return op.emitOpError() << "token pipeNetId must match transfer pipeNetId";
  }
  return success();
}

LogicalResult verifySend(PipeTransferSendOp op, ValueOriginAnalysis &analysis) {
  if (failed(findPipeTransferCreateForTransfer(analysis, op.getTransfer()))) {
    return op.emitOpError()
           << "requires every possible transfer value to derive from the "
              "same ttl.pipe_transfer.create";
  }
  return success();
}

LogicalResult verifyPipeWait(PipeTransferWaitOp op,
                             ValueOriginAnalysis &analysis) {
  auto waitTokenType = cast<PipeTokenType>(op.getToken().getType());
  if (!analysis.getOrigins(op.getToken()).allMatch([&](Value origin) {
        auto post = origin.getDefiningOp<PipeTransferPostOp>();
        if (!post) {
          return false;
        }
        auto postTokenType = cast<PipeTokenType>(post.getToken().getType());
        return waitTokenType.getPipeNetId() == postTokenType.getPipeNetId();
      })) {
    return op.emitOpError()
           << "requires every possible token value to derive from a "
              "ttl.pipe_transfer.post in the same PipeNet";
  }
  return success();
}

struct TTLVerifyTransferProvenancePass
    : impl::TTLVerifyTransferProvenanceBase<TTLVerifyTransferProvenancePass> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(verifyTransferProvenance(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

LogicalResult verifyTransferProvenance(ModuleOp module) {
  ValueOriginAnalysis analysis(module);
  return verifyTransferProvenance(module, analysis);
}

LogicalResult verifyTransferProvenance(ModuleOp module,
                                       ValueOriginAnalysis &analysis) {
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  if (failed(solver.initializeAndRun(module))) {
    return module.emitError("failed to analyze transfer reachability");
  }
  WalkResult result = module.walk([&](Operation *operation) {
    if (!isReachable(solver, operation)) {
      return WalkResult::advance();
    }
    LogicalResult verification =
        llvm::TypeSwitch<Operation *, LogicalResult>(operation)
            .Case<WaitOp>([&](WaitOp op) { return verifyWait(op, analysis); })
            .Case<PipeTransferPostOp>(
                [&](PipeTransferPostOp op) { return verifyPost(op, analysis); })
            .Case<PipeTransferSendOp>(
                [&](PipeTransferSendOp op) { return verifySend(op, analysis); })
            .Case<PipeTransferWaitOp>([&](PipeTransferWaitOp op) {
              return verifyPipeWait(op, analysis);
            })
            .Default([](Operation *) { return success(); });
    return failed(verification) ? WalkResult::interrupt()
                                : WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace mlir::tt::ttl
