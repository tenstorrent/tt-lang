// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"

#include "PipeGraph.h"
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

FailureOr<std::optional<CopyOp>>
findUniquePipeReceiveCopy(ValueOriginAnalysis &analysis, Value value) {
  return analysis.getOrigins(value).uniqueMapped<std::optional<CopyOp>>(
      [](Value origin) -> FailureOr<std::optional<CopyOp>> {
        if (auto copyOp = origin.getDefiningOp<CopyOp>()) {
          return isPipeReceiveCopy(copyOp) ? std::optional<CopyOp>(copyOp)
                                           : std::optional<CopyOp>();
        }
        if (origin.getDefiningOp<PipeTransferSendOp>()) {
          return std::optional<CopyOp>();
        }
        return failure();
      });
}

FailureOr<SmallVector<PipeTransferPostOp>>
findPipeTransferPostsForToken(ValueOriginAnalysis &analysis, Value token) {
  SmallVector<PipeTransferPostOp> posts;
  for (Value origin : analysis.getOrigins(token)) {
    auto postOp = origin.getDefiningOp<PipeTransferPostOp>();
    if (!postOp) {
      return failure();
    }
    posts.push_back(postOp);
  }
  if (posts.empty()) {
    return failure();
  }
  return posts;
}

FailureOr<PipeTransferCreateOp>
findPipeTransferCreateForPosts(ValueOriginAnalysis &analysis,
                               ArrayRef<PipeTransferPostOp> posts) {
  std::optional<PipeTransferCreateOp> commonCreate;
  for (PipeTransferPostOp postOp : posts) {
    FailureOr<PipeTransferCreateOp> maybeCreate =
        findPipeTransferCreateForTransfer(analysis, postOp.getTransfer());
    if (failed(maybeCreate) ||
        (commonCreate &&
         commonCreate->getOperation() != maybeCreate->getOperation())) {
      return failure();
    }
    commonCreate = *maybeCreate;
  }
  if (!commonCreate) {
    return failure();
  }
  return *commonCreate;
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
  if (failed(findUniquePipeReceiveCopy(analysis, op.getXf()))) {
    return op.emitOpError()
           << "requires either every possible source to be the same pipe "
              "receive ttl.copy or no source to be a pipe receive";
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
  FailureOr<PipeReference> pipeRef = getPipeReference(op, create->getPipe());
  if (failed(pipeRef)) {
    return failure();
  }
  auto tokenType = cast<PipeTokenType>(op.getToken().getType());
  if (tokenType.getPipeNetId() != pipeRef->getPipeNetId()) {
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
  FailureOr<SmallVector<PipeTransferPostOp>> maybePosts =
      findPipeTransferPostsForToken(analysis, op.getToken());
  if (failed(maybePosts) ||
      llvm::any_of(*maybePosts, [&](PipeTransferPostOp post) {
        auto postTokenType = cast<PipeTokenType>(post.getToken().getType());
        return waitTokenType.getPipeNetId() != postTokenType.getPipeNetId();
      })) {
    return op.emitOpError()
           << "requires every possible token value to derive from a "
              "ttl.pipe_transfer.post in the same PipeNet";
  }
  if (failed(findPipeTransferCreateForPosts(analysis, *maybePosts))) {
    return op.emitOpError()
           << "requires all possible receive posts to derive from one "
              "ttl.pipe_transfer.create";
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
