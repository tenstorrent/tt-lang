// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/PipeTransferAnalysis.h"

#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"

namespace mlir::tt::ttl {

FailureOr<std::unique_ptr<PipeTransferIndex>>
PipeTransferIndex::create(ModuleOp module, ValueOriginAnalysis &valueOrigins) {
  std::unique_ptr<PipeTransferIndex> index(new PipeTransferIndex());
  if (failed(index->build(module, valueOrigins))) {
    return failure();
  }
  return index;
}

LogicalResult PipeTransferIndex::build(ModuleOp module,
                                       ValueOriginAnalysis &valueOrigins) {
  WalkResult result = module.walk([&](Operation *operation) {
    if (auto waitOp = mlir::dyn_cast<WaitOp>(operation)) {
      FailureOr<std::optional<CopyOp>> maybePost =
          findUniquePipeReceiveCopy(valueOrigins, waitOp.getXf());
      if (failed(maybePost)) {
        waitOp.emitOpError()
            << "requires either every possible source to be the same pipe "
               "receive ttl.copy or no source to be a pipe receive";
        return WalkResult::interrupt();
      }
      if (maybePost->has_value()) {
        Operation *post = (**maybePost).getOperation();
        receivePostByWait[operation] = post;
      }
      return WalkResult::advance();
    }

    if (auto waitOp = mlir::dyn_cast<PipeTransferWaitOp>(operation)) {
      FailureOr<SmallVector<PipeTransferPostOp>> maybePosts =
          findPipeTransferPostsForToken(valueOrigins, waitOp.getToken());
      if (failed(maybePosts)) {
        waitOp.emitOpError()
            << "requires every possible token value to derive from a "
               "ttl.pipe_transfer.post";
        return WalkResult::interrupt();
      }
      for (PipeTransferPostOp postOp : *maybePosts) {
        receivePostsByWait[operation].push_back(postOp.getOperation());
      }
      FailureOr<PipeTransferCreateOp> maybeCreate =
          findPipeTransferCreateForPosts(valueOrigins, *maybePosts);
      if (failed(maybeCreate)) {
        waitOp.emitOpError()
            << "requires all possible receive posts to derive from one "
               "ttl.pipe_transfer.create";
        return WalkResult::interrupt();
      }
      transferCreateByProtocolOp[operation] = maybeCreate->getOperation();
      return WalkResult::advance();
    }

    Value transfer;
    if (auto postOp = mlir::dyn_cast<PipeTransferPostOp>(operation)) {
      transfer = postOp.getTransfer();
    } else if (auto sendOp = mlir::dyn_cast<PipeTransferSendOp>(operation)) {
      transfer = sendOp.getTransfer();
    } else {
      return WalkResult::advance();
    }
    FailureOr<PipeTransferCreateOp> maybeCreate =
        findPipeTransferCreateForTransfer(valueOrigins, transfer);
    if (failed(maybeCreate)) {
      operation->emitOpError()
          << "requires every possible transfer value to derive from the same "
             "ttl.pipe_transfer.create";
      return WalkResult::interrupt();
    }
    transferCreateByProtocolOp[operation] = maybeCreate->getOperation();
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

std::optional<CopyOp> PipeTransferIndex::getReceivePost(WaitOp waitOp) const {
  auto postIt = receivePostByWait.find(waitOp.getOperation());
  if (postIt == receivePostByWait.end()) {
    return std::nullopt;
  }
  return mlir::cast<CopyOp>(postIt->second);
}

ArrayRef<Operation *>
PipeTransferIndex::getPossibleReceivePosts(PipeTransferWaitOp waitOp) const {
  auto postsIt = receivePostsByWait.find(waitOp.getOperation());
  assert(postsIt != receivePostsByWait.end() &&
         "internal receive wait must have a transfer index entry");
  return postsIt->second;
}

PipeTransferCreateOp
PipeTransferIndex::getTransferCreate(Operation *protocolOp) const {
  auto createIt = transferCreateByProtocolOp.find(protocolOp);
  assert(createIt != transferCreateByProtocolOp.end() &&
         "protocol operation must have a transfer index entry");
  return mlir::cast<PipeTransferCreateOp>(createIt->second);
}

} // namespace mlir::tt::ttl
