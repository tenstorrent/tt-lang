// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeTransportDFBAnalysis.h"

#include "DFBAcquireReleaseAnalysis.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace mlir::tt::ttl {
namespace {

/// Return whether `operation` is nested in `loop`.
static bool isInsideLoop(scf::ForOp loop, Operation *operation) {
  return loop->isProperAncestor(operation);
}

/// Return whether `value` is independent of the candidate loop induction.
static bool isLoopInvariant(Value value, scf::ForOp loop) {
  if (loop.isDefinedOutsideOfLoop(value)) {
    return true;
  }
  auto result = dyn_cast<OpResult>(value);
  if (!result || !isPure(result.getOwner())) {
    return false;
  }
  return llvm::all_of(result.getOwner()->getOperands(), [&](Value operand) {
    return isLoopInvariant(operand, loop);
  });
}

/// Return whether `slice` enumerates `blockSpan` DFB blocks from `loop`.
static bool isContiguousLoopSlice(TensorSliceOp slice, scf::ForOp loop,
                                  CircularBufferType dfbType,
                                  int64_t blockSpan) {
  auto sliceType = cast<RankedTensorType>(slice.getResult().getType());
  SmallVector<int64_t> expectedShape(dfbType.getShape());
  if (expectedShape.empty()) {
    return false;
  }
  std::optional<int64_t> groupedLastDimension =
      llvm::checkedMul(expectedShape.back(), blockSpan);
  if (!groupedLastDimension) {
    return false;
  }
  expectedShape.back() = *groupedLastDimension;
  if (sliceType.getShape() != ArrayRef<int64_t>(expectedShape)) {
    return false;
  }
  // Grouping increases the loop step without rewriting the slice indices.
  // A direct induction-variable index therefore names consecutive blocks only
  // when the varying dimension contains one tile per block.
  if (dfbType.getShape().back() != 1 ||
      llvm::any_of(dfbType.getShape().drop_back(),
                   [](int64_t dimension) { return dimension != 1; })) {
    return false;
  }
  ValueRange indices = slice.getIndices();
  if (indices.empty() || indices.back() != loop.getInductionVar()) {
    return false;
  }
  return llvm::all_of(indices.drop_back(), [&](Value index) {
    return isLoopInvariant(index, loop);
  });
}

/// Return whether one lifecycle operation advances `expectedTiles`.
static bool hasExpectedTileCount(Operation *operation, int64_t expectedTiles,
                                 int64_t defaultTiles) {
  auto numTiles = operation->getAttrOfType<IntegerAttr>("num_tiles");
  return numTiles ? numTiles.getInt() == expectedTiles
                  : defaultTiles == expectedTiles;
}

} // namespace

bool hasOnlyPipeTransportLoopUses(scf::ForOp loop, Value dfb) {
  return llvm::all_of(dfb.getUsers(), [&](Operation *operation) {
    return isInsideLoop(loop, operation);
  });
}

bool hasPrivatePipeTransportDFBViews(const PipeTransportDFBUse &dfbUse,
                                     const PipeGraph &pipeGraph) {
  CBReserveOp reserveOp = dfbUse.reserves.front();
  CBWaitOp waitOp = dfbUse.waits.front();
  Value reservedView = reserveOp.getResult();
  Value waitedView = waitOp.getResult();
  SmallVector<Operation *> reservedViewUsers;
  SmallVector<Operation *> waitedViewUsers;

  auto collectViewUsers = [&](Value view,
                              SmallVectorImpl<Operation *> &terminalUsers) {
    for (Operation *user : view.getUsers()) {
      auto attach = dyn_cast<AttachCBOp>(user);
      if (!attach || attach.getTensor() != view) {
        terminalUsers.push_back(user);
        continue;
      }
      llvm::append_range(terminalUsers, attach.getResult().getUsers());
    }
  };
  collectViewUsers(reservedView, reservedViewUsers);
  collectViewUsers(waitedView, waitedViewUsers);
  if (!waitedViewUsers.empty()) {
    return false;
  }

  for (AttachCBOp attach : dfbUse.attaches) {
    if (attach.getTensor() != reservedView &&
        attach.getTensor() != waitedView) {
      return false;
    }
  }

  if (dfbUse.role == PipeTransportDFBRole::Source) {
    return reservedViewUsers.empty();
  }
  if (reservedViewUsers.size() != 1) {
    return false;
  }

  Operation *user = reservedViewUsers.front();
  ArrayRef<PipeTransferNodeId> transferNodes =
      pipeGraph.getPipeTransferNodeIdsForProtocolOp(user);
  return isa<PipeTransferPostOp>(user) &&
         llvm::is_contained(transferNodes, dfbUse.transferNode);
}

FailureOr<PipeTransportDFBUse>
analyzePipeTransportDFBUse(scf::ForOp loop, Value dfb,
                           PipeTransportDFBRole role,
                           PipeTransferNodeId transferNode,
                           const PipeGraph &pipeGraph, std::string &reason) {
  PipeTransportDFBUse dfbUse;
  dfbUse.dfb = dfb;
  dfbUse.bind = dfb.getDefiningOp<BindCBOp>();
  dfbUse.role = role;
  dfbUse.transferNode = transferNode;
  if (!dfbUse.bind) {
    reason = "transport DFB is not defined by ttl.bind_cb";
    return failure();
  }

  loop.walk([&](Operation *operation) {
    if (auto reserve = dyn_cast<CBReserveOp>(operation);
        reserve && reserve.getCb() == dfb) {
      dfbUse.reserves.push_back(reserve);
    } else if (auto push = dyn_cast<CBPushOp>(operation);
               push && push.getCb() == dfb) {
      dfbUse.pushes.push_back(push);
    } else if (auto wait = dyn_cast<CBWaitOp>(operation);
               wait && wait.getCb() == dfb) {
      dfbUse.waits.push_back(wait);
    } else if (auto pop = dyn_cast<CBPopOp>(operation);
               pop && pop.getCb() == dfb) {
      dfbUse.pops.push_back(pop);
    } else if (auto attach = dyn_cast<AttachCBOp>(operation);
               attach && attach.getCb() == dfb) {
      dfbUse.attaches.push_back(attach);
    }
  });

  if (dfbUse.reserves.size() != 1 || dfbUse.pushes.size() != 1 ||
      dfbUse.waits.size() != 1 || dfbUse.pops.size() != 1) {
    reason = "transport DFB requires one reserve/push and one wait/pop";
    return failure();
  }

  auto dfbType = cast<CircularBufferType>(dfb.getType());
  int64_t blockSpan = pipeGraph.getPipeTransferNode(transferNode).blockSpan;
  std::optional<int64_t> expectedTiles =
      llvm::checkedMul(dfbType.getElementsPerBlock(), blockSpan);
  if (!expectedTiles ||
      !hasExpectedTileCount(dfbUse.reserves.front(), *expectedTiles,
                            dfbType.getElementsPerBlock()) ||
      !hasExpectedTileCount(dfbUse.pushes.front(), *expectedTiles,
                            dfbType.getElementsPerBlock()) ||
      !hasExpectedTileCount(dfbUse.waits.front(), *expectedTiles,
                            dfbType.getElementsPerBlock()) ||
      !hasExpectedTileCount(dfbUse.pops.front(), *expectedTiles,
                            dfbType.getElementsPerBlock())) {
    reason = "transport DFB lifecycle does not advance one transfer group";
    return failure();
  }

  ArrayRef<Operation *> reserveOwners =
      pipeGraph.getDFBAcquireReleaseIndex(dfbUse.pushes.front().getOperation())
          .getReleaseIntervalOwners(dfbUse.pushes.front().getOperation());
  ArrayRef<Operation *> waitOwners =
      pipeGraph.getDFBAcquireReleaseIndex(dfbUse.pops.front().getOperation())
          .getReleaseIntervalOwners(dfbUse.pops.front().getOperation());
  if (reserveOwners.size() != 1 ||
      reserveOwners.front() != dfbUse.reserves.front().getOperation() ||
      waitOwners.size() != 1 ||
      waitOwners.front() != dfbUse.waits.front().getOperation()) {
    reason = "transport DFB releases do not have unique acquire owners";
    return failure();
  }

  SmallVector<CopyOp> tensorCopies;
  for (OpOperand &use : dfb.getUses()) {
    Operation *operation = use.getOwner();
    if (!isInsideLoop(loop, operation)) {
      continue;
    }
    if (auto copy = dyn_cast<CopyOp>(operation)) {
      bool expectedDirection = role == PipeTransportDFBRole::Source
                                   ? copy.getDst() == dfb
                                   : copy.getSrc() == dfb;
      if (expectedDirection) {
        tensorCopies.push_back(copy);
        continue;
      }
    }
    if (isa<CBReserveOp, CBPushOp, CBWaitOp, CBPopOp, AttachCBOp>(operation)) {
      continue;
    }
    if (role == PipeTransportDFBRole::Source) {
      ArrayRef<PipeTransferNodeId> transferNodes =
          pipeGraph.getPipeTransferNodeIdsForProtocolOp(operation);
      if (isa<PipeTransferSendOp>(operation) &&
          llvm::is_contained(transferNodes, transferNode)) {
        continue;
      }
    }
    reason = "transport DFB has an unsupported direct use";
    return failure();
  }

  if (tensorCopies.size() != 1) {
    reason = "transport DFB requires one tensor copy";
    return failure();
  }
  dfbUse.tensorCopy = tensorCopies.front();
  Value tensorValue = role == PipeTransportDFBRole::Source
                          ? dfbUse.tensorCopy.getSrc()
                          : dfbUse.tensorCopy.getDst();
  dfbUse.tensorSlice = tensorValue.getDefiningOp<TensorSliceOp>();
  if (!dfbUse.tensorSlice ||
      !isContiguousLoopSlice(dfbUse.tensorSlice, loop, dfbType, blockSpan)) {
    reason = "tensor copy is not a contiguous loop-indexed DFB block";
    return failure();
  }
  if (!dfbUse.tensorCopy.getXf().hasOneUse() ||
      !isa<WaitOp>(*dfbUse.tensorCopy.getXf().getUsers().begin())) {
    reason = "tensor copy completion is observed outside one direct wait";
    return failure();
  }

  return dfbUse;
}

FailureOr<PipeTransportDFBOwnership>
analyzePipeTransportDFBOwnership(const PipeTransferNode &transferNode,
                                 const PipeGraph &pipeGraph,
                                 std::string &reason) {
  if (transferNode.blockSpan <= 1 ||
      transferNode.transferContract != PipeTransferContract::PointToPoint ||
      transferNode.receiverEndpoints.size() != 1) {
    reason = "transport storage requires one grouped point-to-point receiver";
    return failure();
  }

  PipeReceiverEndpointId endpointId = transferNode.receiverEndpoints.front();
  const PipeReceiverEndpoint &endpoint =
      pipeGraph.getPipeReceiverEndpoint(endpointId);
  if (transferNode.pipe.srcX == endpoint.receiver.x &&
      transferNode.pipe.srcY == endpoint.receiver.y) {
    reason = "transport storage cannot alias source and destination nodes";
    return failure();
  }
  if (!endpoint.receiverDFBInfo.hasStaticTileOffset ||
      endpoint.receiverDFBInfo.staticTileOffset != 0) {
    reason = "transport storage requires a zero receiver tile offset";
    return failure();
  }

  auto sendOp = dyn_cast<PipeTransferSendOp>(transferNode.sendOp);
  auto postOp = dyn_cast<PipeTransferPostOp>(endpoint.postOp);
  if (!sendOp || !postOp) {
    reason = "transport storage requires one send and one receiver post";
    return failure();
  }
  auto sendLoop = sendOp->getParentOfType<scf::ForOp>();
  auto postLoop = postOp->getParentOfType<scf::ForOp>();
  if (!sendLoop || sendLoop != postLoop || !sendLoop.getStaticTripCount()) {
    reason = "transport storage requires one shared static transfer loop";
    return failure();
  }

  Value receiverDFB = getAttachedCB(postOp.getDst());
  if (!receiverDFB || receiverDFB == sendOp.getSrc()) {
    reason = "transport storage requires distinct source and destination DFBs";
    return failure();
  }

  FailureOr<PipeTransportDFBUse> source = analyzePipeTransportDFBUse(
      sendLoop, sendOp.getSrc(), PipeTransportDFBRole::Source, transferNode.id,
      pipeGraph, reason);
  if (failed(source) || !hasPrivatePipeTransportDFBViews(*source, pipeGraph)) {
    if (succeeded(source)) {
      reason = "source DFB acquired views escape their transport role";
    }
    return failure();
  }

  FailureOr<PipeTransportDFBUse> destination = analyzePipeTransportDFBUse(
      sendLoop, receiverDFB, PipeTransportDFBRole::Destination, transferNode.id,
      pipeGraph, reason);
  if (failed(destination) ||
      !hasPrivatePipeTransportDFBViews(*destination, pipeGraph)) {
    if (succeeded(destination)) {
      reason = "destination DFB acquired views escape their transport role";
    }
    return failure();
  }

  return PipeTransportDFBOwnership{transferNode.id, endpointId, sendLoop,
                                   std::move(*source), std::move(*destination)};
}

} // namespace mlir::tt::ttl
