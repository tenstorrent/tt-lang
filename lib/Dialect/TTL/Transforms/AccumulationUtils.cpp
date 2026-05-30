// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

namespace mlir::tt::ttl {

bool isTensorLoopState(scf::ForOp loop, unsigned resultIndex) {
  return isa<RankedTensorType>(loop.getInitArgs()[resultIndex].getType());
}

FailureOr<TensorAccumulationMatch>
matchAdditiveTensorAccumulation(scf::ForOp loop, unsigned resultIndex) {
  auto loopResult = loop.getResult(resultIndex);
  if (!loopResult.hasOneUse()) {
    return failure();
  }

  // The final non-accumulating store identifies the externally visible
  // destination. Accumulating stores already represent user-written DFB += and
  // are handled by a separate formation rule.
  auto finalStore = dyn_cast<StoreOp>(*loopResult.getUsers().begin());
  if (!finalStore || finalStore.getAccumulate()) {
    return failure();
  }

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto add = yield.getOperand(resultIndex).getDefiningOp<AddOp>();
  if (!add || add->getBlock() != loop.getBody() ||
      !add.getResult().hasOneUse()) {
    return failure();
  }

  BlockArgument iterArg = loop.getRegionIterArgs()[resultIndex];
  Value contribution;
  if (add.getLhs() == iterArg) {
    contribution = add.getRhs();
  } else if (add.getRhs() == iterArg) {
    contribution = add.getLhs();
  } else {
    return failure();
  }
  if (contribution == iterArg) {
    return failure();
  }

  // The iter_arg may only feed the additive recurrence. Additional uses would
  // require preserving the old loop-carried tensor value independently of the
  // selected accumulation strategy.
  for (OpOperand &use : iterArg.getUses()) {
    if (use.getOwner() != add.getOperation()) {
      return failure();
    }
  }

  auto reserve = finalStore.getView().getDefiningOp<CBReserveOp>();
  if (!reserve) {
    return failure();
  }

  SmallVector<AttachCBOp> deadReserveAttachOps;
  for (OpOperand &reserveUse : reserve.getResult().getUses()) {
    Operation *owner = reserveUse.getOwner();
    if (owner == finalStore.getOperation()) {
      continue;
    }

    // Dead attach views are artifacts of earlier lowering. A live view means
    // the reservation participates in another dataflow use and cannot be owned
    // solely by the accumulation scope.
    auto attach = dyn_cast<AttachCBOp>(owner);
    if (!attach || !attach.getResult().use_empty()) {
      return failure();
    }
    deadReserveAttachOps.push_back(attach);
  }

  if (finalStore->getBlock() != loop->getBlock() ||
      reserve->getBlock() != loop->getBlock()) {
    return failure();
  }

  auto tensorType =
      cast<RankedTensorType>(loop.getInitArgs()[resultIndex].getType());
  return TensorAccumulationMatch{resultIndex,
                                 tensorType,
                                 loop.getInitArgs()[resultIndex],
                                 iterArg,
                                 yield.getOperand(resultIndex),
                                 finalStore,
                                 reserve,
                                 add,
                                 contribution,
                                 deadReserveAttachOps};
}

} // namespace mlir::tt::ttl
