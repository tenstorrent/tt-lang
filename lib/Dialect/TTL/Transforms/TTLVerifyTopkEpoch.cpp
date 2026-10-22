// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Verify TopK Epoch
//===----------------------------------------------------------------------===//
//
// A fused or rank-stamped TopK representation lives in dataflow buffers, not
// in one destination-register section. This pass rejects a section that packs
// and unpacks that representation, a fused stage that does not stay on a
// fused-key buffer, a sort order that changes inside an epoch, and any TopK
// stage left outside a register section.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYTOPKEPOCH
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

bool isTopkStage(Operation *op) {
  return isa<TileTopkLocalSortOp, TileTopkMergeOp, TileTopkRebuildOp>(op);
}

bool isTopkHelper(Operation *op) {
  return isa<TileTopkFuseOp, TileTopkDefuseOp, TileTopkStampLocalPositionsOp,
             TileTopkStripRankTagsOp, TileTopkCanonicalizeNegzeroValuesOp,
             TileTopkUint16MoveDestTileToPackHalfOp>(op);
}

BindCBOp bindOf(Value storage) {
  Value cb;
  if (storage && isa<CircularBufferType>(storage.getType())) {
    cb = storage;
  } else if (Value attached = getAttachedCB(storage)) {
    cb = attached;
  } else if (Operation *acquire = findCBAcquireOp(storage)) {
    if (auto wait = dyn_cast<CBWaitOp>(acquire)) {
      cb = wait.getCb();
    } else if (auto reserve = dyn_cast<CBReserveOp>(acquire)) {
      cb = reserve.getCb();
    }
  }
  return cb ? cb.getDefiningOp<BindCBOp>() : BindCBOp();
}

TopkPayload payloadOf(BindCBOp bind) {
  if (!bind) {
    return TopkPayload::Plain;
  }
  auto attr = bind->getAttrOfType<TopkPayloadAttr>(kTopkPayloadAttrName);
  return attr ? attr.getValue() : TopkPayload::Plain;
}

TopkPayload payloadOf(Value storage) { return payloadOf(bindOf(storage)); }

std::optional<TopkOrder> orderOf(BindCBOp bind) {
  if (!bind) {
    return std::nullopt;
  }
  auto attr = bind->getAttrOfType<TopkOrderAttr>(kTopkOrderAttrName);
  return attr ? std::optional<TopkOrder>(attr.getValue()) : std::nullopt;
}

Value sourceStorage(Operation *op) {
  if (auto copy = dyn_cast<CopyTileOp>(op)) {
    return copy.getSrc();
  }
  if (auto transpose = dyn_cast<TileTransposeOp>(op)) {
    return transpose.getInput();
  }
  return Value();
}

TopkPayload sourcePayload(Operation *op) {
  return payloadOf(sourceStorage(op));
}

std::optional<TopkOrder> orderOf(Operation *op) {
  return TypeSwitch<Operation *, std::optional<TopkOrder>>(op)
      .Case<TileTopkLocalSortOp, TileTopkMergeOp, TileTopkRebuildOp,
            TileTopkFuseOp, TileTopkDefuseOp, TileTopkStampLocalPositionsOp>(
          [](auto typed) { return std::optional<TopkOrder>(typed.getOrder()); })
      .Default([](Operation *) { return std::nullopt; });
}

struct SectionOps {
  SmallVector<Operation *> stages;
  SmallVector<Operation *> helpers;
  /// Stages and helpers that carry `order`, in program order.
  SmallVector<Operation *> orderCarriers;
  SmallVector<Operation *> movement;
  SmallVector<TileStoreOp> stores;
};

void collect(Operation *op, SectionOps &ops) {
  if (isa<DstSectionOp, TileRegsAcquireOp>(op)) {
    return;
  }
  if (isTopkStage(op) || isTopkHelper(op)) {
    if (orderOf(op)) {
      ops.orderCarriers.push_back(op);
    }
  }
  if (isTopkStage(op)) {
    ops.stages.push_back(op);
  } else if (isTopkHelper(op)) {
    ops.helpers.push_back(op);
  } else if (isa<CopyTileOp, TileTransposeOp>(op)) {
    ops.movement.push_back(op);
  } else if (auto store = dyn_cast<TileStoreOp>(op)) {
    ops.stores.push_back(store);
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &inner : block) {
        collect(&inner, ops);
      }
    }
  }
}

bool stageFused(Operation *stage) {
  return TypeSwitch<Operation *, bool>(stage)
      .Case<TileTopkLocalSortOp, TileTopkMergeOp, TileTopkRebuildOp>(
          [](auto op) { return op.getFused(); });
}

bool stageRankStamped(Operation *stage) {
  return TypeSwitch<Operation *, bool>(stage)
      .Case<TileTopkLocalSortOp, TileTopkMergeOp, TileTopkRebuildOp>(
          [](auto op) { return op.getRankStamped(); });
}

bool stageStable(Operation *stage) {
  return TypeSwitch<Operation *, bool>(stage)
      .Case<TileTopkLocalSortOp, TileTopkMergeOp, TileTopkRebuildOp>(
          [](auto op) { return op.getStableSort(); });
}

LogicalResult checkPayloadAgreement(ArrayRef<Operation *> movement,
                                    TopkPayload expected, StringRef what) {
  for (Operation *op : movement) {
    if (sourcePayload(op) != expected) {
      return op->emitOpError() << what << " must read "
                               << stringifyTopkPayload(expected) << " tiles";
    }
  }
  return success();
}

LogicalResult checkStoreAgreement(ArrayRef<TileStoreOp> stores,
                                  TopkPayload expected, StringRef what) {
  for (TileStoreOp store : stores) {
    if (payloadOf(store.getView()) != expected) {
      return store.emitOpError() << what << " must store "
                                 << stringifyTopkPayload(expected) << " tiles";
    }
  }
  return success();
}

/// Every packed buffer a section reads or stores must carry the section's
/// order. Plain buffers carry no order and are not checked.
LogicalResult verifyBufferOrderAgreement(const SectionOps &ops,
                                         TopkOrder sectionOrder) {
  auto check = [&](Operation *op, Value storage,
                   StringRef verb) -> LogicalResult {
    BindCBOp bind = bindOf(storage);
    if (payloadOf(bind) == TopkPayload::Plain) {
      return success();
    }
    std::optional<TopkOrder> bufferOrder = orderOf(bind);
    if (bufferOrder && *bufferOrder == sectionOrder) {
      return success();
    }
    return op->emitOpError()
           << verb << " a " << stringifyTopkPayload(payloadOf(bind))
           << " buffer whose order is not " << stringifyTopkOrder(sectionOrder);
  };
  for (Operation *op : ops.movement) {
    if (failed(check(op, sourceStorage(op), "reads"))) {
      return failure();
    }
  }
  for (TileStoreOp store : ops.stores) {
    if (failed(check(store, store.getView(), "stores into"))) {
      return failure();
    }
  }
  return success();
}

LogicalResult verifySection(SectionOps ops, DenseSet<Operation *> &covered) {
  for (Operation *stage : ops.stages) {
    covered.insert(stage);
  }
  for (Operation *helper : ops.helpers) {
    covered.insert(helper);
  }
  if (ops.stages.empty() && ops.helpers.empty()) {
    return success();
  }

  bool fused = false;
  bool rankStamped = false;
  bool stableSort = false;
  bool sawStage = false;
  bool hasStableLocalSort = false;
  for (Operation *stage : ops.stages) {
    bool stageIsFused = stageFused(stage);
    bool stageIsRanked = stageRankStamped(stage);
    bool stageIsStable = stageStable(stage);
    if (!sawStage) {
      fused = stageIsFused;
      rankStamped = stageIsRanked;
      stableSort = stageIsStable;
      sawStage = true;
    } else if (stageIsFused != fused || stageIsRanked != rankStamped ||
               stageIsStable != stableSort) {
      return stage->emitOpError(
          "TopK stages in one section must agree on fused, rank_stamped, and "
          "stable_sort");
    }
    if (stageIsStable && isa<TileTopkLocalSortOp>(stage)) {
      hasStableLocalSort = true;
    }
  }

  std::optional<TopkOrder> sectionOrder;
  for (Operation *op : ops.orderCarriers) {
    std::optional<TopkOrder> order = orderOf(op);
    if (!sectionOrder) {
      sectionOrder = order;
    } else if (*order != *sectionOrder) {
      return op->emitOpError(
          "TopK operations in one section must agree on order");
    }
  }
  if (sectionOrder && failed(verifyBufferOrderAgreement(ops, *sectionOrder))) {
    return failure();
  }

  bool hasFuse = false;
  bool hasDefuse = false;
  bool hasStamp = false;
  bool hasStrip = false;
  bool hasCanonicalize = false;
  for (Operation *helper : ops.helpers) {
    hasFuse |= isa<TileTopkFuseOp>(helper);
    hasDefuse |= isa<TileTopkDefuseOp>(helper);
    hasStamp |= isa<TileTopkStampLocalPositionsOp>(helper);
    hasStrip |= isa<TileTopkStripRankTagsOp>(helper);
    hasCanonicalize |= isa<TileTopkCanonicalizeNegzeroValuesOp>(helper);
  }

  if (hasFuse && hasDefuse) {
    return ops.helpers.front()->emitOpError(
        "topk_fuse and topk_defuse cannot share a destination-register "
        "section");
  }
  if (hasStamp && hasStrip) {
    return ops.helpers.front()->emitOpError(
        "topk_stamp_local_positions and topk_strip_rank_tags cannot share a "
        "destination-register section");
  }
  if ((hasFuse || hasDefuse) && (hasStamp || hasStrip)) {
    return ops.helpers.front()->emitOpError(
        "fused and rank-stamped helpers cannot share a destination-register "
        "section");
  }
  if (hasCanonicalize && !hasStableLocalSort) {
    return ops.helpers.front()->emitOpError(
        "canonicalize_negzero requires a stable local sort in the same "
        "section");
  }
  if (hasDefuse && fused) {
    return ops.stages.front()->emitOpError(
        "a fused TopK stage cannot share its section with topk_defuse");
  }
  if (hasStrip && rankStamped) {
    return ops.stages.front()->emitOpError(
        "a rank-stamped TopK stage cannot share its section with "
        "topk_strip_rank_tags");
  }

  if (hasFuse && failed(checkStoreAgreement(ops.stores, TopkPayload::FusedKeys,
                                            "topk_fuse"))) {
    return failure();
  }
  if (hasDefuse) {
    if (failed(checkPayloadAgreement(ops.movement, TopkPayload::FusedKeys,
                                     "topk_defuse")) ||
        failed(checkStoreAgreement(ops.stores, TopkPayload::Plain,
                                   "topk_defuse"))) {
      return failure();
    }
  }
  if (hasStamp &&
      failed(checkStoreAgreement(ops.stores, TopkPayload::RankStamped,
                                 "topk_stamp_local_positions"))) {
    return failure();
  }
  if (hasStrip) {
    if (failed(checkPayloadAgreement(ops.movement, TopkPayload::RankStamped,
                                     "topk_strip_rank_tags")) ||
        failed(checkStoreAgreement(ops.stores, TopkPayload::Plain,
                                   "topk_strip_rank_tags"))) {
      return failure();
    }
  }

  if (fused && !hasFuse) {
    if (ops.movement.empty() ||
        failed(checkPayloadAgreement(ops.movement, TopkPayload::FusedKeys,
                                     "a fused TopK stage"))) {
      Operation *stage = ops.stages.front();
      if (ops.movement.empty()) {
        return stage->emitOpError(
            "a fused TopK stage must read fused keys or share its section "
            "with topk_fuse");
      }
      return failure();
    }
    if (failed(checkStoreAgreement(ops.stores, TopkPayload::FusedKeys,
                                   "a fused TopK stage"))) {
      return failure();
    }
  }
  if (rankStamped && !hasStamp) {
    if (ops.movement.empty() ||
        failed(checkPayloadAgreement(ops.movement, TopkPayload::RankStamped,
                                     "a rank-stamped TopK stage"))) {
      if (ops.movement.empty()) {
        return ops.stages.front()->emitOpError(
            "a rank-stamped TopK stage must read rank-stamped tiles or share "
            "its section with topk_stamp_local_positions");
      }
      return failure();
    }
    if (failed(checkStoreAgreement(ops.stores, TopkPayload::RankStamped,
                                   "a rank-stamped TopK stage"))) {
      return failure();
    }
  }
  return success();
}

void collectBlock(Block &block, SectionOps &ops) {
  for (Operation &op : block) {
    collect(&op, ops);
  }
}

LogicalResult verifyAcquire(TileRegsAcquireOp acquire,
                            DenseSet<Operation *> &covered) {
  SectionOps ops;
  Block *block = acquire->getBlock();
  for (auto it = std::next(acquire->getIterator()); it != block->end(); ++it) {
    if (isa<TileRegsReleaseOp>(&*it)) {
      break;
    }
    collect(&*it, ops);
  }
  return verifySection(std::move(ops), covered);
}

struct TTLVerifyTopkEpochPass
    : impl::TTLVerifyTopkEpochBase<TTLVerifyTopkEpochPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DenseSet<Operation *> covered;
    auto interrupt = [&](LogicalResult result) {
      return failed(result) ? WalkResult::interrupt() : WalkResult::advance();
    };
    WalkResult buffers = func.walk([&](BindCBOp bind) {
      bool packed = payloadOf(bind) != TopkPayload::Plain;
      if (packed && !orderOf(bind)) {
        bind.emitOpError() << "a " << stringifyTopkPayload(payloadOf(bind))
                           << " buffer must carry " << kTopkOrderAttrName;
        return WalkResult::interrupt();
      }
      if (!packed && orderOf(bind)) {
        bind.emitOpError() << kTopkOrderAttrName
                           << " applies only to a fused_keys or rank_stamped "
                              "buffer";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (buffers.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    if (func.walk([&](DstSectionOp section) {
              SectionOps ops;
              collectBlock(section.getBody().front(), ops);
              return interrupt(verifySection(std::move(ops), covered));
            })
            .wasInterrupted() ||
        func.walk([&](TileRegsAcquireOp acquire) {
              return interrupt(verifyAcquire(acquire, covered));
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    WalkResult uncovered = func.walk([&](Operation *op) {
      if ((isTopkStage(op) || isTopkHelper(op)) && !covered.contains(op)) {
        op->emitOpError(
            "TopK operation must be inside a destination-register section");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (uncovered.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
