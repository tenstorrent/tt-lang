// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Coalesce DFB Acquires
//===----------------------------------------------------------------------===//
//
// Rewrites N consecutive same-DFB acquires + N matching releases into the
// canonical tt-metal cumulative-wait pattern:
//
//     cb_wait_front(cb, N*k);
//     copy_tile(cb, /*src_idx=*/0,    dst);
//     copy_tile(cb, /*src_idx=*/k,    dst);
//     ...
//     cb_pop_front(cb, N*k);
//
// At the IR level:
//
//     %t1 = ttl.cb_wait %cb            %g  = ttl.cb_wait %cb {num_tiles=N*k}
//     %t2 = ttl.cb_wait %cb            %t1 = extract_slice %g [0, 0]   [1,k]
//     ...                              %t2 = extract_slice %g [0, k]   [1,k]
//     ttl.cb_pop %cb                   ...
//     ttl.cb_pop %cb                   ttl.cb_pop %cb {num_tiles=N*k}
//
// `addSliceOffset` already folds the `extract_slice` offsets into the
// per-tile `src_idx` / `dst_idx` at lowering, so no lowering changes are
// needed. Symmetric for `cb_reserve` / `cb_push`.
//
// See `docs/development/DFBManagement.md`.
//===----------------------------------------------------------------------===//

#include "DFBAcquireReleaseAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-coalesce-dfb-acquires"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLCOALESCEDFBACQUIRES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static RankedTensorType buildCoalescedType(RankedTensorType unitTy,
                                           int64_t totalTiles) {
  auto shape = unitTy.getShape();
  assert(shape.size() == 2 && shape[0] == 1 &&
         "coalesce expects rank-2 acquire with leading 1");
  return RankedTensorType::get({1, totalTiles}, unitTy.getElementType());
}

// Slice into the coalesced result that recovers the i-th member's
// original `<1, k>` view, used as the replacement value for the i-th
// erased acquire.
static tensor::ExtractSliceOp
createPerBlockSlice(OpBuilder &builder, Location loc, Value coalescedResult,
                    RankedTensorType unitTy, int64_t blockIdx, int64_t k) {
  SmallVector<OpFoldResult, 2> offsets = {builder.getIndexAttr(0),
                                          builder.getIndexAttr(blockIdx * k)};
  SmallVector<OpFoldResult, 2> sizes = {builder.getIndexAttr(1),
                                        builder.getIndexAttr(k)};
  SmallVector<OpFoldResult, 2> strides = {builder.getIndexAttr(1),
                                          builder.getIndexAttr(1)};
  return tensor::ExtractSliceOp::create(builder, loc, unitTy, coalescedResult,
                                        offsets, sizes, strides);
}

// The coalescable run anchored at `start`; see
// `collectCoalescableAcquireRun` and "DFB Acquire Coalescing" in
// `docs/development/DFBManagement.md` for the correctness argument.
template <typename AcquireOp>
static SmallVector<AcquireOp> detectGroup(AcquireOp start) {
  SmallVector<AcquireOp> group;
  for (Operation *member : collectCoalescableAcquireRun(start.getOperation())) {
    group.push_back(cast<AcquireOp>(member));
  }
  return group;
}

// The `count` releases on `cb` that the coalesced release will replace,
// in op order. Empty result means the coalesce cannot proceed: either too
// few releases are present, or one of them is already coalesced.
template <typename ReleaseOp>
static SmallVector<ReleaseOp> collectReleases(Operation *start, Value cb,
                                              size_t count) {
  SmallVector<ReleaseOp> releases;
  for (Operation *op = start; op != nullptr; op = op->getNextNode()) {
    auto release = dyn_cast<ReleaseOp>(op);
    if (!release || release.getCb() != cb) {
      continue;
    }
    if (release.getNumTiles().has_value()) {
      return {};
    }
    releases.push_back(release);
    if (releases.size() == count) {
      return releases;
    }
  }
  return {};
}

template <typename AcquireOp, typename ReleaseOp>
static bool tryCoalesceGroup(SmallVectorImpl<AcquireOp> &group,
                             OpBuilder &builder) {
  AcquireOp leader = group.front();
  Value cb = leader.getCb();
  auto unitTy = cast<RankedTensorType>(leader.getResult().getType());
  if (unitTy.getRank() != 2 || unitTy.getShape()[0] != 1) {
    return false;
  }
  int64_t k = unitTy.getShape()[1];
  int64_t N = static_cast<int64_t>(group.size());
  int64_t totalTiles = N * k;

  SmallVector<ReleaseOp> releases =
      collectReleases<ReleaseOp>(group.back()->getNextNode(), cb, group.size());
  if (releases.empty()) {
    return false;
  }

  builder.setInsertionPoint(leader);
  Location loc = leader.getLoc();
  RankedTensorType coalescedTy = buildCoalescedType(unitTy, totalTiles);
  IntegerAttr numTilesAttr = builder.getI64IntegerAttr(totalTiles);
  AcquireOp coalesced =
      AcquireOp::create(builder, loc, coalescedTy, cb, numTilesAttr);

  for (size_t i = 0; i < group.size(); ++i) {
    AcquireOp old = group[i];
    builder.setInsertionPoint(old);
    Location oldLoc = old.getLoc();
    auto slice = createPerBlockSlice(builder, oldLoc, coalesced.getResult(),
                                     unitTy, static_cast<int64_t>(i), k);
    old.getResult().replaceAllUsesWith(slice.getResult());
    old.erase();
  }

  releases.back()->setAttr("num_tiles", numTilesAttr);
  for (size_t i = 0; i + 1 < releases.size(); ++i) {
    releases[i].erase();
  }
  return true;
}

// The candidate set is pre-collected for two reasons: an acquire on a
// different DFB that `detectGroup` walked past as a non-member must still
// be considered as the starting point of a separate group later; and the
// outer iteration must not depend on `getNextNode()` after the rewrite
// erases ops in place.
template <typename AcquireOp, typename ReleaseOp>
static void coalesceInBlock(Block &block, OpBuilder &builder) {
  SmallVector<AcquireOp> candidates;
  for (Operation &op : block) {
    if (auto acquire = dyn_cast<AcquireOp>(&op)) {
      candidates.push_back(acquire);
    }
  }
  DenseSet<Operation *> erased;
  for (AcquireOp leader : candidates) {
    Operation *leaderOp = leader.getOperation();
    if (erased.contains(leaderOp)) {
      continue;
    }
    if (leader.getNumTiles().has_value()) {
      continue;
    }
    SmallVector<AcquireOp> group = detectGroup<AcquireOp>(leader);
    if (group.size() < 2) {
      continue;
    }
    if (tryCoalesceGroup<AcquireOp, ReleaseOp>(group, builder)) {
      for (AcquireOp member : group) {
        erased.insert(member.getOperation());
      }
    }
  }
}

struct TTLCoalesceDFBAcquiresPass
    : public impl::TTLCoalesceDFBAcquiresBase<TTLCoalesceDFBAcquiresPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    func.walk([&](Block *block) {
      if (block->empty()) {
        return;
      }
      coalesceInBlock<CBWaitOp, CBPopOp>(*block, builder);
      coalesceInBlock<CBReserveOp, CBPushOp>(*block, builder);
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
