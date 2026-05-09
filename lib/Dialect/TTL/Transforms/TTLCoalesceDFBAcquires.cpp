// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Coalesce DFB Acquires
//===----------------------------------------------------------------------===//
//
// Rewrites N consecutive same-DFB acquires + N matching releases into the
// canonical tt-metal cumulative-wait shape:
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
// See issue #556 and `docs/development/DFBManagement.md`.
//===----------------------------------------------------------------------===//

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

// Return true if `op` (sitting between two same-DFB acquires on `cb`) might
// directly or transitively cause a release on `cb` before our coalesced
// release executes -- i.e., it must terminate the candidate group. See
// "DFB Acquire Coalescing" in `docs/development/DFBManagement.md` for the
// correctness argument. Two locally-checkable conditions cover the cases
// that matter:
//
//   1. The op operates on `cb` itself (uses `cb` as an operand) -- includes
//      same-DFB releases (cb_pop / cb_push) and any other op that touches
//      `cb` directly.
//   2. The op consumes the SSA result of an in-progress group member,
//      since that consume can flow into a release on `cb` somewhere
//      downstream.
//
// Region-bearing ops are treated as opaque (terminate the group) because
// their bodies might contain a release on `cb`.
//
// `ttl.attach_cb` is an SSA-only identity (lowering erases it) that always
// references the group's results and `cb`; allow it explicitly.
static bool mayReleaseDFB(Operation *op, Value cb,
                          ArrayRef<Operation *> group) {
  if (isa<AttachCBOp>(op)) {
    return false;
  }
  if (op->getNumRegions() > 0) {
    return true;
  }
  for (Value operand : op->getOperands()) {
    if (operand == cb) {
      return true;
    }
    for (Operation *member : group) {
      if (operand == member->getResult(0)) {
        return true;
      }
    }
  }
  return false;
}

// Build the coalesced acquire's result type. For the common rank-2 case
// `tensor<1 x k x elem>` (matching the `num_tiles` shape convention from
// `cb_ops_invalid.mlir` and `TTLSubblockComputeForDST`), produce
// `tensor<1 x (N*k) x elem>`. Higher-rank shapes are not coalesced.
static RankedTensorType buildCoalescedType(RankedTensorType unitTy,
                                           int64_t totalTiles) {
  auto shape = unitTy.getShape();
  assert(shape.size() == 2 && shape[0] == 1 &&
         "coalesce expects rank-2 acquire with leading 1");
  return RankedTensorType::get({1, totalTiles}, unitTy.getElementType());
}

// `tensor.extract_slice` for the i-th member of an N-block group:
// offsets = [0, i*k], sizes = [1, k], strides = [1, 1].
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

// Detect a group of same-CB acquires of kind `AcquireOp` starting at
// `start`. The group is maximal: walks forward in the block, adding each
// same-kind same-cb acquire (with no pre-existing `num_tiles`) and skipping
// any op that doesn't touch `cb` or the group's results (per
// `mayReleaseDFB`). An acquire that already carries `num_tiles` (already
// coalesced or set by `TTLSubblockComputeForDST`) terminates the group.
template <typename AcquireOp>
static SmallVector<AcquireOp> detectGroup(AcquireOp start) {
  SmallVector<AcquireOp> group;
  group.push_back(start);
  Value cb = start.getCb();
  SmallVector<Operation *> groupOps = {start.getOperation()};
  for (Operation *cur = start->getNextNode(); cur; cur = cur->getNextNode()) {
    if (auto next = dyn_cast<AcquireOp>(cur)) {
      if (next.getCb() == cb) {
        if (next.getNumTiles().has_value()) {
          break;
        }
        group.push_back(next);
        groupOps.push_back(cur);
        continue;
      }
      // Different-CB acquire of the same kind -- doesn't touch our cb or
      // our group's results; skip past.
    }
    if (mayReleaseDFB(cur, cb, groupOps)) {
      break;
    }
  }
  return group;
}

// Collect the first `count` matching releases of kind `ReleaseOp` on `cb`
// starting at `start`, walking forward in the same block. Returns empty if
// fewer than `count` are found before block end, or if a same-CB release
// already carries `num_tiles` (a partial earlier coalesce we shouldn't
// extend).
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
  // Conservative: only coalesce the rank-2 leading-1 shape that the
  // existing `num_tiles` convention covers. Other shapes flow through
  // unchanged.
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

// Apply coalescing to acquires of kind `AcquireOp` in `block`. Pre-collects
// the candidate set so that other-CB acquires which `detectGroup` skips
// past still get a chance to lead their own group on a later iteration --
// we don't rely on traversing erased ops via `getNextNode()`.
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
