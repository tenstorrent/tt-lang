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
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
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

// Applies one planned group: the members become slices of one multi-block
// acquisition at the leader, and the last replaced release carries the merged
// tile count.
template <typename AcquireOp, typename ReleaseOp>
static void applyCoalescedGroup(const CoalescedAcquireGroup &group,
                                OpBuilder &builder) {
  auto leader = cast<AcquireOp>(group.acquires.front());
  auto unitTy = cast<RankedTensorType>(leader.getResult().getType());
  int64_t k = unitTy.getShape()[1];
  int64_t totalTiles = static_cast<int64_t>(group.acquires.size()) * k;

  builder.setInsertionPoint(leader);
  IntegerAttr numTilesAttr = builder.getI64IntegerAttr(totalTiles);
  AcquireOp coalesced = AcquireOp::create(
      builder, leader.getLoc(), buildCoalescedType(unitTy, totalTiles),
      leader.getCb(), numTilesAttr);
  for (auto [index, member] : llvm::enumerate(group.acquires)) {
    auto old = cast<AcquireOp>(member);
    builder.setInsertionPoint(old);
    auto slice =
        createPerBlockSlice(builder, old.getLoc(), coalesced.getResult(),
                            unitTy, static_cast<int64_t>(index), k);
    old.getResult().replaceAllUsesWith(slice.getResult());
    old.erase();
  }
  group.releases.back()->setAttr("num_tiles", numTilesAttr);
  for (Operation *release : ArrayRef(group.releases).drop_back()) {
    cast<ReleaseOp>(release).erase();
  }
}

struct TTLCoalesceDFBAcquiresPass
    : public impl::TTLCoalesceDFBAcquiresBase<TTLCoalesceDFBAcquiresPass> {
  using impl::TTLCoalesceDFBAcquiresBase<
      TTLCoalesceDFBAcquiresPass>::TTLCoalesceDFBAcquiresBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());
    // With sync-user-dfbs=false only compiler-created DFBs are coalesced.
    auto isEligible = [&](const CoalescedAcquireGroup &group) {
      Value dfb = group.acquires.front()->getOperand(0);
      return syncUserDFBs || !isUserManagedDFB(dfb);
    };

    func.walk([&](Block *block) {
      if (block->empty()) {
        return;
      }
      for (const CoalescedAcquireGroup &group : planCoalescedAcquireGroups(
               *block, DFBAcquireReleaseKind::Consumer)) {
        if (isEligible(group)) {
          applyCoalescedGroup<CBWaitOp, CBPopOp>(group, builder);
        }
      }
      for (const CoalescedAcquireGroup &group : planCoalescedAcquireGroups(
               *block, DFBAcquireReleaseKind::Producer)) {
        if (isEligible(group)) {
          applyCoalescedGroup<CBReserveOp, CBPushOp>(group, builder);
        }
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
