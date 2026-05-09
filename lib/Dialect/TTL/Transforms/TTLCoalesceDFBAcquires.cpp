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

// Ops permitted to interleave between consecutive acquires without breaking
// a coalescable group. Verified empirically against
// `test/ttlang/Dialect/TTL/Transforms/insert_cb_sync.mlir:812-817`: the
// frontend emits `cb_wait` immediately followed by `attach_cb`, so a
// three-wait group has six interleaved ops.
static bool isInterleaveOk(Operation *op) {
  return isa<AttachCBOp, arith::ConstantOp>(op);
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

// Detect a group of N >= 1 strictly-consecutive same-CB acquires of kind
// `AcquireOp` starting at `start`. Returns the group; an acquire that
// already carries a `num_tiles` attribute terminates the group (it has
// already been coalesced or was emitted by `TTLSubblockComputeForDST`).
template <typename AcquireOp>
static SmallVector<AcquireOp> detectGroup(AcquireOp start) {
  SmallVector<AcquireOp> group;
  group.push_back(start);
  Value cb = start.getCb();
  for (Operation *cur = start->getNextNode(); cur; cur = cur->getNextNode()) {
    if (auto next = dyn_cast<AcquireOp>(cur)) {
      if (next.getCb() == cb && !next.getNumTiles().has_value()) {
        group.push_back(next);
        continue;
      }
      break; // Same-kind acquire on different CB or already coalesced.
    }
    if (!isInterleaveOk(cur)) {
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

// Walk `block` once, applying coalescing to consecutive acquires.
template <typename AcquireOp, typename ReleaseOp>
static void coalesceInBlock(Block &block, OpBuilder &builder) {
  Operation *op = &block.front();
  while (op) {
    Operation *next = op->getNextNode();
    if (auto acquire = dyn_cast<AcquireOp>(op)) {
      if (!acquire.getNumTiles().has_value()) {
        SmallVector<AcquireOp> group = detectGroup<AcquireOp>(acquire);
        if (group.size() >= 2) {
          // Capture the resume point before the rewrite; the last group
          // member is erased but the op after it (if any) remains valid.
          Operation *resume = group.back()->getNextNode();
          if (tryCoalesceGroup<AcquireOp, ReleaseOp>(group, builder)) {
            op = resume;
            continue;
          }
        }
      }
    }
    op = next;
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
