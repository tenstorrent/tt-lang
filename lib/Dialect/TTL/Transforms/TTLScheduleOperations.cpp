// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Schedule Operations Pass
//===----------------------------------------------------------------------===//
//
// This file implements tile operation scheduling within DST sync regions.
// See the ttl-schedule-operations pass description in Passes.td.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-schedule-operations"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLSCHEDULEOPERATIONS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Extract dst_idx attribute value from an operation, or return max int64
/// for deterministic ordering of ops without dst_idx.
static int64_t getDstIdx(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
    return attr.getInt();
  }
  return std::numeric_limits<int64_t>::max();
}

/// Compute an init-affinity key for sub-sorting within the same op type.
/// Ops with the same affinity share one init call; different affinities
/// require re-init. This groups, e.g., all COL bcasts before ROW bcasts.
static int64_t getInitAffinity(Operation *op) {
  // TileBcastOp: bcast type (Col=1, Row=2, Scalar=3) determines init.
  if (auto bcast = dyn_cast<TileBcastOp>(op)) {
    return static_cast<int64_t>(bcast.getBcastType());
  }
  // CopyTileOp: group by input CB so copies from the same CB stay adjacent
  // (avoiding redundant copy_tile_init re-inits). After loop lowering, the
  // source is a scalar tile from tensor.extract — trace through it to reach
  // the underlying tensor, then find the attached CB.
  if (auto copy = dyn_cast<CopyTileOp>(op)) {
    Value src = copy.getSrc();
    // Trace through tensor.extract to get the source tensor.
    if (auto extract = src.getDefiningOp<mlir::tensor::ExtractOp>()) {
      src = extract.getTensor();
    }
    if (auto cb = getAttachedCB(src)) {
      if (auto bindCb = cb.getDefiningOp<BindCBOp>()) {
        return bindCb.getCbIndex().getSExtValue();
      }
    }
  }
  return 0;
}

/// Sort key for a tile operation within a sync region.
struct TileOpSortKey {
  unsigned depthLevel;
  TileOpCategory category;
  llvm::StringRef opName;
  int64_t initAffinity;
  int64_t dstIdx;
  unsigned originalPosition;
  Operation *op;

  bool operator<(const TileOpSortKey &other) const {
    // Primary: dependency depth (must come first for correctness).
    if (depthLevel != other.depthLevel) {
      return depthLevel < other.depthLevel;
    }
    // Secondary: category (groups CopyTile before FPU before SFPU).
    if (category != other.category) {
      return static_cast<uint8_t>(category) <
             static_cast<uint8_t>(other.category);
    }
    // Tertiary: op name (groups identical op types for init sharing).
    // Uses string comparison for deterministic ordering across runs.
    if (opName != other.opName) {
      return opName < other.opName;
    }
    // Quaternary: init affinity (groups ops sharing one init call,
    // e.g., COL bcasts vs ROW bcasts, or copies from different CBs).
    if (initAffinity != other.initAffinity) {
      return initAffinity < other.initAffinity;
    }
    // Quinary: dst_idx for deterministic ordering.
    if (dstIdx != other.dstIdx) {
      return dstIdx < other.dstIdx;
    }
    // Stable sort: preserve original order for ties.
    return originalPosition < other.originalPosition;
  }
};

/// Get the DST indices that an op reads from. CopyTileOp reads from a CB (not
/// DST), so it has no DST read indices. All other tile ops read from DST slots
/// determined by their SSA operands' defining ops.
static llvm::SmallVector<int64_t, 2>
getReadDstIndices(Operation *op, const llvm::DenseSet<Operation *> &tileOpSet) {
  llvm::SmallVector<int64_t, 2> indices;

  // CopyTile reads from CB, not DST.
  if (isa<CopyTileOp>(op)) {
    return indices;
  }

  // Trace SSA operands to find their defining tile ops' DST indices.
  for (Value operand : op->getOperands()) {
    if (auto *defOp = operand.getDefiningOp()) {
      if (tileOpSet.contains(defOp)) {
        int64_t idx = getDstIdx(defOp);
        if (idx != std::numeric_limits<int64_t>::max()) {
          indices.push_back(idx);
        }
      }
    }
  }
  return indices;
}

/// Compute the dependency depth of each tile op. Assumes tileOps are in
/// original block order (used for WAW/WAR "most recent" tracking).
///
/// The depth is the length of the longest path through predecessors,
/// considering:
///   - RAW (Read-After-Write): via SSA def-use chains
///   - WAW (Write-After-Write): ops writing the same DST index
///   - WAR (Write-After-Read): a write must come after prior reads of that DST
///
/// DST register hazards matter because multiple tile iterations may reuse the
/// same DST slot (e.g., copy b -> dst1, use dst1, then copy c -> dst1).
/// Without WAR tracking, the scheduler could move the second copy before the
/// consumer of the first, clobbering the value.
static llvm::DenseMap<Operation *, unsigned>
computeDepthLevels(llvm::ArrayRef<Operation *> tileOps) {
  llvm::DenseSet<Operation *> tileOpSet(tileOps.begin(), tileOps.end());
  llvm::DenseMap<Operation *, unsigned> levels;

  // Track DST register hazards.
  // lastWriter[i]: the most recent op that wrote to DST[i].
  // pendingReaders[i]: ops that read DST[i] since the last writer.
  llvm::DenseMap<int64_t, Operation *> lastWriter;
  llvm::DenseMap<int64_t, llvm::SmallVector<Operation *, 4>> pendingReaders;

  for (auto *op : tileOps) {
    unsigned maxPredLevel = 0;

    // RAW dependencies (SSA def-use chains).
    for (Value operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        if (tileOpSet.contains(defOp)) {
          maxPredLevel = std::max(maxPredLevel, levels[defOp] + 1);
        }
      }
    }

    // Determine DST indices this op reads from and writes to.
    auto readIndices = getReadDstIndices(op, tileOpSet);
    int64_t writeIdx = getDstIdx(op);

    // Register reads for WAR tracking.
    for (int64_t ri : readIndices) {
      pendingReaders[ri].push_back(op);
    }

    // WAW + WAR dependencies for the written DST index.
    if (writeIdx != std::numeric_limits<int64_t>::max()) {
      // WAW: must come after the previous writer to this DST index.
      if (auto it = lastWriter.find(writeIdx); it != lastWriter.end()) {
        maxPredLevel = std::max(maxPredLevel, levels[it->second] + 1);
      }
      // WAR: must come after all readers of the previous value at this index.
      if (auto it = pendingReaders.find(writeIdx); it != pendingReaders.end()) {
        for (Operation *reader : it->second) {
          if (reader != op) {
            maxPredLevel = std::max(maxPredLevel, levels[reader] + 1);
          }
        }
      }
      // Update tracking: new writer, clear pending readers.
      lastWriter[writeIdx] = op;
      pendingReaders[writeIdx].clear();
    }

    levels[op] = maxPredLevel;
  }
  return levels;
}

/// Process a single sync region: reorder tile ops between acquire and commit.
static void scheduleOpsInRegion(ArrayRef<Operation *> tileOps) {
  if (tileOps.size() <= 1) {
    return;
  }

  // Compute dependency levels.
  auto levels = computeDepthLevels(tileOps);

  // Build sort keys.
  llvm::SmallVector<TileOpSortKey, 16> keys;
  keys.reserve(tileOps.size());
  for (auto [i, op] : llvm::enumerate(tileOps)) {
    keys.push_back({levels[op], classifyTileOp(op),
                    op->getName().getStringRef(), getInitAffinity(op),
                    getDstIdx(op), static_cast<unsigned>(i), op});
  }

  // Skip sort and IR mutation if already in order.
  if (llvm::is_sorted(keys)) {
    return;
  }

  llvm::sort(keys);

  LLVM_DEBUG({
    llvm::dbgs() << "Scheduled " << keys.size() << " ops in sync region:\n";
    for (auto &key : keys) {
      llvm::dbgs() << "  depth=" << key.depthLevel
                   << " cat=" << static_cast<unsigned>(key.category)
                   << " dst=" << key.dstIdx << " " << *key.op << "\n";
    }
  });

  // Reposition ops using moveBefore. Place each op before the first
  // non-tile-op after the region, maintaining sorted order.
  Operation *insertionPoint = tileOps.back()->getNextNode();
  assert(insertionPoint && "expected commit op after tile ops in sync region");

  for (auto &key : keys) {
    key.op->moveBefore(insertionPoint);
  }
}

struct TTLScheduleOperationsPass
    : public impl::TTLScheduleOperationsBase<TTLScheduleOperationsPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    funcOp.walk([](DstSectionOp dstSection) {
      SmallVector<Operation *, 16> mathOps;
      for (Operation &op : dstSection.getBody().front().without_terminator()) {
        if (isa<TileStoreOp>(op)) {
          break;
        }
        TileOpCategory cat = classifyTileOp(&op);
        if (cat != TileOpCategory::Unknown) {
          mathOps.push_back(&op);
        }
      }
      if (!mathOps.empty()) {
        scheduleOpsInRegion(mathOps);
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
