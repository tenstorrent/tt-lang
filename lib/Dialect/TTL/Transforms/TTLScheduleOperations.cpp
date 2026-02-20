// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Schedule Operations Pass
//===----------------------------------------------------------------------===//
//
// Reorders tile operations within sync regions (acquire -> commit) to group by
// operation kind. This enables init consolidation in the subsequent
// convert-ttl-to-ttkernel + ttkernel-consolidate-inits passes.
//
// After lower-to-loops unrolls tile loops for DST maximization, each tile
// iteration produces interleaved ops of different kinds:
//
//   copy(0); add(0); mul(0); exp(0); copy(1); add(1); mul(1); exp(1); ...
//
// This pass reorders to group ops at the same dependency level:
//
//   copy(0); copy(1); add(0); add(1); mul(0); mul(1); exp(0); exp(1); ...
//
// The reordering respects both SSA data dependencies AND DST register hazards.
// SSA only captures RAW (true) dependencies. When compute chains reuse the same
// DST slot (e.g., copy b -> dst1, consume dst1, then copy c -> dst1), we must
// also track anti-dependencies (WAR) and output dependencies (WAW) to prevent
// the scheduler from moving a write before a prior read of the same slot.
// In principle, DST index renaming (re-running assign-dst) could eliminate
// these false dependencies, but DST capacity is small (8 bf16 / 4 f32 slots)
// and already fully utilized by subblocking, so we conservatively respect them
// here.
//
// For each op, we compute its dependency depth considering:
//   - RAW (Read-After-Write): captured by SSA def-use chains
//   - WAW (Write-After-Write): two ops writing the same DST index must maintain
//     their original order
//   - WAR (Write-After-Read): an op writing to a DST index must come after all
//     prior reads of that index (since the write would clobber the value)
//
// Ops at the same depth are independent and can be freely reordered. Within
// each depth level, ops are sorted by (category, typeId, dstIdx) to maximize
// grouping.
//
// Stores are not reordered (they are already after the commit/wait boundary,
// placed by reorderStoresAfterSync in lower-to-loops).
//
// References:
//   - Hennessy & Patterson, "Computer Architecture: A Quantitative Approach",
//     Chapter 3 (ILP): defines RAW, WAR, WAW data hazards through shared
//     registers. The DST register file is analogous to a hardware register
//     file.
//   - Cooper & Torczon, "Engineering a Compiler", Chapter 12 (Instruction
//     Scheduling): list scheduling using a data-precedence graph with
//     topological depth levels -- the same algorithm used here.
//   - Benoit de Dinechin & Sid Touati, "Advanced Backend Optimization", Part 2:
//   Instruction scheduling
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

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

/// Sort key for a tile operation within a sync region.
struct TileOpSortKey {
  unsigned depthLevel;
  TileOpCategory category;
  mlir::TypeID typeId;
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
    // Tertiary: TypeID (groups identical op types for init sharing).
    if (typeId != other.typeId) {
      return typeId.getAsOpaquePointer() < other.typeId.getAsOpaquePointer();
    }
    // Quaternary: dst_idx for deterministic ordering.
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

/// Compute the dependency depth of each tile op. The depth is the length of
/// the longest path through predecessors, considering:
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
    bool hasPred = false;

    // RAW dependencies (SSA def-use chains).
    for (Value operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        if (tileOpSet.contains(defOp)) {
          hasPred = true;
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
        hasPred = true;
        maxPredLevel = std::max(maxPredLevel, levels[it->second] + 1);
      }
      // WAR: must come after all readers of the previous value at this index.
      if (auto it = pendingReaders.find(writeIdx); it != pendingReaders.end()) {
        for (Operation *reader : it->second) {
          if (reader != op) {
            hasPred = true;
            maxPredLevel = std::max(maxPredLevel, levels[reader] + 1);
          }
        }
      }
      // Update tracking: new writer, clear pending readers.
      lastWriter[writeIdx] = op;
      pendingReaders[writeIdx].clear();
    }

    levels[op] = hasPred ? maxPredLevel : 0;
  }
  return levels;
}

/// Process a single sync region: reorder tile ops between acquire and commit.
static void scheduleOpsInRegion(llvm::SmallVectorImpl<Operation *> &tileOps) {
  if (tileOps.size() <= 1) {
    return;
  }

  // Compute dependency levels.
  auto levels = computeDepthLevels(tileOps);

  // Build sort keys.
  llvm::SmallVector<TileOpSortKey, 16> keys;
  keys.reserve(tileOps.size());
  for (auto [i, op] : llvm::enumerate(tileOps)) {
    keys.push_back({levels[op], classifyTileOp(op), op->getName().getTypeID(),
                    getDstIdx(op), static_cast<unsigned>(i), op});
  }

  // Sort by (depth, category, typeID, dst_idx, originalPosition).
  llvm::sort(keys);

  // Check if already in order (avoid unnecessary moves).
  bool alreadySorted = true;
  for (unsigned i = 0; i < keys.size(); ++i) {
    if (keys[i].originalPosition != i) {
      alreadySorted = false;
      break;
    }
  }
  if (alreadySorted) {
    return;
  }

  // Reposition ops using moveBefore. Place each op before the first
  // non-tile-op after the region, maintaining sorted order.
  Operation *insertionPoint = tileOps.back()->getNextNode();

  for (auto &key : keys) {
    key.op->moveBefore(insertionPoint);
  }
}

struct TTLScheduleOperationsPass
    : public impl::TTLScheduleOperationsBase<TTLScheduleOperationsPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    funcOp.walk([](Block *block) {
      llvm::SmallVector<Operation *, 16> currentRegionOps;
      bool inSyncRegion = false;

      for (Operation &op : *block) {
        // Detect acquire (start of sync region).
        if (isa<TileRegsAcquireOp>(op)) {
          inSyncRegion = true;
          currentRegionOps.clear();
          continue;
        }

        // Detect commit (end of sync region) - schedule collected ops.
        if (isa<TileRegsCommitOp>(op)) {
          if (inSyncRegion) {
            scheduleOpsInRegion(currentRegionOps);
          }
          inSyncRegion = false;
          currentRegionOps.clear();
          continue;
        }

        if (!inSyncRegion) {
          continue;
        }

        // Collect tile ops (classified ops only).
        TileOpCategory cat = classifyTileOp(&op);
        if (cat != TileOpCategory::Unknown) {
          currentRegionOps.push_back(&op);
        }
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
